import torch
import os
import wandb
from tqdm import tqdm
from loguru import logger
import numpy as np
import argparse

from model import cspdarknet
from dataloader import InfiniteDataLoader
from dataset import ClassificationDataset
from general import init_seeds, generate_hash, forced_load
from model_loss import create_optimizer, create_scheduler, EarlyStop
from ema import ModelEMA
from copy import deepcopy


@torch.no_grad()
def val(model, class_names, val_loader, half=True, debug_mode=False):
    # Initialize/load model and set device
    device = next(model.parameters()).device
    half &= device.type != "cpu"  # half precision only supported on CUDA
    model.half() if half else model.float()

    # Configure
    model.eval()
    cuda = device.type != "cpu"
    s = ("%20s" + "%11s") % ("Class", "Accuracy")
    pbar = tqdm(val_loader, desc=s, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")  # progress bar
    nc = len(class_names)

    out_t = np.zeros(nc)
    out_f = np.zeros(nc)

    for im, targets in pbar:
        if cuda:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32

        # Inference
        out = model(im)  # inference outputs

        # Metrics
        for i, pred in enumerate(out):
            pred = torch.argmax(pred)
            gt = targets[i]
            if pred == gt:
                out_t[gt] += 1
            else:
                out_f[gt] += 1
        if debug_mode:
            break

    # Compute metrics
    pf = "%20s" + "%11.3g"  # print format
    acc = out_t.sum() / (out_t.sum() + out_f.sum())
    print(pf % ("all", acc))
    for i in range(nc):
        cls_acc = out_t[i] * 1.0 / (out_t[i] + out_f[i])
        print(pf % (class_names[i], cls_acc))

    # Return results
    model.float()  # for training
    return acc


def train(
    project_name,
    debug_mode,
    max_epochs,
    loss_titles,
    device_id,
    trainset_path,
    valset_path,
    batch_size,
    image_size,
    class_names,
    model_name,
    save_dir,
    pretrained_pt_path,
    autocast_enabled,
    optimizer_type,
    lr0,
    lrf,
    momentum,
    weight_decay,
    warmup_epochs,
    warmup_momentum,
    warmup_bias_lr,
    nbs,
    cos_lr,
    patience,
    ema_enabled,
    wandb_enabled,
    early_stop,
    raw_args={},
):
    # init random seeds
    init_seeds()

    # create device
    device = torch.device(device_id)
    print(f"set device to {device.type}[{device.index}]")
    num_workers = os.cpu_count()
    autocast_enabled &= device.type != "cpu"

    # create loaders
    assert trainset_path is not None
    assert valset_path is not None
    train_loader = InfiniteDataLoader(
        dataset=ClassificationDataset(
            training=True,
            dataset_path=trainset_path,
            image_size=image_size,
        ),
        training=True,
        batch_size=batch_size,
        collate_fn=ClassificationDataset.collate_fn,
        num_workers=num_workers,
    )
    val_loader = InfiniteDataLoader(
        dataset=ClassificationDataset(
            training=False,
            dataset_path=valset_path,
            image_size=image_size,
        ),
        training=False,
        batch_size=batch_size,
        collate_fn=ClassificationDataset.collate_fn,
        num_workers=num_workers,
    )

    # rescale hyps
    best_fitness = 0.0
    nb = len(train_loader)
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    weight_decay = weight_decay * accumulate / nbs  # scale weight_decay
    last_opt_step = -1
    nw = max(round(warmup_epochs * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)

    # create model
    model = cspdarknet(num_classes=len(class_names))
    if pretrained_pt_path is not None:
        model = forced_load(model, pretrained_pt_path)
    model = model.to(device)

    # create optimizer
    optimizer = create_optimizer(model, optimizer_type, lr0, momentum, weight_decay)
    scheduler, lf = create_scheduler(optimizer, lrf, max_epochs, cos_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type != "cpu"))
    if early_stop:
        early_stopper = EarlyStop(patience=patience)
    if ema_enabled:
        ema = ModelEMA(model)

    # create loss
    loss_func = torch.nn.CrossEntropyLoss()

    # create save directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # create wandb logger
    wandb_enabled &= not debug_mode
    if wandb_enabled:
        wandb.init(project=project_name, dir=save_dir, mode="offline")
        wandb.watch(model)
        wandb.run.log_code("./", include_fn=lambda path: path.endswith(".py"))
        wandb.config.update(raw_args)
        logger.success("Wandb logger created")

    # start training loop
    for epoch in range(max_epochs):
        mloss = torch.zeros(len(loss_titles), device=device)  # mean losses

        # Set progress bar
        pbar = enumerate(train_loader)
        print(("\n" + "%10s" * (4 + len(loss_titles))) % ("Epoch", "gpu_mem", *loss_titles, "labels", "img_size"))
        pbar = tqdm(pbar, total=nb, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")  # progress bar

        model.train()  # set to training mode
        optimizer.zero_grad()

        for i, (imgs, targets) in pbar:
            imgs: torch.Tensor
            targets: torch.Tensor

            # warmup_step
            ni = i + nb * epoch  # number integrated batches (since train start)
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(ni, xi, [warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [warmup_momentum, momentum])

            # forward step
            imgs = imgs.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                out = model(imgs)
                loss = loss_func(out, targets.to(device))  # forward
                loss_items = loss.detach().unsqueeze(0)

            # check nan
            if any(torch.isnan(loss_items)):
                logger.warning(f"nan value found in loss: {loss_items}")
                continue

            # backward step
            if autocast_enabled:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= accumulate:
                if autocast_enabled:
                    scaler.unscale_(optimizer)  # unscale gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                    scaler.step(optimizer)  # optimizer.step
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                    optimizer.step()
                optimizer.zero_grad()
                if ema_enabled:
                    ema.update(model)
                last_opt_step = ni

            # log training info
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = f"{torch.cuda.memory_reserved(device) / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
            pbar.set_description(("%10s" * 2 + "%10.4g" * (2 + len(loss_items))) % (f"{epoch+1}/{max_epochs}", mem, *mloss, targets.shape[0], imgs.shape[-1]))

            # quit epoch loop in debug_mode
            if debug_mode:
                break

        # after an epoch --------------
        scheduler.step()

        # update best mAP
        model_for_val = model if not ema_enabled else ema.ema
        acc = val(model_for_val, class_names, val_loader, half=autocast_enabled, debug_mode=debug_mode)

        # update wandb
        if wandb_enabled:
            for metric, value in zip([f"loss_{ln}" for ln in loss_titles], mloss):
                wandb.log({metric: value.detach().item()})
            wandb.log({"Accuracy": acc})

        # save best model pt
        fi = acc
        if fi > best_fitness:
            best_fitness = fi
        if best_fitness == fi:
            if ema_enabled:
                torch.save(deepcopy(ema.ema).state_dict(), f"{save_dir}/{model_name}.pt")
            else:
                torch.save(deepcopy(model).state_dict(), f"{save_dir}/{model_name}.pt")
        torch.save(deepcopy(model).state_dict(), f"{save_dir}/{model_name}_latest.pt")

        # check early stop
        if early_stop and early_stopper(epoch=epoch, fitness=fi):
            break

    # finish training
    if wandb_enabled:
        wandb.finish()  # Marks a run as finished, and finishes uploading all data.

    torch.cuda.empty_cache()
    logger.success("CUDA memory successfully recycled")


if __name__ == "__main__":
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser()

    # 添加命令行参数
    parser.add_argument("--project_name", type=str, default="my_classification", help="项目名称")
    parser.add_argument("--debug_mode", action="store_true", help="是否开启debug模式(开启后每epoch均只运行1个iter)")
    parser.add_argument("--max_epochs", type=int, default=100, help="最大训练epoch数")
    parser.add_argument("--loss_titles", nargs="+", default=["cls"], help="默认损失函数项")
    parser.add_argument("--device_id", type=str, default="mps", help="设备类型: cuda, mps, cpu, 0, 1, ...(显卡device id)")
    parser.add_argument("--trainset_path", type=str, default="/nfs/datasets/classficiation_2_classes_eb__non_eb_20240602", help="训练集images文件夹位置, 会根据images->labels规则寻找标签文件夹")
    parser.add_argument("--valset_path", type=str, default="/nfs/datasets/classficiation_2_classes_eb__non_eb_20240602", help="测试集文件夹位置")
    parser.add_argument("--batch_size", type=int, default=3, help="批处理大小")
    parser.add_argument("--image_size", type=int, default=640, help="图像尺寸")
    parser.add_argument("--class_names", nargs="+", default=ClassificationDataset.class_names, help="类别名称")
    parser.add_argument("--model_name", type=str, default=generate_hash(), help="模型名称, 默认生成随机hash")
    parser.add_argument("--save_dir", type=str, default="./runs", help="日志&权重存储路径")
    parser.add_argument("--pretrained_pt_path", type=str, default=None, help="预训练模型路径")
    parser.add_argument("--autocast_enabled", action="store_true", help="是否启用autocast, 目前有已知的精度issue")
    parser.add_argument("--optimizer_type", type=str, default="SGD", help="优化器选项, 可选: Adam, AdamW, RMSProp, SGD")
    parser.add_argument("--lr0", type=float, default=0.01, help="初始学习率")
    parser.add_argument("--lrf", type=float, default=0.01, help="最终学习率")
    parser.add_argument("--momentum", type=float, default=0.937, help="动量")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="权重衰减")
    parser.add_argument("--warmup_epochs", type=float, default=3.0, help="warmup轮数")
    parser.add_argument("--warmup_momentum", type=float, default=0.8, help="warmup动量")
    parser.add_argument("--warmup_bias_lr", type=float, default=0.1, help="warmup学习率")
    parser.add_argument("--nbs", type=int, default=64, help="nominal batch size")
    parser.add_argument("--cos_lr", action="store_true", help="是否使用cosine lr scheduler")
    parser.add_argument("--patience", type=int, default=30, help="earlystop触发轮数")
    parser.add_argument("--ema_enabled", action="store_true", help="是否启用Model EMA平均")
    parser.add_argument("--wandb_enabled", action="store_true", help="是否启用wandb日志")
    parser.add_argument("--early_stop", action="store_true", help="是否启用earlystop策略")

    # 解析命令行参数
    args = parser.parse_args()
    train(**vars(args), raw_args=args)
