import cv2
import math
import torch
import numpy as np
import os
import shutil
from torchvision import transforms as T
from PIL import Image
from torchvision.transforms.functional import to_tensor
from model import cspdarknet


def im2tensor(image: np.ndarray):
    assert np.issubsctype(image, np.integer)  # 0~255 BGR int -> 0~1 RGB float
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return to_tensor(image)


def tensor2im(x: torch.Tensor):
    assert isinstance(x, torch.Tensor)
    if len(x.shape) == 4:
        assert x.shape[0] == 1
        x = x.squeeze(0)
    assert len(x.shape) == 3  # 0~1 RGB float -> 0~255 BGR int
    np_img = (x * 255).int().numpy().astype(np.uint8)
    np_img = np_img[::-1].transpose((1, 2, 0))  # CHW to HWC, RGB to BGR, 0~1 to 0~255
    np_img = np.ascontiguousarray(np_img)
    return np_img


def letterbox_padding(frame: torch.Tensor, gs=32):
    if len(frame.shape) == 4:
        n, c, h, w = frame.shape
        if w % gs == 0 and h % gs == 0:
            return frame
        exp_h = math.ceil(h / gs) * gs
        exp_w = math.ceil(w / gs) * gs
        background = torch.zeros(n, c, exp_h, exp_w, device=frame.device)
        pad_h = (exp_h - h) // 2
        pad_w = (exp_w - w) // 2
        background[:, :, pad_h : pad_h + h, pad_w : pad_w + w] = frame  # noqa:E203
        return background
    elif len(frame.shape) == 3:
        c, h, w = frame.shape
        if w % gs == 0 and h % gs == 0:
            return frame
        exp_h = math.ceil(h / gs) * gs
        exp_w = math.ceil(w / gs) * gs
        background = torch.zeros(c, exp_h, exp_w, device=frame.device)
        pad_h = (exp_h - h) // 2
        pad_w = (exp_w - w) // 2
        background[:, pad_h : pad_h + h, pad_w : pad_w + w] = frame  # noqa:E203
        return background


def get_class(output, class_names):

    class_idx = output.argmax(dim=1).item()
    return class_names[class_idx]


def resize_with_padding(image, target_size):
    # 获取原始图像的尺寸
    original_width, original_height = image.size

    # 计算比例
    ratio = min(target_size[0] / original_width, target_size[1] / original_height)

    # 计算新的尺寸
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    # 调整图像大小
    resize_transform = T.Resize((new_height, new_width))
    resized_image = resize_transform(image)

    # 创建黑色背景
    new_image = Image.new("RGB", target_size)

    # 计算图像在新图像上的位置
    paste_position = ((target_size[0] - new_width) // 2, (target_size[1] - new_height) // 2)

    # 将调整大小后的图像粘贴到新图像上
    new_image.paste(resized_image, paste_position)

    return new_image


def classify_and_move(input_dir, output_dir2, mymodel, class_name, class_names):
    """
    遍历input_dir2中的图片,分类并移动到output_dir2下的对应文件夹
    Args:
        input_dir2 (str): 输入文件夹路径
        output_dir2 (str): 输出文件夹路径
    """
    # 加载模型
    num_classes0 = len(class_names)
    model = cspdarknet(num_classes=num_classes0).eval()
    model.load_state_dict(torch.load(mymodel))
    # 遍历输入文件夹
    input_dir = os.path.join(input_dir, class_name)
    for filename in os.listdir(input_dir):
        if filename.startswith("._"):
            continue
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".png") or filename.lower().endswith(".jpeg") or filename.lower().endswith(".bmp"):
            image_path = os.path.join(input_dir, filename)
            np_image = cv2.imread(image_path)
            cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
            to_pil = T.ToPILImage()
            image = to_pil(np_image)
            image = resize_with_padding(image, (640, 640))
            transform = T.Compose(
                [
                    T.Resize((640, 640)),
                    T.ToTensor(),
                ],
            )
            torch_image = transform(image)
            torch_image = letterbox_padding(torch_image).unsqueeze(0)

            with torch.no_grad():
                y = model(torch_image)  # 3个类别的置信度
            class_name = get_class(y, class_names)
            print(y, class_name)
            cv2.imshow("im", tensor2im(torch_image))
            cv2.waitKey(0)

            # # 复制图片到对应的输出文件夹
            # output_path3 = os.path.join(output_dir2, class_name, filename)
            # shutil.copy(image_path, output_path3)
            # print(f"Copy {filename} to {class_name} folder.")


if __name__ == "__main__":
    class_names = ["bicycle", "electric_bicycle", "gastank"]
    mymodel = "runs/ligui_0618.pt"

    # input_dir = "/home/test/yolov8_mc/cspdarknet-classification/data/val/"
    input_dir = "/nfs/datasets/quaming_classification_3_class/val"
    output_dir = "/home/test/yolov8_mc/cspdarknet-classification/data/classified"
    # for class_name in class_names:
    #     output_dir2 = output_dir + "_" + class_name
    #     for class_name in class_names:
    #         # 创建输出文件夹
    #         os.makedirs(os.path.join(output_dir2, class_name), exist_ok=True)
    for class_name in class_names:
        output_dir2 = output_dir + "_" + class_name
        classify_and_move(input_dir, output_dir2, mymodel, class_name, class_names)
