# cspdarknet分类网络训练示例

## 数据集格式要求
该模型仅针对自行车/电瓶车误检，如有负样本建议在第一阶段训练时加入，不要在这个模型内处理
- mydataset
    - train
        - class1名字
            - xxx.jpg
            - xxx.jpg
        - class2名字
            - xxx.jpg
            - xxx.jpg
        - ...
    - val
        - class1名字
            - xxx.jpg
            - xxx.jpg
        - class2名字
            - xxx.jpg
            - xxx.jpg
        - ...
具体内容需要和`dataset.py`中的class_names对应，请注意配置

## 训练模型
```shell
python trainer.py --device_id cuda \
--batch_size 16 \
--max_epochs 300 \
--ema_enabled \
--wandb_enabled \
--early_stop \
--trainset_path /nfs/datasets/mydataset/train \
--valset_path /nfs/datasets/mydataset/val
```
训练时会自动padding并等比缩放到(image_size, image_size)大小，如需配置，可指定`--image_size`

## 导出模型
```shell
python export.py --weight my_weight.pt --input_shape 1 3 640 640 --input_names image --opset_version 13 --enable_onnxsim
```

