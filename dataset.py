import cv2
import torch
import os
import random
from torch.utils.data import Dataset

from torchvision import transforms as T
from general import get_im_files
from transforms import augment_hsv
from PIL import Image


class ClassificationDataset(Dataset):
    class_names = ["bicycle", "electric_bicycle", "gastank"]  # 请按实际需要修改

    # Base Class For making datasets which are compatible with nano
    def __init__(
        self,
        training,
        dataset_path,
        image_size,
        degrees=15,  # image rotation (+/- deg)
        # perspective=0.0,  # image perspective (+/- fraction), range 0-0.001
        hsv_h=0.015,  # image HSV-Hue augmentation (fraction)
        hsv_s=0.7,  # image HSV-Saturation augmentation (fraction)
        hsv_v=0.4,  # image HSV-Value augmentation (fraction)
    ):
        self.training = training
        self.image_size = image_size

        self.im_files = []
        for i, class_name in enumerate(self.class_names):
            image_dir = f"{dataset_path}/{class_name}"
            assert os.path.exists(image_dir), image_dir
            im_files = get_im_files(image_dir)
            self.im_files += [(i, x) for x in im_files]
        if self.training:
            random.shuffle(self.im_files)

        # Custom transforms
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v

        # Torchvision tranforms
        if training:
            self.transform = T.Compose(
                [
                    T.RandomRotation(degrees=degrees),
                    # T.RandomPerspective(p=perspective),
                    T.RandomHorizontalFlip(),
                    T.Resize((image_size, image_size)),
                    T.ToTensor(),
                ],
            )
        else:
            self.transform = T.Compose(
                [
                    T.Resize((image_size, image_size)),
                    T.ToTensor(),
                ],
            )

    def resize_with_padding(self, image, target_size):
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

    def load_image_and_cls(self, index):
        cls, f = self.im_files[index]
        image = cv2.imread(f)  # BGR
        assert image is not None, f"Image Not Found {f}"
        to_pil = T.ToPILImage()
        image = to_pil(image)
        self.resize_with_padding(image, (self.image_size, self.image_size))
        return image, cls

    def __getitem__(self, index):
        # Load image
        image, cls = self.load_image_and_cls(index)

        # Custom augmentations
        # if self.training:
        #     # HSV color-space
        #     augment_hsv(image, hgain=self.hsv_h, sgain=self.hsv_s, vgain=self.hsv_v)

        # Torchvision augmentations
        image = self.transform(image)
        return image, cls

    def __len__(self):
        return len(self.im_files)

    @staticmethod
    def collate_fn(batch):
        im, label = zip(*batch)  # transposed
        return torch.stack(im, 0), torch.tensor(label)
