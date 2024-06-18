import sys
import cv2


sys.path.append(".")

from dataset import ClassificationDataset  # noqa:E402
from canvas import Canvas  # noqa:E402


def visualize(dataset: ClassificationDataset):
    canvas = Canvas()
    for i in range(len(dataset)):
        image, cls = dataset.__getitem__(i)
        _, H, W = image.shape
        canvas.load(image)
        print(f'image[{i}] cls={cls}')
        canvas.show(wait_key=1)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break


if __name__ == "__main__":
    dataset = ClassificationDataset(
        training=True,
        dataset_path="/nfs/datasets/quaming_classification_3_class/val",
        image_size=640,
    )
    visualize(dataset)
