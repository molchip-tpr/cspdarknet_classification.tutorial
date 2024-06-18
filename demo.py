import cv2
import math
import torch
import numpy as np

from torchvision.transforms.functional import to_tensor
from model import cspdarknet


def im2tensor(image: np.ndarray):
    assert np.issubsctype(image, np.integer)  # 0~255 BGR int -> 0~1 RGB float
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return to_tensor(image)


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


if __name__ == "__main__":
    # image_path = "test_bicycle.jpg"
    image_path = "test_electric_bicycle.jpg"
    np_image = cv2.imread(image_path)
    torch_image = im2tensor(np_image)
    torch_image = letterbox_padding(torch_image).unsqueeze(0)

    model = cspdarknet(num_classes=3).eval()
    model.load_state_dict(torch.load("runs/00f9b9.pt"))
    with torch.no_grad():
        y = model(torch_image)
    print(y)
