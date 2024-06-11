import math

import torch
import torch.nn as nn


class ConvModule(nn.Module):
    def __init__(self, c_in, c_out, k, s, p, act, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(c_out)
        if act == "relu6":
            self.relu = nn.ReLU6()
        elif act == "relu":
            self.relu = nn.ReLU()
        elif act == "leakyrelu":
            self.relu = nn.LeakyReLU()
        elif act == "hardswish":
            self.relu = nn.Hardswish()
        else:
            raise NotImplementedError(f"conv with activation={act} not implemented yet")
        self.fused = False

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def fused_forward(self, x):
        return self.relu(self.conv(x))

    def fuse(self):
        if self.fused:
            return self
        std = (self.bn.running_var + self.bn.eps).sqrt()
        bias = self.bn.bias - self.bn.running_mean * self.bn.weight / std

        t = (self.bn.weight / std).reshape(-1, 1, 1, 1)
        weights = self.conv.weight * t

        self.conv = nn.Conv2d(
            in_channels=self.conv.in_channels,
            out_channels=self.conv.out_channels,
            kernel_size=self.conv.kernel_size,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
            bias=True,
            padding_mode=self.conv.padding_mode,
        )
        self.conv.weight = torch.nn.Parameter(weights)
        self.conv.bias = torch.nn.Parameter(bias)
        self.forward = self.fused_forward
        self.fused = True
        return self


class DarknetBottleneck(nn.Module):
    def __init__(self, c_in, c_out, add=True, act=None):
        super().__init__()
        self.cv1 = ConvModule(c_in, int(0.5 * c_in), 3, 1, 1, act=act)
        self.cv2 = ConvModule(int(0.5 * c_in), c_out, 3, 1, 1, act=act)
        self.shortcut = add

    def forward(self, x):
        if self.shortcut:
            out = self.cv1(x)
            out = self.cv2(out)
            return x + out
        else:
            x = self.cv1(x)
            x = self.cv2(x)
            return x


class CSPLayer_2Conv(nn.Module):
    def __init__(self, c_in, c_out, add, n, act):
        super().__init__()
        half_out = int(0.5 * c_out)
        self.conv_in_left = ConvModule(c_in, half_out, 1, 1, 0, act=act)  # same result as split later
        self.conv_in_right = ConvModule(c_in, half_out, 1, 1, 0, act=act)  # same result as split later
        self.bottlenecks = nn.ModuleList()
        for _ in range(n):
            self.bottlenecks.append(DarknetBottleneck(half_out, half_out, add, act=act))
        self.conv_out = ConvModule(half_out * (n + 2), c_out, 1, 1, 0, act=act)

    def forward(self, x):
        x_left = self.conv_in_left(x)
        x_right = self.conv_in_right(x)  # main branch
        collection = [x_left, x_right]
        x = x_right
        for b in self.bottlenecks:
            x = b(x)
            collection.append(x)
        x = torch.cat(collection, dim=1)
        x = self.conv_out(x)
        return x


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c_in, k=5, act=None):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c_in // 2  # hidden channels
        self.cv1 = ConvModule(c_in, c_, 1, 1, 0, act=act)
        self.cv2 = ConvModule(c_ * 4, c_in, 1, 1, 0, act=act)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))


class Model(nn.Module):
    def __init__(self, num_classes=80, d=0.33, w=0.25, r=2.0, act="relu6"):
        super().__init__()
        _64xw = int(64 * w)
        _128xw = int(128 * w)
        _256xw = int(256 * w)
        _512xw = int(512 * w)
        _512xwxr = int(512 * w * r)
        _3xd = int(math.ceil(3 * d))
        _6xd = int(math.ceil(6 * d))
        self.stem_layer = ConvModule(3, _64xw, k=3, s=2, p=1, act=act)
        self.stage_layer_1 = nn.Sequential(
            ConvModule(_64xw, _128xw, k=3, s=2, p=1, act=act),
            CSPLayer_2Conv(_128xw, _128xw, add=True, n=_3xd, act=act),
        )
        self.stage_layer_2 = nn.Sequential(
            ConvModule(_128xw, _256xw, k=3, s=2, p=1, act=act),
            CSPLayer_2Conv(_256xw, _256xw, add=True, n=_6xd, act=act),
        )
        self.stage_layer_3 = nn.Sequential(
            ConvModule(_256xw, _512xw, k=3, s=2, p=1, act=act),
            CSPLayer_2Conv(_512xw, _512xw, add=True, n=_6xd, act=act),
        )
        self.stage_layer_4 = nn.Sequential(
            ConvModule(_512xw, _512xwxr, k=3, s=2, p=1, act=act),
            CSPLayer_2Conv(_512xwxr, _512xwxr, add=True, n=_3xd, act=act),
            SPPF(_512xwxr, act=act),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(_512xwxr, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True

    def forward(self, x):
        p1 = self.stem_layer(x)
        p2 = self.stage_layer_1(p1)
        p3 = self.stage_layer_2(p2)
        p4 = self.stage_layer_3(p3)
        p5 = self.stage_layer_4(p4)
        out = self.gap(p5)
        out = torch.flatten(out, start_dim=1)
        y = self.fc(out)
        return y


def cspdarknet(num_classes=2):
    return Model(num_classes, d=0.33, w=0.25, r=2.0, act="relu6")


if __name__ == "__main__":
    model = cspdarknet(2)
    x = torch.rand(4, 3, 640, 640)
    y = model(x)
    print(y.shape)
