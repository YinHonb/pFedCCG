import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_norm(norm: str, c: int):
    norm = norm.lower()
    if norm == "bn":
        return nn.BatchNorm2d(c)
    if norm == "bn_nostats":
        return nn.BatchNorm2d(c, track_running_stats=False)
    if norm == "gn":
        g = 8
        while g > 1 and (c % g != 0):
            g //= 2
        return nn.GroupNorm(g, c)
    raise ValueError(f"Unknown norm: {norm}")


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm="gn"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.n1 = _make_norm(norm, planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.n2 = _make_norm(norm, planes)

        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                _make_norm(norm, planes),
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.n1(self.conv1(x)), inplace=True)
        out = self.n2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = F.relu(out + identity, inplace=True)
        return out


class ResNetBackbone(nn.Module):
    def __init__(self, layers=(2,2,2,2), width=64, norm="gn"):
        super().__init__()
        self.in_planes = width
        self.conv1 = nn.Conv2d(3, width, 3, stride=1, padding=1, bias=False)
        self.n0 = _make_norm(norm, width)

        self.layer1 = self._make_layer(width,     layers[0], stride=1, norm=norm)
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2, norm=norm)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2, norm=norm)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2, norm=norm)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feat_dim = width * 8

    def _make_layer(self, planes, blocks, stride, norm):
        layers = [BasicBlock(self.in_planes, planes, stride=stride, norm=norm)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes, stride=1, norm=norm))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.n0(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class ResNetCIFAR(nn.Module):
    def __init__(self, n_classes, layers=(2,2,2,2), width=64, norm="gn"):
        super().__init__()
        self.base = ResNetBackbone(layers=layers, width=width, norm=norm)
        self.classifier = nn.Linear(self.base.feat_dim, n_classes)

    def forward(self, x):
        return self.classifier(self.base(x))


def cifar_cnn(n_classes):
    return ResNetCIFAR(n_classes=n_classes, layers=(1,1,1,1), width=64, norm="gn")
