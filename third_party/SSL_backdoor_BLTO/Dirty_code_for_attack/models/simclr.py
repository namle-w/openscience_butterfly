import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.downsample(identity)
        out = self.relu(out)
        return out


class ResNetCIFAR(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 10)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def resnet18_cifar():
    return ResNetCIFAR(BasicBlock, [2, 2, 2, 2])


class ResNetSimCLRBackbone(nn.Module):
    def __init__(self, base_model):
        super().__init__()

        if not hasattr(base_model, "fc"):
            raise ValueError("Backbone has no fc.in_features")

        self.output_dim = base_model.fc.in_features

        # If the incoming backbone already exposes a feature sequential `f`,
        # reuse it directly. Otherwise build the expected CIFAR SimCLR layout.
        if hasattr(base_model, "f") and isinstance(base_model.f, nn.Sequential):
            self.f = base_model.f
        else:
            relu = base_model.relu if hasattr(base_model, "relu") else nn.ReLU(inplace=True)
            self.f = nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                relu,
                base_model.layer1,
                base_model.layer2,
                base_model.layer3,
                base_model.layer4,
            )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.f(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        return x


class SimCLR(nn.Module):
    def __init__(self, backbone_name="resnet18", projector_hidden=2048, projector_out=128):
        super().__init__()

        # Accept either:
        # 1) a backbone name string, e.g. "resnet18"
        # 2) an already-built backbone module, e.g. get_backbone(...)
        if isinstance(backbone_name, str):
            if backbone_name == "resnet18":
                base_model = resnet18_cifar()
            else:
                raise ValueError(f"Unsupported backbone: {backbone_name}")
        elif isinstance(backbone_name, nn.Module):
            base_model = backbone_name
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.backbone = ResNetSimCLRBackbone(base_model)

        self.projector = nn.Sequential(
            nn.Linear(self.backbone.output_dim, projector_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(projector_hidden, projector_out),
        )

    def forward(self, x):
        feat = self.backbone(x)
        proj = self.projector(feat)
        return feat, proj

    def load_state_dict(self, state_dict, strict=True):
        from collections import OrderedDict

        canonical_sd = OrderedDict()

        for k, v in state_dict.items():
            nk = k

            # legacy alias: f.f.xxx -> backbone.f.xxx
            if nk.startswith("f.f."):
                nk = "backbone" + nk[1:]

            if nk in canonical_sd:
                if canonical_sd[nk].shape != v.shape:
                    raise RuntimeError(
                        f"Duplicate checkpoint key mapped to {nk} with inconsistent shapes: "
                        f"{canonical_sd[nk].shape} vs {v.shape}"
                    )
                continue

            canonical_sd[nk] = v

        return super().load_state_dict(canonical_sd, strict=strict)