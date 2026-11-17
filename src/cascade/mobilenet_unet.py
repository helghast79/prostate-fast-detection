"""MobileNetV3-Small encoder + lightweight U-Net decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MobileNetV3UNet(nn.Module):
    """U-Net-style decoder on MobileNetV3-Small features."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        weights = (
            MobileNet_V3_Small_Weights.DEFAULT
            if pretrained and in_channels == 3
            else None
        )
        backbone = mobilenet_v3_small(weights=weights)
        feats = backbone.features

        if in_channels != 3:
            old = feats[0][0]
            new = nn.Conv2d(
                in_channels,
                old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=False,
            )
            with torch.no_grad():
                if old.weight.shape[1] == 3 and in_channels == 1:
                    new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
            feats[0][0] = new

        # Tap points chosen for ~1/2, 1/4, 1/8, 1/16 resolutions.
        self.stem = nn.Sequential(*feats[0:2])
        self.block2 = nn.Sequential(*feats[2:4])
        self.block3 = nn.Sequential(*feats[4:9])
        self.block4 = nn.Sequential(*feats[9:])

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 160, 160)
            s1 = self.stem(dummy)
            s2 = self.block2(s1)
            s3 = self.block3(s2)
            s4 = self.block4(s3)
            c1, c2, c3, c4 = s1.shape[1], s2.shape[1], s3.shape[1], s4.shape[1]

        self.up3 = nn.ConvTranspose2d(c4, 128, 2, stride=2)
        self.dec3 = ConvBNAct(128 + c3, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBNAct(64 + c2, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBNAct(32 + c1, 32)
        self.up0 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.head = nn.Conv2d(16, out_channels, 1)

    def _match(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.stem(x)
        s2 = self.block2(s1)
        s3 = self.block3(s2)
        s4 = self.block4(s3)

        d3 = self._match(self.up3(s4), s3)
        d3 = self.dec3(torch.cat([d3, s3], dim=1))
        d2 = self._match(self.up2(d3), s2)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))
        d1 = self._match(self.up1(d2), s1)
        d1 = self.dec1(torch.cat([d1, s1], dim=1))
        d0 = self._match(self.up0(d1), x)
        return self.head(d0)


def build_segmenter(name: str, in_channels: int = 1, pretrained: bool = False) -> nn.Module:
    key = name.lower().replace("-", "_")
    if key in {"enet", "enet_binary"}:
        from .enet import ENetBinary

        return ENetBinary(in_channels=in_channels, out_channels=1)
    if key in {"mobilenet_unet", "mobilenetv3_unet", "mobilenet"}:
        return MobileNetV3UNet(
            in_channels=in_channels, out_channels=1, pretrained=pretrained
        )
    raise ValueError(f"Unknown segmenter {name!r}; expected enet|mobilenet_unet")
