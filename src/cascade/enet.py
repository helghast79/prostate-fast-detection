"""Lightweight ENet-style binary segmenter for ROI crops."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InitialBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 16) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch - in_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(torch.cat([self.conv(x), self.pool(x)], dim=1)))


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        downsample: bool = False,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        stride = 2 if downsample else 1
        internal = out_ch // 4
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_ch, internal, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(internal)
        self.conv2 = nn.Conv2d(
            internal,
            internal,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(internal)
        self.conv3 = nn.Conv2d(internal, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU(out_ch)
        self.drop = nn.Dropout2d(dropout)
        self.proj = None
        if downsample or in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.drop(self.bn3(self.conv3(out)))
        residual = x if self.proj is None else self.proj(x)
        return self.act(out + residual)


class UpsampleBottleneck(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1) -> None:
        super().__init__()
        internal = out_ch // 4 if out_ch >= 4 else out_ch
        self.conv1 = nn.Conv2d(in_ch, internal, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(internal)
        self.deconv = nn.ConvTranspose2d(internal, internal, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(internal)
        self.conv3 = nn.Conv2d(internal, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU(out_ch)
        self.drop = nn.Dropout2d(dropout)
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.deconv(out)), inplace=True)
        out = self.drop(self.bn3(self.conv3(out)))
        residual = F.interpolate(self.proj(x), size=out.shape[-2:], mode="nearest")
        return self.act(out + residual)


class ENetBinary(nn.Module):
    """Compact ENet for binary prostate ROI segmentation."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1) -> None:
        super().__init__()
        self.initial = InitialBlock(in_channels, 16)
        self.down1 = nn.Sequential(
            Bottleneck(16, 64, downsample=True),
            Bottleneck(64, 64),
            Bottleneck(64, 64),
        )
        self.down2 = nn.Sequential(
            Bottleneck(64, 128, downsample=True),
            Bottleneck(128, 128),
            Bottleneck(128, 128, dilation=2),
            Bottleneck(128, 128),
            Bottleneck(128, 128, dilation=4),
        )
        self.up1 = UpsampleBottleneck(128, 64)
        self.up2 = UpsampleBottleneck(64, 16)
        self.head = nn.ConvTranspose2d(16, out_channels, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.up1(x)
        x = self.up2(x)
        return self.head(x)
