"""Shallow CNN localizer: prostate presence + centre regression."""

from __future__ import annotations

import torch
import torch.nn as nn


class ShallowProstateLocalizer(nn.Module):
    """
    Tiny 2D CNN for 128×128 axial slices.

    Outputs:
      - presence logit (BCEWithLogits)
      - centre (y, x) in [0, 1] via sigmoid
    """

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16
            nn.Conv2d(64, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.presence = nn.Linear(96, 1)
        self.center = nn.Linear(96, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x).flatten(1)
        presence = self.presence(feat).squeeze(1)
        center = torch.sigmoid(self.center(feat))
        return presence, center


def localizer_loss(
    presence_logits: torch.Tensor,
    center_pred: torch.Tensor,
    presence_tgt: torch.Tensor,
    center_tgt: torch.Tensor,
    *,
    center_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    bce = nn.functional.binary_cross_entropy_with_logits(
        presence_logits, presence_tgt.float()
    )
    # Only supervise centre on positive slices.
    pos = presence_tgt > 0.5
    if pos.any():
        center_l1 = nn.functional.smooth_l1_loss(center_pred[pos], center_tgt[pos])
    else:
        center_l1 = presence_logits.new_zeros(())
    loss = bce + center_weight * center_l1
    return loss, {
        "bce": float(bce.detach().item()),
        "center_l1": float(center_l1.detach().item()),
        "loss": float(loss.detach().item()),
    }
