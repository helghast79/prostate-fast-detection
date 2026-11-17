"""Segmentation losses shared by ENet / MobileNet-UNet."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    ``logits``: (B,1,H,W) or (B,H,W); ``targets`` binary float/bool same spatial.
    """
    if logits.ndim == 3:
        logits = logits.unsqueeze(1)
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = 1.0 - (2.0 * inter + eps) / (denom + eps)
    return bce_weight * bce + dice_weight * dice.mean()


@torch.no_grad()
def binary_dice(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    if logits.ndim == 3:
        logits = logits.unsqueeze(1)
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    preds = (torch.sigmoid(logits) > 0.5).float()
    targets = targets.float()
    inter = (preds * targets).sum()
    denom = preds.sum() + targets.sum()
    return float(((2.0 * inter + eps) / (denom + eps)).item())
