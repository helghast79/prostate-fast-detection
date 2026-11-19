"""Cascade v3: localizer-v3 + prostate_detection-v3 (gland + PZ).

Same 192² preprocessing as v2 (full-slice localizer → CROP_MM segmenter), but
the detection head predicts 3 classes: background / gland_non_pz / PZ.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cascade.v2 import (
    LOCALIZER_THR,
    N_PIX,
    SIZE,
    LocalizerHead,
    center_error_mm,
    crop_box,
    full_box,
    keep_largest_slice_run,
    normalize,
    paste_back,
    preprocess,
    sample_box,
    series_center,
    write_json,
)
from src.model import ce_multiclass_dice_loss, multiclass_metrics
from src.zone_labels import CLASS_GLAND, CLASS_NAMES, CLASS_PZ

NUM_CLASSES = 3
CROP_MM_DEFAULT = 110.0


@dataclass
class Patient:
    pid: str
    image: np.ndarray  # (Z,H,W) float32
    mask: np.ndarray  # (Z,H,W) uint8 class index 0/1/2
    spacing_yx: tuple[float, float]
    slice_mm: float

    @property
    def binary_mask(self) -> np.ndarray:
        return (self.mask > 0).astype(np.uint8)

    @property
    def gt_center(self) -> tuple[float, float] | None:
        zz, yy, xx = np.nonzero(self.mask)
        if yy.size == 0:
            return None
        return (float(yy.mean() + 0.5), float(xx.mean() + 0.5))


def load_patient(root: Path, pid: str) -> Patient:
    npz = np.load(root / "cases" / f"{pid}.npz")
    meta = json.loads((root / "cases" / f"{pid}.json").read_text())
    sp = meta["spacing_yx_mm"]
    return Patient(
        pid=pid,
        image=np.asarray(npz["image"], dtype=np.float32),
        mask=np.asarray(npz["mask"], dtype=np.uint8),
        spacing_yx=(float(sp[0]), float(sp[1])),
        slice_mm=float(meta["slice_mm"]),
    )


def localizer_tensors(p: Patient) -> tuple[np.ndarray, np.ndarray]:
    Z, H, W = p.image.shape
    box = full_box(H, W)
    bin_m = p.binary_mask
    x = np.stack([preprocess(p.image[z], box) for z in range(Z)])
    m = np.stack([sample_box(bin_m[z].astype(np.float32), *box, nearest=True) for z in range(Z)])
    return x[:, None], m[:, None]


def detection_tensors(
    p: Patient, center: tuple[float, float], crop_mm: float
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Images (Z,1,S,S) and class masks (Z,1,S,S) with values 0/1/2."""
    box = crop_box(center, p.spacing_yx, crop_mm)
    Z = p.image.shape[0]
    x = np.stack([preprocess(p.image[z], box) for z in range(Z)])
    m = np.stack([sample_box(p.mask[z].astype(np.float32), *box, nearest=True) for z in range(Z)])
    return x[:, None], m[:, None], box


class DetectionHead(nn.Module):
    """Softmax multi-class head for gland / PZ."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=1)


def augment(
    x: torch.Tensor, m: torch.Tensor, *, scale: float, shift: float, rot_deg: float
) -> tuple[torch.Tensor, torch.Tensor]:
    b = x.shape[0]
    dev = x.device
    s = 1.0 + (torch.rand(b, device=dev) * 2 - 1) * scale
    a = (torch.rand(b, device=dev) * 2 - 1) * math.radians(rot_deg)
    tx = (torch.rand(b, device=dev) * 2 - 1) * shift * 2
    ty = (torch.rand(b, device=dev) * 2 - 1) * shift * 2
    flip = torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)
    cos, sin = torch.cos(a) / s, torch.sin(a) / s
    theta = torch.stack(
        [torch.stack([cos * flip, -sin, tx], 1), torch.stack([sin * flip, cos, ty], 1)], 1
    )
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    m = F.grid_sample(m, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    gamma = torch.exp((torch.rand(b, 1, 1, 1, device=dev) * 2 - 1) * math.log(1.5))
    x = x.clamp(0, 1) ** gamma
    x = (x + 0.02 * torch.randn_like(x)).clamp(0, 1)
    return x, m


def bce_dice(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * target).sum() + 1.0) / (p.sum() + target.sum() + 1.0)
    return bce + dice


def fit_binary(
    net: nn.Module,
    train: tuple[torch.Tensor, torch.Tensor],
    val: tuple[torch.Tensor, torch.Tensor],
    *,
    aug: dict,
    epochs: int,
    patience: int,
    batch_size: int = 32,
    lr: float = 2e-3,
    on_epoch: Callable[[dict], None] | None = None,
    log: Callable[[str], None] = print,
) -> tuple[nn.Module, dict]:
    from src.cascade.v2 import evaluate_slices, fit

    return fit(
        net,
        train,
        val,
        aug=aug,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        lr=lr,
        on_epoch=on_epoch,
        log=log,
    )


def fit_multiclass(
    net: nn.Module,
    train: tuple[torch.Tensor, torch.Tensor],
    val: tuple[torch.Tensor, torch.Tensor],
    *,
    aug: dict,
    epochs: int,
    patience: int,
    batch_size: int = 24,
    lr: float = 2e-3,
    on_epoch: Callable[[dict], None] | None = None,
    log: Callable[[str], None] = print,
) -> tuple[nn.Module, dict]:
    dev = next(net.parameters()).device
    xt, mt = train
    xv, mv = val
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = {"val_loss": float("inf"), "epoch": 0}
    best_state = None
    stale = 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(xt.shape[0], device=dev)
        total, n = 0.0, 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i : i + batch_size]
            xb, mb = augment(xt[idx], mt[idx], **aug)
            tgt = mb[:, 0].round().clamp(0, NUM_CLASSES - 1).long()
            loss = ce_multiclass_dice_loss(net(xb), tgt, num_classes=NUM_CLASSES)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
            n += len(idx)
        sched.step()
        vm = evaluate_multiclass(net, xv, mv, batch_size)
        row = {
            "epoch": epoch,
            "train_loss": total / n,
            "val_loss": vm["loss"],
            "val_dice": vm["dice"],
            "val_dice_gland": vm["dice_c1"],
            "val_dice_pz": vm["dice_c2"],
            "best": False,
        }
        if vm["loss"] < best["val_loss"] - 1e-4:
            best = {
                "val_loss": vm["loss"],
                "val_dice": vm["dice"],
                "val_dice_gland": vm["dice_c1"],
                "val_dice_pz": vm["dice_c2"],
                "epoch": epoch,
            }
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            stale = 0
            row["best"] = True
        else:
            stale += 1
        row["stale"] = stale
        log(
            f"epoch {epoch:03d} train={row['train_loss']:.4f} val={vm['loss']:.4f} "
            f"dice={vm['dice']:.3f} g={vm['dice_c1']:.3f} pz={vm['dice_c2']:.3f} "
            f"stale={stale} t={time.time() - t0:.0f}s"
        )
        if on_epoch:
            on_epoch({**row, "best_epoch": best["epoch"], "best_val_loss": best["val_loss"]})
        if stale >= patience:
            log(f"early stop at epoch {epoch} (best {best['epoch']})")
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, best


@torch.no_grad()
def evaluate_multiclass(
    net: nn.Module, x: torch.Tensor, m: torch.Tensor, batch_size: int = 64
) -> dict:
    net.eval()
    losses = []
    metrics_acc = {"dice": 0.0, "dice_c1": 0.0, "dice_c2": 0.0}
    n = 0
    for i in range(0, x.shape[0], batch_size):
        logits = net(x[i : i + batch_size])
        tgt = m[i : i + batch_size, 0].round().clamp(0, NUM_CLASSES - 1).long()
        loss = ce_multiclass_dice_loss(logits, tgt, num_classes=NUM_CLASSES)
        met = multiclass_metrics(logits, tgt, num_classes=NUM_CLASSES)
        bs = int(tgt.shape[0])
        losses.append(float(loss) * bs)
        for k in metrics_acc:
            metrics_acc[k] += float(met.get(k, 0.0)) * bs
        n += bs
    return {"loss": sum(losses) / max(n, 1), **{k: v / max(n, 1) for k, v in metrics_acc.items()}}


@torch.no_grad()
def run_localizer(head: LocalizerHead, p: Patient, device: torch.device) -> dict:
    x, _ = localizer_tensors(p)
    head.eval()
    presence, mass, center, _ = head(torch.from_numpy(x).to(device))
    return {
        "presence": presence.cpu().numpy(),
        "mass": mass.cpu().numpy(),
        "center_frac": center.cpu().numpy(),
    }


def paste_class_back(
    classes: np.ndarray, box: tuple[float, float, float, float], H: int, W: int
) -> np.ndarray:
    """Nearest map of SIZE² class indices into native HxW."""
    y0, x0, h, w = box
    out = np.zeros((H, W), dtype=np.uint8)
    iy = np.floor(((np.arange(H) + 0.5) - y0) * SIZE / h).astype(np.int64)
    ix = np.floor(((np.arange(W) + 0.5) - x0) * SIZE / w).astype(np.int64)
    vy = (iy >= 0) & (iy < SIZE)
    vx = (ix >= 0) & (ix < SIZE)
    yy = np.where(vy)[0]
    xx = np.where(vx)[0]
    if yy.size and xx.size:
        out[np.ix_(yy, xx)] = classes[np.ix_(iy[yy], ix[xx])]
    return out


@torch.no_grad()
def run_cascade(
    loc_head: LocalizerHead,
    det_head: DetectionHead,
    p: Patient,
    crop_mm: float,
    device: torch.device,
    center_override: tuple[float, float] | None = None,
) -> dict:
    Z, H, W = p.image.shape
    loc = run_localizer(loc_head, p, device)
    center = center_override or series_center(loc["mass"], loc["center_frac"], H, W)
    if center is None:
        return {
            "center": None,
            "pred": np.zeros_like(p.mask),
            "pred_binary": np.zeros_like(p.binary_mask),
            "loc": loc,
            "box": None,
        }
    x, _, box = detection_tensors(p, center, crop_mm)
    det_head.eval()
    prob = det_head(torch.from_numpy(x).to(device)).cpu().numpy()  # (Z,C,S,S)
    classes = prob.argmax(axis=1).astype(np.uint8)
    pred = np.stack([paste_class_back(classes[z], box, H, W) for z in range(Z)])
    binary = keep_largest_slice_run((pred > 0).astype(np.uint8))
    pred = pred * binary  # drop components outside largest run
    return {"center": center, "pred": pred, "pred_binary": binary, "loc": loc, "box": box}


def dice3d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    denom = a.sum() + b.sum()
    return 1.0 if denom == 0 else float(2 * (a & b).sum() / denom)


def zone_dice3d(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    return {
        "dice_gland": dice3d(pred == CLASS_GLAND, gt == CLASS_GLAND),
        "dice_pz": dice3d(pred == CLASS_PZ, gt == CLASS_PZ),
        "dice_prostate": dice3d(pred > 0, gt > 0),
    }


__all__ = [
    "CLASS_NAMES",
    "CLASS_GLAND",
    "CLASS_PZ",
    "CROP_MM_DEFAULT",
    "DetectionHead",
    "LocalizerHead",
    "NUM_CLASSES",
    "Patient",
    "SIZE",
    "center_error_mm",
    "detection_tensors",
    "dice3d",
    "fit_binary",
    "fit_multiclass",
    "load_patient",
    "localizer_tensors",
    "run_cascade",
    "run_localizer",
    "series_center",
    "write_json",
    "zone_dice3d",
]
