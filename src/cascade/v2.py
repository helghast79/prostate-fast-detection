"""Cascade v2: one preprocessing path, 192×192 for both models.

Coordinates are "edge" pixel coordinates: pixel r spans [r, r+1), so the
full image box is (y0=0, x0=0, h=H, w=W). Every step here has a line-for-line
JavaScript equivalent in VIEWER_INTEGRATION_V2.md.

Pipeline per series:
  1. localizer on every slice: full-image box → 192² → presence, mass, centre
  2. series centre = mass-weighted mean of slice centres
  3. segmenter on every slice: square box of CROP_MM around the series centre
     → 192² → prob → paste back (nearest) into native H×W
  4. keep the largest run of consecutive non-empty slices (slices in IPP order)
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

from .io_nifti import dataset_paths, load_nifti, load_patient_meta

SIZE = 192
N_PIX = SIZE * SIZE
# Percentile indices into the sorted 192² values (1st / 99th percentile).
LO_IDX = round(0.01 * (N_PIX - 1))
HI_IDX = round(0.99 * (N_PIX - 1))
LOCALIZER_THR = 0.5


# --------------------------------------------------------------------------- #
# Preprocessing (mirror exactly in JS)
# --------------------------------------------------------------------------- #


def sample_box(
    image: np.ndarray, y0: float, x0: float, h: float, w: float, *, nearest: bool = False
) -> np.ndarray:
    """Resample box [y0, y0+h) × [x0, x0+w) of a 2D image to SIZE×SIZE.

    Bilinear (or nearest for masks); samples outside the image are 0.
    """
    H, W = image.shape
    img = image.astype(np.float64, copy=False)
    cy = y0 + (np.arange(SIZE) + 0.5) * (h / SIZE)
    cx = x0 + (np.arange(SIZE) + 0.5) * (w / SIZE)
    if nearest:
        ry = np.floor(cy).astype(np.int64)
        rx = np.floor(cx).astype(np.int64)
        vy = (ry >= 0) & (ry < H)
        vx = (rx >= 0) & (rx < W)
        out = np.zeros((SIZE, SIZE), dtype=np.float64)
        out[np.ix_(vy, vx)] = img[np.ix_(ry[vy], rx[vx])]
        return out.astype(np.float32)

    sy = cy - 0.5
    sx = cx - 0.5
    fy = np.floor(sy).astype(np.int64)
    fx = np.floor(sx).astype(np.int64)
    ty = (sy - fy)[:, None]
    tx = (sx - fx)[None, :]

    def take(yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
        vy = (yy >= 0) & (yy < H)
        vx = (xx >= 0) & (xx < W)
        out = np.zeros((SIZE, SIZE), dtype=np.float64)
        out[np.ix_(vy, vx)] = img[np.ix_(yy[vy], xx[vx])]
        return out

    v00 = take(fy, fx)
    v01 = take(fy, fx + 1)
    v10 = take(fy + 1, fx)
    v11 = take(fy + 1, fx + 1)
    out = (
        v00 * (1 - ty) * (1 - tx)
        + v01 * (1 - ty) * tx
        + v10 * ty * (1 - tx)
        + v11 * ty * tx
    )
    return out.astype(np.float32)


def normalize(x: np.ndarray) -> np.ndarray:
    """Per-input percentile normalisation: sorted[LO_IDX], sorted[HI_IDX] → [0,1]."""
    s = np.sort(x.reshape(-1))
    lo = float(s[LO_IDX])
    hi = float(s[HI_IDX])
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def preprocess(image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    return normalize(sample_box(image, *box))


def full_box(H: int, W: int) -> tuple[float, float, float, float]:
    return (0.0, 0.0, float(H), float(W))


def crop_box(
    center_yx: tuple[float, float], spacing_yx: tuple[float, float], crop_mm: float
) -> tuple[float, float, float, float]:
    """Square physical box (crop_mm × crop_mm) centred on an edge-coord centre."""
    h = crop_mm / spacing_yx[0]
    w = crop_mm / spacing_yx[1]
    return (center_yx[0] - h / 2, center_yx[1] - w / 2, h, w)


def paste_back(prob: np.ndarray, box: tuple[float, float, float, float], H: int, W: int,
               thr: float = 0.5) -> np.ndarray:
    """Nearest-neighbour map of a SIZE² probability map back onto native H×W."""
    y0, x0, h, w = box
    iy = np.floor(((np.arange(H) + 0.5) - y0) * SIZE / h).astype(np.int64)
    ix = np.floor(((np.arange(W) + 0.5) - x0) * SIZE / w).astype(np.int64)
    vy = (iy >= 0) & (iy < SIZE)
    vx = (ix >= 0) & (ix < SIZE)
    out = np.zeros((H, W), dtype=np.uint8)
    out[np.ix_(vy, vx)] = (prob[np.ix_(iy[vy], ix[vx])] > thr).astype(np.uint8)
    return out


def keep_largest_slice_run(pred: np.ndarray) -> np.ndarray:
    """Keep the run of consecutive non-empty slices with the most mask pixels."""
    counts = pred.reshape(pred.shape[0], -1).sum(1)
    best, best_mass, z = (0, 0), 0, 0
    while z < len(counts):
        if counts[z] > 0:
            end = z
            while end < len(counts) and counts[end] > 0:
                end += 1
            mass = int(counts[z:end].sum())
            if mass > best_mass:
                best, best_mass = (z, end), mass
            z = end
        else:
            z += 1
    out = np.zeros_like(pred)
    out[best[0] : best[1]] = pred[best[0] : best[1]]
    return out


def series_center(mass: np.ndarray, center_frac: np.ndarray, H: int, W: int
                  ) -> tuple[float, float] | None:
    """Mass-weighted mean of per-slice centres → native edge coords, or None."""
    total = float(mass.sum())
    if total <= 0:
        return None
    cy = float((mass * center_frac[:, 0]).sum() / total) * H
    cx = float((mass * center_frac[:, 1]).sum() / total) * W
    return (cy, cx)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


@dataclass
class Patient:
    pid: str
    image: np.ndarray  # (Z,H,W) raw float32
    mask: np.ndarray  # (Z,H,W) uint8
    spacing_yx: tuple[float, float]
    slice_mm: float

    @property
    def gt_center(self) -> tuple[float, float] | None:
        zz, yy, xx = np.nonzero(self.mask)
        if yy.size == 0:
            return None
        return (float(yy.mean() + 0.5), float(xx.mean() + 0.5))


def load_patient(root: Path, pid: str) -> Patient:
    paths = dataset_paths(root, pid)
    image, affine = load_nifti(paths["image"])
    mask, _ = load_nifti(paths["mask"])
    meta = load_patient_meta(paths["meta"])
    sp = meta["dicom"]["pixel_spacing_yx"]
    return Patient(
        pid=pid,
        image=np.asarray(image, dtype=np.float32),
        mask=(np.asarray(mask) > 0).astype(np.uint8),
        spacing_yx=(float(sp[0]), float(sp[1])),
        slice_mm=float(np.linalg.norm(np.asarray(affine)[:3, 2])),
    )


def localizer_tensors(p: Patient) -> tuple[np.ndarray, np.ndarray]:
    Z, H, W = p.image.shape
    box = full_box(H, W)
    x = np.stack([preprocess(p.image[z], box) for z in range(Z)])
    m = np.stack([sample_box(p.mask[z].astype(np.float32), *box, nearest=True) for z in range(Z)])
    return x[:, None], m[:, None]


def segmenter_tensors(p: Patient, center: tuple[float, float], crop_mm: float
                      ) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    box = crop_box(center, p.spacing_yx, crop_mm)
    Z = p.image.shape[0]
    x = np.stack([preprocess(p.image[z], box) for z in range(Z)])
    m = np.stack([sample_box(p.mask[z].astype(np.float32), *box, nearest=True) for z in range(Z)])
    return x[:, None], m[:, None], box


# --------------------------------------------------------------------------- #
# Model heads (used for Python inference and ONNX export alike)
# --------------------------------------------------------------------------- #


class LocalizerHead(nn.Module):
    """Wrap a 192² mask network → presence, mass, centre (fractions of the box)."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net
        grid = (torch.arange(SIZE, dtype=torch.float32) + 0.5) / SIZE
        self.register_buffer("gy", grid.view(1, 1, SIZE, 1), persistent=False)
        self.register_buffer("gx", grid.view(1, 1, 1, SIZE), persistent=False)

    def forward(self, x: torch.Tensor):
        prob = torch.sigmoid(self.net(x))
        w = torch.relu(prob - LOCALIZER_THR)
        wsum = w.sum(dim=(1, 2, 3))
        cy = (w * self.gy).sum(dim=(1, 2, 3)) / (wsum + 1e-6)
        cx = (w * self.gx).sum(dim=(1, 2, 3)) / (wsum + 1e-6)
        presence = prob.amax(dim=(1, 2, 3))
        mass = wsum / N_PIX
        return presence, mass, torch.stack([cy, cx], dim=1), prob


class SegmenterHead(nn.Module):
    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def augment(x: torch.Tensor, m: torch.Tensor, *, scale: float, shift: float, rot_deg: float
            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Random flip / zoom / shift / rotation / gamma on GPU batches."""
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
    """BCE + Dice over the whole batch (stable with empty slices)."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * target).sum() + 1.0) / (p.sum() + target.sum() + 1.0)
    return bce + dice


def fit(
    net: nn.Module,
    train: tuple[torch.Tensor, torch.Tensor],
    val: tuple[torch.Tensor, torch.Tensor],
    *,
    aug: dict,
    epochs: int,
    patience: int,
    batch_size: int = 32,
    lr: float = 2e-3,
    val_loss_target: float | None = None,
    on_epoch: Callable[[dict], None] | None = None,
    log: Callable[[str], None] = print,
) -> tuple[nn.Module, dict]:
    """Train with early stopping on val loss; returns net loaded with best weights."""
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
            loss = bce_dice(net(xb), mb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
            n += len(idx)
        sched.step()
        vm = evaluate_slices(net, xv, mv, batch_size)
        row = {"epoch": epoch, "train_loss": total / n, "val_loss": vm["loss"],
               "val_dice": vm["dice"], "best": False}
        if vm["loss"] < best["val_loss"] - 1e-4:
            best = {"val_loss": vm["loss"], "val_dice": vm["dice"], "epoch": epoch}
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            stale = 0
            row["best"] = True
        else:
            stale += 1
        row["stale"] = stale
        log(f"epoch {epoch:03d} train={row['train_loss']:.4f} val={vm['loss']:.4f} "
            f"dice={vm['dice']:.3f} stale={stale} t={time.time() - t0:.0f}s")
        if on_epoch:
            on_epoch({**row, "best_epoch": best["epoch"], "best_val_loss": best["val_loss"]})
        if stale >= patience:
            log(f"early stop at epoch {epoch} (best {best['epoch']})")
            break
        if val_loss_target is not None and vm["loss"] <= val_loss_target:
            log(f"val loss target {val_loss_target} reached at epoch {epoch}")
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, best


@torch.no_grad()
def evaluate_slices(net: nn.Module, x: torch.Tensor, m: torch.Tensor, batch_size: int = 64
                    ) -> dict:
    net.eval()
    losses, inter, psum, tsum = [], 0.0, 0.0, 0.0
    for i in range(0, x.shape[0], batch_size):
        logits = net(x[i : i + batch_size])
        tgt = m[i : i + batch_size]
        losses.append(float(bce_dice(logits, tgt)) * len(tgt))
        pred = (logits > 0).float()
        inter += float((pred * tgt).sum())
        psum += float(pred.sum())
        tsum += float(tgt.sum())
    return {
        "loss": sum(losses) / x.shape[0],
        "dice": (2 * inter + 1e-6) / (psum + tsum + 1e-6),
    }


# --------------------------------------------------------------------------- #
# Inference (reference implementation of the browser pipeline)
# --------------------------------------------------------------------------- #


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


@torch.no_grad()
def run_cascade(
    loc_head: LocalizerHead,
    seg_head: SegmenterHead,
    p: Patient,
    crop_mm: float,
    device: torch.device,
    center_override: tuple[float, float] | None = None,
) -> dict:
    Z, H, W = p.image.shape
    loc = run_localizer(loc_head, p, device)
    center = center_override or series_center(loc["mass"], loc["center_frac"], H, W)
    if center is None:
        return {"center": None, "pred": np.zeros_like(p.mask), "loc": loc, "box": None}
    x, _, box = segmenter_tensors(p, center, crop_mm)
    seg_head.eval()
    prob = seg_head(torch.from_numpy(x).to(device)).cpu().numpy()[:, 0]
    pred = keep_largest_slice_run(np.stack([paste_back(prob[z], box, H, W) for z in range(Z)]))
    return {"center": center, "pred": pred, "loc": loc, "box": box}


def dice3d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    denom = a.sum() + b.sum()
    return 1.0 if denom == 0 else float(2 * (a & b).sum() / denom)


def center_error_mm(c: tuple[float, float] | None, gt: tuple[float, float] | None,
                    spacing_yx: tuple[float, float]) -> float | None:
    if c is None or gt is None:
        return None
    return float(math.hypot((c[0] - gt[0]) * spacing_yx[0], (c[1] - gt[1]) * spacing_yx[1]))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
