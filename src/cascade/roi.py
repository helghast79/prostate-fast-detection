"""Deterministic ROI extraction around a predicted / GT prostate centre."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .decisions import CASCADE_DECISIONS


@dataclass(frozen=True)
class RoiCrop:
    """Square crop in full-slice pixel coordinates (inclusive-exclusive)."""

    y0: int
    x0: int
    y1: int
    x1: int
    center_y: float
    center_x: float
    size: int
    pad_top: int
    pad_left: int
    pad_bottom: int
    pad_right: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_square_roi(
    center_yx: tuple[float, float],
    rows: int,
    cols: int,
    size: int | None = None,
    margin_frac: float | None = None,
    gland_extent_px: float | None = None,
) -> RoiCrop:
    """
    Build a square ROI centred on ``center_yx`` (row, col).

    If ``gland_extent_px`` is given (e.g. max bbox side), the crop is at least
    ``extent * (1 + 2*margin_frac)``, then clamped to ``size`` when size is set
    as a fixed model input (resize happens after extract).
    """
    size = CASCADE_DECISIONS.roi_crop_size if size is None else int(size)
    margin_frac = (
        CASCADE_DECISIONS.roi_margin_frac if margin_frac is None else float(margin_frac)
    )
    cy, cx = float(center_yx[0]), float(center_yx[1])

    if gland_extent_px is not None and gland_extent_px > 0:
        needed = int(np.ceil(gland_extent_px * (1.0 + 2.0 * margin_frac)))
        crop = max(size, needed)
    else:
        crop = size

    # Prefer odd/even consistency: keep requested size for model training.
    crop = int(size)

    half = crop / 2.0
    y0_f, x0_f = cy - half, cx - half
    y0 = int(np.floor(y0_f))
    x0 = int(np.floor(x0_f))
    y1 = y0 + crop
    x1 = x0 + crop

    pad_top = max(0, -y0)
    pad_left = max(0, -x0)
    pad_bottom = max(0, y1 - rows)
    pad_right = max(0, x1 - cols)

    y0_c = max(0, y0)
    x0_c = max(0, x0)
    y1_c = min(rows, y1)
    x1_c = min(cols, x1)

    return RoiCrop(
        y0=y0_c,
        x0=x0_c,
        y1=y1_c,
        x1=x1_c,
        center_y=cy,
        center_x=cx,
        size=crop,
        pad_top=pad_top,
        pad_left=pad_left,
        pad_bottom=pad_bottom,
        pad_right=pad_right,
    )


def extract_roi(
    image: np.ndarray,
    crop: RoiCrop,
    *,
    fill: float = 0.0,
) -> np.ndarray:
    """Extract a ``crop.size × crop.size`` patch with zero/constant padding."""
    if image.ndim == 2:
        out = np.full((crop.size, crop.size), fill, dtype=np.float32)
        patch = image[crop.y0 : crop.y1, crop.x0 : crop.x1]
        out[
            crop.pad_top : crop.pad_top + patch.shape[0],
            crop.pad_left : crop.pad_left + patch.shape[1],
        ] = patch.astype(np.float32)
        return out
    if image.ndim == 3:
        # (C, H, W) or (Z, H, W) — treat last two as spatial
        spatial = extract_roi(image.reshape(-1, image.shape[-2], image.shape[-1])[0], crop, fill=fill)
        channels = []
        flat = image.reshape(-1, image.shape[-2], image.shape[-1])
        for ch in flat:
            channels.append(extract_roi(ch, crop, fill=fill))
        return np.stack(channels, axis=0).reshape(image.shape[:-2] + (crop.size, crop.size))
    raise ValueError(f"Unsupported image ndim={image.ndim}")


def extract_roi_mask(mask: np.ndarray, crop: RoiCrop) -> np.ndarray:
    return extract_roi(mask.astype(np.float32), crop, fill=0.0).astype(np.uint8)


def jitter_center(
    center_yx: tuple[float, float],
    crop_size: int,
    jitter_frac: float | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Displace a GT centre so the segmenter sees imperfect localization."""
    jitter_frac = (
        CASCADE_DECISIONS.roi_center_jitter_frac
        if jitter_frac is None
        else float(jitter_frac)
    )
    rng = rng or np.random.default_rng()
    amp = crop_size * jitter_frac
    dy = float(rng.uniform(-amp, amp))
    dx = float(rng.uniform(-amp, amp))
    return center_yx[0] + dy, center_yx[1] + dx


def paste_roi_into_full(
    full_shape_yx: tuple[int, int],
    crop: RoiCrop,
    roi_mask: np.ndarray,
) -> np.ndarray:
    """Map a ROI mask back into full-slice coordinates (Stage 5 helper)."""
    out = np.zeros(full_shape_yx, dtype=np.uint8)
    rh, rw = roi_mask.shape[-2:]
    y0 = crop.y0 - crop.pad_top
    x0 = crop.x0 - crop.pad_left
    # roi coords that land inside the original image
    src_y0 = crop.pad_top
    src_x0 = crop.pad_left
    src_y1 = src_y0 + (crop.y1 - crop.y0)
    src_x1 = src_x0 + (crop.x1 - crop.x0)
    patch = roi_mask[src_y0:src_y1, src_x0:src_x1]
    out[crop.y0 : crop.y1, crop.x0 : crop.x1] = (patch > 0).astype(np.uint8)
    return out
