"""Derive localization / ROI supervision from 3D binary masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SliceLabel:
    z: int
    present: bool
    center_yx: tuple[float, float] | None  # pixel coords (row, col) in full slice
    bbox_yxhw: tuple[float, float, float, float] | None  # y, x, h, w
    area: int


def mask_slice_label(mask2d: np.ndarray, z: int) -> SliceLabel:
    m = np.asarray(mask2d).astype(bool)
    area = int(m.sum())
    if area == 0:
        return SliceLabel(z=z, present=False, center_yx=None, bbox_yxhw=None, area=0)
    ys, xs = np.where(m)
    cy = float(ys.mean())
    cx = float(xs.mean())
    y0, y1 = float(ys.min()), float(ys.max())
    x0, x1 = float(xs.min()), float(xs.max())
    return SliceLabel(
        z=z,
        present=True,
        center_yx=(cy, cx),
        bbox_yxhw=(y0, x0, y1 - y0 + 1.0, x1 - x0 + 1.0),
        area=area,
    )


def volume_slice_labels(mask_zyx: np.ndarray) -> list[SliceLabel]:
    mask_zyx = np.asarray(mask_zyx)
    return [mask_slice_label(mask_zyx[z], z) for z in range(mask_zyx.shape[0])]


def normalize_center(
    center_yx: tuple[float, float],
    rows: int,
    cols: int,
) -> tuple[float, float]:
    """Map pixel center to [0, 1] relative coordinates (y, x)."""
    y, x = center_yx
    return y / max(rows - 1, 1), x / max(cols - 1, 1)


def denormalize_center(
    center_yx_norm: tuple[float, float],
    rows: int,
    cols: int,
) -> tuple[float, float]:
    y, x = center_yx_norm
    return y * max(rows - 1, 1), x * max(cols - 1, 1)
