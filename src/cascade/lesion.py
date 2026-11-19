"""Lesion model inputs: T2 + ADC + DWI sampled on the same 192² grid.

The grid is the cascade v2 segmenter box (CROP_MM around the prostate series
centre) on each T2 slice. ADC and DWI are sampled at the *patient-space*
positions of those grid points (bilinear in-plane, linear between slices),
so no prior registration/resampling of the volumes is needed. Each channel is
then normalised on its own with the same p1/p99 rule as the prostate models.
Mirrored line-for-line in frontend/prostate_cascade_v2.js.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from .v2 import SIZE, normalize, sample_box


@dataclass
class Series:
    """IPP-sorted slices of one series (all with the same IOP / spacing / size)."""

    pixels: np.ndarray  # (Z,H,W) float32, rescale applied
    ipp: np.ndarray  # (Z,3)
    iop: np.ndarray  # (6,)
    spacing: np.ndarray  # (2,) [row spacing, column spacing]

    @property
    def row_dir(self) -> np.ndarray:
        return self.iop[:3]

    @property
    def col_dir(self) -> np.ndarray:
        return self.iop[3:]

    @property
    def normal(self) -> np.ndarray:
        n = np.cross(self.row_dir, self.col_dir)
        return n / np.linalg.norm(n)

    def to_npz(self, prefix: str) -> dict:
        return {f"{prefix}_pixels": self.pixels, f"{prefix}_ipp": self.ipp,
                f"{prefix}_iop": self.iop, f"{prefix}_spacing": self.spacing}

    @classmethod
    def from_npz(cls, d, prefix: str) -> "Series":
        return cls(d[f"{prefix}_pixels"].astype(np.float32), d[f"{prefix}_ipp"].astype(np.float64),
                   d[f"{prefix}_iop"].astype(np.float64), d[f"{prefix}_spacing"].astype(np.float64))


def grid_points(t2: Series, z: int, box: tuple[float, float, float, float]) -> np.ndarray:
    """Patient coordinates (SIZE,SIZE,3) of the box grid centres on T2 slice z."""
    y0, x0, h, w = box
    r = y0 + (np.arange(SIZE) + 0.5) * (h / SIZE) - 0.5
    c = x0 + (np.arange(SIZE) + 0.5) * (w / SIZE) - 0.5
    return (
        t2.ipp[z][None, None, :]
        + c[None, :, None] * t2.spacing[1] * t2.row_dir[None, None, :]
        + r[:, None, None] * t2.spacing[0] * t2.col_dir[None, None, :]
    )


def sample_series(s: Series, pts: np.ndarray) -> np.ndarray:
    """Sample a series at patient points; 0 outside (half a slice gap beyond the ends)."""
    Z, H, W = s.pixels.shape
    pos = s.ipp @ s.normal
    wpos = pts @ s.normal
    if Z == 1:
        k = np.zeros(wpos.shape, dtype=np.int64)
        t = np.zeros(wpos.shape)
        valid = np.abs(wpos - pos[0]) <= 2.0
        k1 = k
    else:
        k = np.clip(np.searchsorted(pos, wpos, side="right") - 1, 0, Z - 2)
        t = (wpos - pos[k]) / (pos[k + 1] - pos[k])
        lo_ok = wpos >= pos[0] - 0.5 * (pos[1] - pos[0])
        hi_ok = wpos <= pos[-1] + 0.5 * (pos[-1] - pos[-2])
        valid = lo_ok & hi_ok
        t = np.clip(t, 0.0, 1.0)
        k1 = k + 1

    def bilinear(kk: np.ndarray) -> np.ndarray:
        d = pts - s.ipp[kk]
        col = (d * s.row_dir).sum(-1) / s.spacing[1]
        row = (d * s.col_dir).sum(-1) / s.spacing[0]
        fr = np.floor(row).astype(np.int64)
        fc = np.floor(col).astype(np.int64)
        tr = row - fr
        tc = col - fc

        def at(rr, cc):
            ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            out = np.zeros(rr.shape, dtype=np.float64)
            out[ok] = s.pixels[kk[ok], rr[ok], cc[ok]]
            return out

        return (at(fr, fc) * (1 - tr) * (1 - tc) + at(fr, fc + 1) * (1 - tr) * tc
                + at(fr + 1, fc) * tr * (1 - tc) + at(fr + 1, fc + 1) * tr * tc)

    v = bilinear(k) * (1 - t) + bilinear(k1) * t
    return np.where(valid, v, 0.0).astype(np.float32)


def lesion_input(t2: Series, adc: Series, dwi: Series, z: int,
                 box: tuple[float, float, float, float]) -> np.ndarray:
    pts = grid_points(t2, z, box)
    return np.stack([
        normalize(sample_box(t2.pixels[z], *box)),
        normalize(sample_series(adc, pts)),
        normalize(sample_series(dwi, pts)),
    ])


@dataclass
class LesionCase:
    pid: str
    t2: Series
    adc: Series
    dwi: Series
    mask: np.ndarray  # (Z,H,W) lesion mask on T2 grid
    meta: dict
    prostate: np.ndarray | None = None  # (Z,H,W) binary gland mask on T2 grid, optional


def load_case(root: Path, pid: str) -> LesionCase:
    import json

    d = np.load(root / "cases" / f"{pid}.npz")
    meta = json.loads((root / "cases" / f"{pid}.json").read_text())
    prostate = d["prostate"].astype(np.uint8) if "prostate" in d.files else None
    return LesionCase(
        pid,
        Series.from_npz(d, "t2"),
        Series.from_npz(d, "adc"),
        Series.from_npz(d, "dwi"),
        d["mask"].astype(np.uint8),
        meta,
        prostate,
    )


def prostate_center_yx(prostate: np.ndarray) -> tuple[float, float] | None:
    """Mass centroid of a binary prostate volume in native edge coords."""
    zz, yy, xx = np.nonzero(prostate)
    if yy.size == 0:
        return None
    return (float(yy.mean() + 0.5), float(xx.mean() + 0.5))


def restrict_to_prostate(lesion: np.ndarray, prostate: np.ndarray, radius_px: int = 3) -> np.ndarray:
    """Keep lesion pixels within a (2r+1)² square of the prostate mask on the same slice."""
    size = (1, 2 * radius_px + 1, 2 * radius_px + 1)
    return (lesion.astype(bool) & (ndimage.maximum_filter(prostate.astype(np.uint8), size=size) > 0)).astype(np.uint8)


def lesion_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """3D Dice + lesion-level detection (connected components, any overlap = hit)."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    denom = pred.sum() + gt.sum()
    gl, ng = ndimage.label(gt)
    pl, npred = ndimage.label(pred)
    hits = sum(1 for i in range(1, ng + 1) if pred[gl == i].any())
    fps = sum(1 for i in range(1, npred + 1) if not gt[pl == i].any())
    return {
        "dice3d": 1.0 if denom == 0 else float(2 * (pred & gt).sum() / denom),
        "gt_lesions": ng,
        "detected": hits,
        "pred_components": npred,
        "false_positive_components": fps,
    }
