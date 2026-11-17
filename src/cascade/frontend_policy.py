"""Robust series-level policy for browser cascade inference.

Per-slice ``presence >= 0.5`` gating is brittle in the viewer (domain shift,
apex/base under-confidence). Prefer:

1. score every slice
2. lock onto the peak-presence slice
3. grow a contiguous slab around it
4. run the segmenter on that slab (with centre fallback/smoothing)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SliceScore:
    z: int
    presence: float
    center_yx: tuple[float, float]  # native pixels (row, col)


@dataclass(frozen=True)
class SeriesPlan:
    peak_z: int
    peak_presence: float
    band: tuple[int, int]  # inclusive [z0, z1]
    active_z: list[int]
    centers_yx: dict[int, tuple[float, float]]
    reason: str


def plan_series(
    scores: list[SliceScore],
    *,
    rows: int,
    cols: int,
    slice_spacing_mm: float = 3.0,
    gland_half_height_mm: float = 28.0,
    peak_min: float = 0.35,
    band_rel: float = 0.45,
    band_abs: float = 0.30,
    local_center_min: float = 0.40,
) -> SeriesPlan | None:
    """
    Build a contiguous active band from per-slice localizer scores.

    Parameters
    ----------
    peak_min:
        If the best slice is below this, treat the series as empty.
    band_rel / band_abs:
        Keep slices with ``presence >= max(band_abs, band_rel * peak)``,
        and always keep a geometric window around the peak from spacing.
    local_center_min:
        Below this presence, reuse / interpolate centre instead of trusting
        the localizer's centre head.
    """
    if not scores:
        return None
    scores = sorted(scores, key=lambda s: s.z)
    pres = np.array([s.presence for s in scores], dtype=np.float64)
    zs = [s.z for s in scores]
    peak_i = int(np.argmax(pres))
    peak_z = zs[peak_i]
    peak_p = float(pres[peak_i])
    if peak_p < peak_min:
        return None

    # Geometric half-width in slices (prostate ~5–6 cm tall).
    half_slices = max(2, int(np.ceil(gland_half_height_mm / max(slice_spacing_mm, 1e-3))))
    geo_lo = max(zs[0], peak_z - half_slices)
    geo_hi = min(zs[-1], peak_z + half_slices)

    thr = max(band_abs, band_rel * peak_p)
    # Grow contiguous band from peak while above thr, then union with geo window.
    lo = peak_i
    while lo > 0 and scores[lo - 1].presence >= thr:
        lo -= 1
    hi = peak_i
    while hi < len(scores) - 1 and scores[hi + 1].presence >= thr:
        hi += 1
    z0 = min(scores[lo].z, geo_lo)
    z1 = max(scores[hi].z, geo_hi)
    active = [s.z for s in scores if z0 <= s.z <= z1]

    by_z = {s.z: s for s in scores}
    centers: dict[int, tuple[float, float]] = {}
    # Seed with peak centre (usually most reliable).
    peak_c = by_z[peak_z].center_yx
    for z in active:
        s = by_z[z]
        if s.presence >= local_center_min:
            centers[z] = s.center_yx
        else:
            centers[z] = peak_c

    # Smooth centres along z (3-tap) to kill single-slice outliers.
    for z in active:
        neigh = [centers[zz] for zz in (z - 1, z, z + 1) if zz in centers]
        cy = float(np.mean([c[0] for c in neigh]))
        cx = float(np.mean([c[1] for c in neigh]))
        # Clamp to image.
        cy = float(np.clip(cy, 0, max(rows - 1, 0)))
        cx = float(np.clip(cx, 0, max(cols - 1, 0)))
        centers[z] = (cy, cx)

    return SeriesPlan(
        peak_z=peak_z,
        peak_presence=peak_p,
        band=(z0, z1),
        active_z=active,
        centers_yx=centers,
        reason=(
            f"peak_z={peak_z} p={peak_p:.3f} thr={thr:.3f} "
            f"band=[{z0},{z1}] n={len(active)} half_slices={half_slices}"
        ),
    )
