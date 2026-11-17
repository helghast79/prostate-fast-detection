"""MRI intensity normalization helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .decisions import CASCADE_DECISIONS


def percentile_normalize(
    volume: np.ndarray,
    low_pct: float | None = None,
    high_pct: float | None = None,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Clip to percentiles and scale to [0, 1]. Operates on the full volume."""
    low_pct = CASCADE_DECISIONS.percentile_low if low_pct is None else low_pct
    high_pct = CASCADE_DECISIONS.percentile_high if high_pct is None else high_pct
    arr = np.asarray(volume, dtype=np.float32)
    sample = arr[mask.astype(bool)] if mask is not None else arr
    if sample.size == 0:
        sample = arr
    lo = float(np.percentile(sample, low_pct))
    hi = float(np.percentile(sample, high_pct))
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    meta = {
        "method": "percentile_01",
        "percentile_low": low_pct,
        "percentile_high": high_pct,
        "intensity_low": lo,
        "intensity_high": hi,
    }
    return out, meta


def zscore_normalize(
    volume: np.ndarray,
    mask: np.ndarray | None = None,
    eps: float = 1e-6,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Optional alternative: zero-mean unit-variance (not the cascade default)."""
    arr = np.asarray(volume, dtype=np.float32)
    sample = arr[mask.astype(bool)] if mask is not None else arr
    mean = float(sample.mean()) if sample.size else 0.0
    std = float(sample.std()) if sample.size else 1.0
    if std < eps:
        std = 1.0
    out = ((arr - mean) / std).astype(np.float32)
    return out, {"method": "zscore", "mean": mean, "std": std}
