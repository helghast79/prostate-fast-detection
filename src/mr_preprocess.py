"""MRI intensity helpers: DICOM → percentile-normalized grayscale PNG array."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pydicom
from PIL import Image


def dicom_bytes_to_percentile_png(
    dicom_bytes: bytes,
    size: int,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
) -> tuple[bytes, dict[str, Any]]:
    ds = pydicom.dcmread(io.BytesIO(dicom_bytes))
    arr = np.asarray(ds.pixel_array, dtype=np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept

    lo = float(np.percentile(arr, low_pct))
    hi = float(np.percentile(arr, high_pct))
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    gray = (norm * 255.0).astype(np.uint8)
    img = Image.fromarray(gray, mode="L")
    if img.width != size or img.height != size:
        img = img.resize((size, size), resample=Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    meta = {
        "percentile_low": low_pct,
        "percentile_high": high_pct,
        "intensity_low": lo,
        "intensity_high": hi,
        "rescale_slope": slope,
        "rescale_intercept": intercept,
    }
    return buf.getvalue(), meta
