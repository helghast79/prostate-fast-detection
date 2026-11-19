"""Local DICOM lesion mask volumes (T2-aligned)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.cascade.io_dicom import DicomSlice, load_dicom_folder
from src.cascade.orthanc_volumes import find_mr_target
from src.geometry import resample_label_mask_to_geometry
from src.seg_masks import (
    extract_prostate_frames,
    is_lesion_label,
    probability_bin_at_least,
    referenced_series_uids,
)


def _lesion_label_filter(label: str) -> bool:
    text = str(label or "").strip()
    if not text:
        return True  # unlabeled SEG: keep all segments
    if is_lesion_label(text) or "neoplasm" in text.lower():
        return True
    # Probability-binned neoplasm labels, e.g. "Neoplasm (60-70%)"
    if probability_bin_at_least(text, 60.0):
        return True
    return False


def lesion_mask_from_seg_dir(
    seg_dir: Path,
    mr_slices: list[DicomSlice],
    *,
    match_tol_mm: float = 1.5,
    fractional_threshold: float = 0.6,
) -> tuple[np.ndarray, dict]:
    """Map lesion SEG frames onto the MR slice grid. Values: 0/1."""
    planes: dict[str, dict] = {}
    refs: list[str] = []
    for path in sorted(Path(seg_dir).glob("*.dcm")):
        frames, seg_ds = extract_prostate_frames(
            path.read_bytes(),
            label_filter=_lesion_label_filter,
            fractional_threshold=fractional_threshold,
        )
        refs = referenced_series_uids(seg_ds) or refs
        for f in frames:
            prev = planes.get(f.merge_key)
            if prev is None:
                planes[f.merge_key] = {
                    "mask": f.mask.astype(np.uint8),
                    "geometry": f.geometry,
                    "sop": f.referenced_sop_instance_uid,
                }
            else:
                prev["mask"] = (prev["mask"].astype(bool) | f.mask.astype(bool)).astype(np.uint8)

    rows, cols = mr_slices[0].pixels.shape
    vol = np.zeros((len(mr_slices), rows, cols), dtype=np.uint8)
    by_sop = {s.sop_instance_uid: s for s in mr_slices}
    z_of = {s.sop_instance_uid: i for i, s in enumerate(mr_slices)}
    matched = unmatched = 0
    for plane in planes.values():
        target, _ = find_mr_target(plane["sop"], plane["geometry"].ipp, by_sop, mr_slices, match_tol_mm)
        if target is None:
            unmatched += 1
            continue
        aligned = resample_label_mask_to_geometry(plane["mask"], plane["geometry"], target.geometry)
        z = z_of[target.sop_instance_uid]
        vol[z] = np.maximum(vol[z], (aligned > 0).astype(np.uint8))
        matched += 1
    return vol, {
        "matched_planes": matched,
        "unmatched_planes": unmatched,
        "referenced_series_uids": refs,
    }


def load_t2_and_lesion(t2_dir: Path, seg_dir: Path) -> tuple[list[DicomSlice], np.ndarray, dict]:
    slices = load_dicom_folder(t2_dir)
    if not slices:
        raise ValueError(f"no T2 slices in {t2_dir}")
    mask, info = lesion_mask_from_seg_dir(seg_dir, slices)
    return slices, mask, info


def overlay_lesion_rgb(
    t2_slice: np.ndarray,
    mask_slice: np.ndarray,
    *,
    lo: float | None = None,
    hi: float | None = None,
) -> np.ndarray:
    """Return uint8 RGB with magenta lesion overlay."""
    sl = np.asarray(t2_slice, dtype=np.float32)
    if lo is None or hi is None:
        lo_v, hi_v = np.percentile(sl, (1, 99))
    else:
        lo_v, hi_v = float(lo), float(hi)
    if hi_v <= lo_v:
        hi_v = lo_v + 1.0
    gray = np.clip((sl - lo_v) / (hi_v - lo_v), 0, 1)
    rgb = np.stack([gray, gray, gray], axis=-1)
    m = np.asarray(mask_slice) > 0
    rgb[m, 0] = np.maximum(rgb[m, 0], 0.95)
    rgb[m, 1] *= 0.25
    rgb[m, 2] = np.maximum(rgb[m, 2], 0.85)
    return (rgb * 255).astype(np.uint8)


def save_lesion_overlay_png(
    t2_slice: np.ndarray,
    mask_slice: np.ndarray,
    out_png: Path,
    *,
    lo: float | None = None,
    hi: float | None = None,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay_lesion_rgb(t2_slice, mask_slice, lo=lo, hi=hi)).save(out_png)
