"""Local DICOM gland (non-PZ) + peripheral-zone mask volumes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.cascade.io_dicom import DicomSlice, load_dicom_folder
from src.cascade.orthanc_volumes import find_mr_target
from src.geometry import resample_label_mask_to_geometry
from src.seg_masks import extract_zone_frames, referenced_series_uids


def zone_mask_from_seg_dir(
    seg_dir: Path,
    mr_slices: list[DicomSlice],
    *,
    match_tol_mm: float = 1.5,
) -> tuple[np.ndarray, dict]:
    """Map zone SEG frames onto the MR slice grid. Values: 0/1/2."""
    planes: dict[str, dict] = {}
    refs: list[str] = []
    for path in sorted(Path(seg_dir).glob("*.dcm")):
        frames, seg_ds = extract_zone_frames(path.read_bytes())
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
                # Keep PZ (2) over gland (1) on conflict.
                merged = prev["mask"].copy()
                g = f.mask == 1
                p = f.mask == 2
                merged[g] = np.maximum(merged[g], 1)
                merged[p] = 2
                prev["mask"] = merged

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
        # Prefer PZ when both write the same voxel.
        cur = vol[z]
        g = aligned == 1
        p = aligned == 2
        cur[g & (cur == 0)] = 1
        cur[p] = 2
        matched += 1
    return vol, {"matched_planes": matched, "unmatched_planes": unmatched, "referenced_series_uids": refs}


def load_t2_and_zones(t2_dir: Path, seg_dir: Path) -> tuple[list[DicomSlice], np.ndarray, dict]:
    slices = load_dicom_folder(t2_dir)
    if not slices:
        raise ValueError(f"no T2 slices in {t2_dir}")
    mask, info = zone_mask_from_seg_dir(seg_dir, slices)
    return slices, mask, info


def overlay_zone_rgb(
    t2_slice: np.ndarray,
    mask_slice: np.ndarray,
    *,
    lo: float | None = None,
    hi: float | None = None,
) -> np.ndarray:
    """Return uint8 RGB with teal gland (1) and amber PZ (2)."""
    sl = np.asarray(t2_slice, dtype=np.float32)
    if lo is None or hi is None:
        lo_v, hi_v = np.percentile(sl, (1, 99))
    else:
        lo_v, hi_v = float(lo), float(hi)
    if hi_v <= lo_v:
        hi_v = lo_v + 1.0
    gray = np.clip((sl - lo_v) / (hi_v - lo_v), 0, 1)
    rgb = np.stack([gray, gray, gray], axis=-1)
    m = np.asarray(mask_slice)
    g = m == 1
    p = m == 2
    rgb[g, 1] = np.maximum(rgb[g, 1], 0.75)
    rgb[g, 2] = np.maximum(rgb[g, 2], 0.55)
    rgb[p, 0] = np.maximum(rgb[p, 0], 0.95)
    rgb[p, 1] = np.maximum(rgb[p, 1], 0.55)
    rgb[p, 2] *= 0.25
    return (rgb * 255).astype(np.uint8)


def save_zone_overlay_png(
    t2_slice: np.ndarray,
    mask_slice: np.ndarray,
    out_png: Path,
    *,
    lo: float | None = None,
    hi: float | None = None,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay_zone_rgb(t2_slice, mask_slice, lo=lo, hi=hi)).save(out_png)
