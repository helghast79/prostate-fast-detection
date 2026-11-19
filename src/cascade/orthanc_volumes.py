"""Load MR series and SEG masks from Orthanc onto an MR slice grid."""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.geometry import resample_mask_to_geometry
from src.orthanc_client import OrthancClient
from src.seg_masks import extract_prostate_frames, referenced_series_uids

from .io_dicom import DicomSlice, read_dicom_bytes, sort_slices_by_ipp


def load_mr_slices(client: OrthancClient, series_uid: str) -> list[DicomSlice]:
    series_id = client.find_series_by_uid(series_uid)
    if not series_id:
        return []
    series = client.get_series(series_id)
    out = [read_dicom_bytes(client.download_instance_file(iid)) for iid in series.get("Instances") or []]
    return sort_slices_by_ipp(out)


def keep_lowest_signal_per_position(slices: list[DicomSlice]) -> list[DicomSlice]:
    """Multi-b DWI series: keep one image per slice position, the darkest (highest b)."""
    if not slices:
        return []
    normal = slices[0].geometry.normal
    groups: dict[float, list[DicomSlice]] = {}
    for s in slices:
        groups.setdefault(round(float(s.geometry.ipp @ normal), 2), []).append(s)
    picked = [min(g, key=lambda s: float(s.pixels.mean())) for g in groups.values()]
    return sort_slices_by_ipp(picked)


def find_mr_target(
    frame_sop: str | None,
    seg_ipp: np.ndarray,
    by_sop: dict[str, DicomSlice],
    mr_slices: list[DicomSlice],
    match_tol_mm: float,
) -> tuple[DicomSlice | None, str]:
    if frame_sop and frame_sop in by_sop:
        return by_sop[frame_sop], "sop"
    if not mr_slices:
        return None, "none"
    normal = mr_slices[0].geometry.normal
    seg_z = float(np.asarray(seg_ipp) @ normal)
    zs = np.array([float(s.geometry.ipp @ normal) for s in mr_slices], dtype=np.float64)
    best_i = int(np.argmin(np.abs(zs - seg_z)))
    best_dz = float(abs(zs[best_i] - seg_z))
    if best_dz > match_tol_mm:
        return None, f"geom_miss:{best_dz:.3f}"
    return mr_slices[best_i], f"geom_z:{best_dz:.3f}"


def seg_to_mask_volume(
    client: OrthancClient,
    seg_orthanc_series_id: str,
    mr_slices: list[DicomSlice],
    *,
    label_filter: Callable[[str], bool] | None = None,
    fractional_threshold: float | None = None,
    match_tol_mm: float = 1.5,
) -> tuple[np.ndarray, dict]:
    """OR-merge the selected SEG segments onto the (IPP-sorted) MR slice grid."""
    planes: dict[str, dict] = {}
    refs: list[str] = []
    for iid in client.get_series(seg_orthanc_series_id).get("Instances") or []:
        frames, seg_ds = extract_prostate_frames(
            client.download_instance_file(iid),
            label_filter=label_filter,
            fractional_threshold=fractional_threshold,
        )
        refs = referenced_series_uids(seg_ds) or refs
        for f in frames:
            prev = planes.get(f.merge_key)
            if prev is None:
                planes[f.merge_key] = {"mask": f.mask, "geometry": f.geometry, "sop": f.referenced_sop_instance_uid}
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
        aligned = resample_mask_to_geometry(plane["mask"], plane["geometry"], target.geometry)
        z = z_of[target.sop_instance_uid]
        vol[z] = np.maximum(vol[z], aligned.astype(np.uint8))
        matched += 1
    return vol, {"matched_planes": matched, "unmatched_planes": unmatched, "referenced_series_uids": refs}


def segmentation_type(client: OrthancClient, series: dict) -> str | None:
    """SegmentationType (BINARY / FRACTIONAL) of a manifest SEG series."""
    try:
        iid = (client.get_series(series["orthanc_series_id"]).get("Instances") or [None])[0]
        return str(client.get_instance_tags(iid, simplify=True).get("SegmentationType") or "") or None
    except Exception:  # noqa: BLE001
        return None
