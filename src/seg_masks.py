"""Extract prostate binary masks from DICOM SEG (merge all segments)."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from PIL import Image

from .geometry import SliceGeometry, ipp_key
from .zone_labels import CLASS_GLAND, CLASS_PZ, zone_class_for_label


@dataclass
class FrameMask:
    referenced_sop_instance_uid: str | None
    referenced_series_instance_uid: str | None
    segment_numbers: list[int]
    mask: np.ndarray  # uint8 HxW, values 0/1
    geometry: SliceGeometry
    merge_key: str


def _referenced_sop_from_frame(frame_fg: Dataset) -> str | None:
    der = getattr(frame_fg, "DerivationImageSequence", None)
    if der:
        for der_item in der:
            src = getattr(der_item, "SourceImageSequence", None)
            if not src:
                continue
            for src_item in src:
                sop = str(getattr(src_item, "ReferencedSOPInstanceUID", "") or "")
                if sop:
                    return sop
    ref_img = getattr(frame_fg, "ReferencedImageSequence", None)
    if ref_img:
        for item in ref_img:
            sop = str(getattr(item, "ReferencedSOPInstanceUID", "") or "")
            if sop:
                return sop
    return None


def _series_uid_from_shared(seg_ds: Dataset) -> str | None:
    refs = getattr(seg_ds, "ReferencedSeriesSequence", None)
    if refs:
        return str(getattr(refs[0], "SeriesInstanceUID", "") or "") or None
    return None


def _pixel_array_frames(seg_ds: Dataset) -> np.ndarray:
    arr = seg_ds.pixel_array
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim == 4:
        arr = arr[..., 0]
    return np.asarray(arr)


def _shared_or_frame_geometry(seg_ds: Dataset, frame_fg: Dataset) -> SliceGeometry:
    shared = getattr(seg_ds, "SharedFunctionalGroupsSequence", [None])[0]

    def _from(seq_owner: Dataset | None, attr: str, field: str):
        if seq_owner is None:
            return None
        seq = getattr(seq_owner, attr, None)
        if not seq:
            return None
        return getattr(seq[0], field, None)

    iop = _from(frame_fg, "PlaneOrientationSequence", "ImageOrientationPatient")
    if iop is None:
        iop = _from(shared, "PlaneOrientationSequence", "ImageOrientationPatient")
    ipp = _from(frame_fg, "PlanePositionSequence", "ImagePositionPatient")
    if ipp is None:
        ipp = _from(shared, "PlanePositionSequence", "ImagePositionPatient")
    spacing = _from(frame_fg, "PixelMeasuresSequence", "PixelSpacing")
    if spacing is None:
        spacing = _from(shared, "PixelMeasuresSequence", "PixelSpacing")
    if iop is None or ipp is None or spacing is None:
        raise ValueError("SEG frame missing IOP/IPP/PixelSpacing")
    return SliceGeometry(
        ipp=np.array([float(x) for x in ipp], dtype=np.float64),
        iop=np.array([float(x) for x in iop], dtype=np.float64),
        spacing=np.array([float(x) for x in spacing], dtype=np.float64),
        rows=int(seg_ds.Rows),
        cols=int(seg_ds.Columns),
    )


def all_segment_numbers(seg_ds: Dataset) -> list[int]:
    segments = list(getattr(seg_ds, "SegmentSequence", []) or [])
    nums = sorted({int(getattr(s, "SegmentNumber", 0)) for s in segments if getattr(s, "SegmentNumber", None)})
    return [n for n in nums if n > 0] or [1]


def _is_seminal_vesicle(label: str) -> bool:
    text = " ".join(str(label or "").strip().lower().split())
    return "seminal" in text or text in {"sv", "s.v.", "s.v"}


def extract_prostate_frames(
    seg_bytes: bytes, *, exclude_seminal_vesicles: bool = False
) -> tuple[list[FrameMask], Dataset]:
    """
    Extract per-plane prostate masks, OR-merging **all** segments.

    Two-segment zone labels (e.g. TZ+CZ / PZ) are joined into one whole-gland mask.
    Seminal-vesicle segments are not prostate; drop them with
    ``exclude_seminal_vesicles=True``.
    """
    seg_ds = pydicom.dcmread(io.BytesIO(seg_bytes))
    segment_numbers = all_segment_numbers(seg_ds)
    if exclude_seminal_vesicles:
        sv_numbers = {
            int(getattr(s, "SegmentNumber", 0))
            for s in getattr(seg_ds, "SegmentSequence", []) or []
            if _is_seminal_vesicle(getattr(s, "SegmentLabel", ""))
        }
        segment_numbers = [n for n in segment_numbers if n not in sv_numbers] or segment_numbers
    series_uid = _series_uid_from_shared(seg_ds)
    pixels = _pixel_array_frames(seg_ds)
    per_frame = list(getattr(seg_ds, "PerFrameFunctionalGroupsSequence", []) or [])
    if len(per_frame) != pixels.shape[0]:
        n = min(len(per_frame), pixels.shape[0])
        per_frame = per_frame[:n]
        pixels = pixels[:n]

    wanted = set(segment_numbers)
    merged: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}

    for idx, frame_fg in enumerate(per_frame):
        seg_id = getattr(frame_fg, "SegmentIdentificationSequence", None)
        frame_seg_num = None
        if seg_id:
            frame_seg_num = int(getattr(seg_id[0], "ReferencedSegmentNumber", -1))
        if frame_seg_num is not None and frame_seg_num not in wanted:
            continue

        try:
            geometry = _shared_or_frame_geometry(seg_ds, frame_fg)
        except ValueError:
            continue

        sop_uid = _referenced_sop_from_frame(frame_fg)
        if sop_uid:
            key = f"sop:{sop_uid}"
        else:
            key = "ipp:" + ",".join(str(x) for x in ipp_key(geometry.ipp, decimals=3))

        mask = (pixels[idx] > 0).astype(np.uint8)
        prev = merged.get(key)
        if prev is None:
            merged[key] = mask
            meta[key] = {"sop": sop_uid, "geometry": geometry}
        else:
            merged[key] = (prev.astype(bool) | mask.astype(bool)).astype(np.uint8)
            if meta[key]["sop"] is None and sop_uid:
                meta[key]["sop"] = sop_uid

    frames = [
        FrameMask(
            referenced_sop_instance_uid=meta[key]["sop"],
            referenced_series_instance_uid=series_uid,
            segment_numbers=segment_numbers,
            mask=mask,
            geometry=meta[key]["geometry"],
            merge_key=key,
        )
        for key, mask in merged.items()
    ]
    return frames, seg_ds


def referenced_series_uids(seg_ds: Dataset) -> list[str]:
    uids: list[str] = []
    for item in getattr(seg_ds, "ReferencedSeriesSequence", []) or []:
        uid = str(getattr(item, "SeriesInstanceUID", "") or "")
        if uid and uid not in uids:
            uids.append(uid)
    return uids


def _segment_number_to_zone_class(seg_ds: Dataset) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for seg in list(getattr(seg_ds, "SegmentSequence", []) or []):
        num = int(getattr(seg, "SegmentNumber", 0) or 0)
        if num <= 0:
            continue
        label = str(getattr(seg, "SegmentLabel", "") or getattr(seg, "SegmentDescription", "") or "")
        zone = zone_class_for_label(label)
        if zone is not None:
            mapping[num] = zone
    return mapping


def extract_zone_frames(seg_bytes: bytes) -> tuple[list[FrameMask], Dataset]:
    """
    Extract per-plane **class-index** masks:
      0 = background, 1 = gland (non-PZ), 2 = peripheral zone.

    On overlap, PZ (2) overwrites gland (1).
    """
    seg_ds = pydicom.dcmread(io.BytesIO(seg_bytes))
    class_by_seg = _segment_number_to_zone_class(seg_ds)
    if CLASS_GLAND not in class_by_seg.values() or CLASS_PZ not in class_by_seg.values():
        return [], seg_ds

    series_uid = _series_uid_from_shared(seg_ds)
    pixels = _pixel_array_frames(seg_ds)
    per_frame = list(getattr(seg_ds, "PerFrameFunctionalGroupsSequence", []) or [])
    if len(per_frame) != pixels.shape[0]:
        n = min(len(per_frame), pixels.shape[0])
        per_frame = per_frame[:n]
        pixels = pixels[:n]

    # plane_key -> {class_id: binary mask}
    plane_bins: dict[str, dict[int, np.ndarray]] = {}
    meta: dict[str, dict] = {}

    for idx, frame_fg in enumerate(per_frame):
        seg_id = getattr(frame_fg, "SegmentIdentificationSequence", None)
        frame_seg_num = None
        if seg_id:
            frame_seg_num = int(getattr(seg_id[0], "ReferencedSegmentNumber", -1))
        zone = class_by_seg.get(frame_seg_num) if frame_seg_num is not None else None
        if zone is None:
            continue

        try:
            geometry = _shared_or_frame_geometry(seg_ds, frame_fg)
        except ValueError:
            continue

        sop_uid = _referenced_sop_from_frame(frame_fg)
        if sop_uid:
            key = f"sop:{sop_uid}"
        else:
            key = "ipp:" + ",".join(str(x) for x in ipp_key(geometry.ipp, decimals=3))

        bin_mask = (pixels[idx] > 0).astype(np.uint8)
        buckets = plane_bins.setdefault(key, {})
        prev = buckets.get(zone)
        if prev is None:
            buckets[zone] = bin_mask
        else:
            buckets[zone] = (prev.astype(bool) | bin_mask.astype(bool)).astype(np.uint8)
        if key not in meta:
            meta[key] = {"sop": sop_uid, "geometry": geometry}
        elif meta[key]["sop"] is None and sop_uid:
            meta[key]["sop"] = sop_uid

    frames: list[FrameMask] = []
    for key, buckets in plane_bins.items():
        h, w = next(iter(buckets.values())).shape
        index = np.zeros((h, w), dtype=np.uint8)
        if CLASS_GLAND in buckets:
            index[buckets[CLASS_GLAND] > 0] = CLASS_GLAND
        if CLASS_PZ in buckets:
            index[buckets[CLASS_PZ] > 0] = CLASS_PZ
        frames.append(
            FrameMask(
                referenced_sop_instance_uid=meta[key]["sop"],
                referenced_series_instance_uid=series_uid,
                segment_numbers=sorted(class_by_seg.keys()),
                mask=index,
                geometry=meta[key]["geometry"],
                merge_key=key,
            )
        )
    return frames, seg_ds


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbor resize for binary or class-index masks."""
    img = Image.fromarray(mask.astype(np.uint8), mode="L")
    img = img.resize((size, size), resample=Image.Resampling.NEAREST)
    return np.asarray(img).astype(np.uint8)


def resize_binary_mask(mask: np.ndarray, size: int) -> np.ndarray:
    return (resize_mask((mask.astype(np.uint8) * 255), size) > 127).astype(np.uint8)


def save_mask_png(mask: np.ndarray, path: str) -> None:
    """Save binary (0/1) as 0/255, or class-index masks as-is (0,1,2,…)."""
    arr = mask.astype(np.uint8)
    if arr.max() <= 1:
        arr = arr * 255
    Image.fromarray(arr, mode="L").save(path)


def save_image_array(image: np.ndarray, path: str) -> None:
    Image.fromarray(image.astype(np.uint8), mode="L").save(path)
