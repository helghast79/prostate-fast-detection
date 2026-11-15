"""Choose a representative SEG preview frame (densest area, else middle)."""

from __future__ import annotations

import io
from functools import lru_cache
from typing import Any

import numpy as np
import pydicom

from .orthanc_client import OrthancClient


def _referenced_sop_for_frame(ds: pydicom.Dataset, frame_index: int) -> str | None:
    pffg = getattr(ds, "PerFrameFunctionalGroupsSequence", None) or []
    if frame_index < 0 or frame_index >= len(pffg):
        return None
    item = pffg[frame_index]
    der = getattr(item, "DerivationImageSequence", None) or []
    if not der:
        return None
    sources = getattr(der[0], "SourceImageSequence", None) or []
    if not sources:
        return None
    sop = getattr(sources[0], "ReferencedSOPInstanceUID", None)
    return str(sop) if sop else None


def choose_seg_preview_frame(seg_bytes: bytes) -> dict[str, Any]:
    """
    Pick the best SEG frame for a thumbnail.

    Prefers the frame with the largest foreground area; if all empty, uses the
    middle frame. Also returns the referenced SOP UID for that frame when present.
    """
    ds = pydicom.dcmread(io.BytesIO(seg_bytes))
    arr = ds.pixel_array
    if arr.ndim == 2:
        frames = arr[np.newaxis, ...]
    elif arr.ndim == 3:
        frames = arr
    else:
        # Unexpected shape; fall back to middle.
        n = int(getattr(ds, "NumberOfFrames", 1) or 1)
        mid = max(0, n // 2)
        return {
            "frame_index": mid,
            "area": 0,
            "referenced_sop_instance_uid": _referenced_sop_for_frame(ds, mid),
            "reason": "fallback_middle",
        }

    areas = [int((frame > 0).sum()) for frame in frames]
    nonzero = [i for i, area in enumerate(areas) if area > 0]
    if nonzero:
        best = max(nonzero, key=lambda i: areas[i])
        reason = "densest"
    else:
        best = len(frames) // 2
        reason = "middle_empty"
    return {
        "frame_index": int(best),
        "area": int(areas[best]) if areas else 0,
        "referenced_sop_instance_uid": _referenced_sop_for_frame(ds, best),
        "reason": reason,
        "nonzero_frames": len(nonzero),
        "total_frames": len(frames),
    }


@lru_cache(maxsize=512)
def resolve_seg_thumbnail_target(client_key: str, series_id: str) -> dict[str, Any]:
    """
    Cacheable SEG preview target.

    ``client_key`` is unused for lookup and exists only to keep the cache keyed
    by Orthanc endpoint when multiple clients exist.
    """
    del client_key  # cache key only
    # Re-create client from env for cache safety across process lifetime.
    client = OrthancClient()
    series = client.get_series(series_id)
    instances = series.get("Instances") or []
    if not instances:
        raise RuntimeError(f"SEG series {series_id} has no instances")
    instance_id = str(instances[0])
    seg_bytes = client.download_instance_file(instance_id)
    choice = choose_seg_preview_frame(seg_bytes)

    # Prefer the referenced MR/CT slice at the densest SEG plane when resolvable.
    ref_sop = choice.get("referenced_sop_instance_uid")
    if ref_sop:
        ref_instance = client.find_instance_by_sop_uid(str(ref_sop))
        if ref_instance:
            return {
                "mode": "referenced_instance",
                "instance_id": ref_instance,
                "frame_index": None,
                "seg_instance_id": instance_id,
                "seg_frame_index": choice["frame_index"],
                "area": choice["area"],
                "reason": choice["reason"],
            }

    return {
        "mode": "seg_frame",
        "instance_id": instance_id,
        "frame_index": choice["frame_index"],
        "seg_instance_id": instance_id,
        "seg_frame_index": choice["frame_index"],
        "area": choice["area"],
        "reason": choice["reason"],
    }
