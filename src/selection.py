"""Select t2 MR + prostate-regions SEG pairs from review annotations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .zone_labels import has_zone_supervision


@dataclass
class ProstatePair:
    patient_id: str
    orthanc_patient_ids: list[str]
    t2_series_instance_uid: str
    t2_orthanc_series_id: str
    t2_description: str
    seg_series_instance_uid: str
    seg_orthanc_series_id: str
    seg_description: str
    segment_labels: list[str]
    referenced_series_uids: list[str]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_t2_prostate_pairs(
    manifest_path: Path,
    annotations_path: Path,
) -> list[ProstatePair]:
    """
    Keep patients that have:
      - at least one non-discarded MR tagged ``t2``
      - at least one non-discarded SEG with ``is_prostate_regions=True``

    Prefer a SEG that references the chosen t2 series.
    """
    manifest = _load_json(manifest_path)
    annotations = _load_json(annotations_path).get("series") or {}

    pairs: list[ProstatePair] = []
    for patient in manifest.get("patients") or []:
        patient_id = str(patient.get("patient_id") or "")
        series_list = patient.get("series") or []
        by_uid = {str(s.get("series_instance_uid")): s for s in series_list}

        t2_candidates: list[dict] = []
        seg_candidates: list[dict] = []
        for series in series_list:
            uid = str(series.get("series_instance_uid") or "")
            ann = annotations.get(uid) or {}
            if ann.get("discarded"):
                continue
            modality = str(series.get("modality") or "").upper()
            if modality == "MR" and ann.get("mr_tag") == "t2":
                t2_candidates.append(series)
            if modality == "SEG" and ann.get("is_prostate_regions") is True:
                seg_candidates.append(series)

        if not t2_candidates or not seg_candidates:
            continue

        # Prefer the first t2; then a SEG that references it.
        t2 = t2_candidates[0]
        t2_uid = str(t2.get("series_instance_uid") or "")
        linked = [
            s
            for s in seg_candidates
            if t2_uid in (s.get("referenced_series_uids") or [])
        ]
        seg = linked[0] if linked else seg_candidates[0]

        pairs.append(
            ProstatePair(
                patient_id=patient_id,
                orthanc_patient_ids=list(patient.get("orthanc_patient_ids") or []),
                t2_series_instance_uid=t2_uid,
                t2_orthanc_series_id=str(t2.get("orthanc_series_id") or ""),
                t2_description=str(t2.get("series_description") or ""),
                seg_series_instance_uid=str(seg.get("series_instance_uid") or ""),
                seg_orthanc_series_id=str(seg.get("orthanc_series_id") or ""),
                seg_description=str(seg.get("series_description") or ""),
                segment_labels=list(seg.get("segment_labels") or []),
                referenced_series_uids=list(seg.get("referenced_series_uids") or []),
            )
        )

    pairs.sort(key=lambda p: p.patient_id)
    return pairs


def select_t2_zone_pairs(
    manifest_path: Path,
    annotations_path: Path,
) -> list[ProstatePair]:
    """Like ``select_t2_prostate_pairs`` but require TZ/CZ + PZ zone labels."""
    return [
        p
        for p in select_t2_prostate_pairs(manifest_path, annotations_path)
        if has_zone_supervision(p.segment_labels)
    ]
