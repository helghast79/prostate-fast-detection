"""Text + JSON series catalog for prostate Orthanc review."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


MR_TAGS = ("t2", "adc", "dwi")
UID_RE = re.compile(r"^[0-9]+(\.[0-9]+)+$")


@dataclass
class SeriesEntry:
    modality: str
    series_description: str
    series_instance_uid: str
    orthanc_series_id: str = ""
    orthanc_study_id: str = ""
    preview_instance_id: str = ""
    series_number: str = ""
    referenced_series_uids: list[str] = field(default_factory=list)
    segment_labels: list[str] = field(default_factory=list)


@dataclass
class PatientEntry:
    patient_id: str
    orthanc_patient_ids: list[str] = field(default_factory=list)
    series: list[SeriesEntry] = field(default_factory=list)


def format_series_line(entry: SeriesEntry) -> str:
    desc = " ".join(str(entry.series_description or "").split())
    return f"{entry.modality} {desc} {entry.series_instance_uid}".strip()


def parse_series_line(line: str) -> tuple[str, str, str] | None:
    """Parse '[Modality] [Series Description] [SeriesInstanceUID]'."""
    parts = line.strip().split()
    if len(parts) < 2:
        return None
    modality = parts[0].upper()
    uid = parts[-1]
    if not UID_RE.match(uid):
        return None
    description = " ".join(parts[1:-1])
    return modality, description, uid


def write_text_manifest(path: Path, patients: Iterable[PatientEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks: list[str] = []
    for patient in patients:
        lines = [patient.patient_id]
        for series in patient.series:
            lines.append(format_series_line(series))
        blocks.append("\n".join(lines))
    text = "\n\n".join(blocks)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def read_text_manifest(path: Path) -> list[PatientEntry]:
    raw = path.read_text(encoding="utf-8")
    patients: list[PatientEntry] = []
    current: PatientEntry | None = None

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            if current is not None:
                patients.append(current)
                current = None
            continue
        parsed = parse_series_line(stripped)
        if parsed is None:
            if current is not None:
                patients.append(current)
            current = PatientEntry(patient_id=stripped)
            continue
        if current is None:
            raise ValueError(f"Series line before PatientID: {stripped}")
        modality, description, uid = parsed
        current.series.append(
            SeriesEntry(
                modality=modality,
                series_description=description,
                series_instance_uid=uid,
            )
        )

    if current is not None:
        patients.append(current)
    return patients


def patients_to_json_dict(patients: list[PatientEntry]) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "patient_count": len(patients),
        "series_count": sum(len(p.series) for p in patients),
        "patients": [
            {
                "patient_id": p.patient_id,
                "orthanc_patient_ids": list(p.orthanc_patient_ids),
                "series": [asdict(s) for s in p.series],
            }
            for p in patients
        ],
    }


def write_json_manifest(path: Path, patients: list[PatientEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(patients_to_json_dict(patients), indent=2) + "\n", encoding="utf-8")


def read_json_manifest(path: Path) -> list[PatientEntry]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    patients: list[PatientEntry] = []
    for item in payload.get("patients") or []:
        series = [
            SeriesEntry(
                modality=str(s.get("modality", "")).upper(),
                series_description=str(s.get("series_description", "")),
                series_instance_uid=str(s.get("series_instance_uid", "")),
                orthanc_series_id=str(s.get("orthanc_series_id", "")),
                orthanc_study_id=str(s.get("orthanc_study_id", "")),
                preview_instance_id=str(s.get("preview_instance_id", "")),
                series_number=str(s.get("series_number", "")),
                referenced_series_uids=list(s.get("referenced_series_uids") or []),
                segment_labels=list(s.get("segment_labels") or []),
            )
            for s in (item.get("series") or [])
        ]
        patients.append(
            PatientEntry(
                patient_id=str(item.get("patient_id", "")),
                orthanc_patient_ids=list(item.get("orthanc_patient_ids") or []),
                series=series,
            )
        )
    return patients


LESION_SEG_CHOICES = ("binary", "prob60", "no")
LESION_TOKENS = ("lesion", "neoplasm", "heatmap", "cspca", "cancer", "tumor", "tumour")


def is_lesion_candidate(description: str, segment_labels: list[str] | None = None) -> bool:
    blob = " ".join([description or ""] + list(segment_labels or [])).lower()
    return any(tok in blob for tok in LESION_TOKENS)


def suggest_lesion_seg(
    description: str,
    segment_labels: list[str] | None,
    segmentation_type: str | None,
    patient_has_plain_binary: bool,
) -> str | None:
    """binary = any marked pixel; prob60 = probability ≥ 0.6; no = not a lesion label."""
    labels = list(segment_labels or [])
    if not is_lesion_candidate(description, labels):
        return None
    blob = " ".join([description or ""] + labels).lower()
    if "broad" in blob or "core roi" in blob:
        return "no"
    probabilistic = (segmentation_type or "").upper() == "FRACTIONAL" or "%" in blob
    if probabilistic:
        return "no" if patient_has_plain_binary else "prob60"
    return "binary"


def suggest_mr_tag(description: str) -> str | None:
    text = description.lower()
    if "adc" in text:
        return "adc"
    if (
        "dwi" in text
        or "diff" in text
        or "bval" in text
        or "b-value" in text
        or re.search(r"\bs?b\d{3,4}\b", text)
    ):
        return "dwi"
    if re.search(r"\bt2\b", text) or "t2_" in text or "t2w" in text:
        return "t2"
    return None


def suggest_prostate_regions(description: str, segment_labels: list[str] | None = None) -> bool | None:
    blob = " ".join(
        [description or ""] + list(segment_labels or [])
    ).lower()
    if not blob.strip():
        return None
    negative = (
        "neoplasm",
        "lesion",
        "heatmap",
        "cspca",
        "probability",
        "cancer",
        "tumor",
    )
    if any(tok in blob for tok in negative):
        return False
    positive = (
        "prostate",
        "transition zone",
        "peripheral",
        "tz+cz",
        " tz",
        "pz",
        "central zone",
        "regions",
    )
    if any(tok in blob for tok in positive):
        return True
    return None


def enrich_patients(
    patients_raw: list[dict[str, Any]],
    annotations: dict[str, Any],
    seg_type: Callable[[dict[str, Any]], str | None],
) -> list[dict]:
    """Saved review tags, falling back to suggestions for untagged series."""
    ann_series = annotations.get("series") or {}
    out: list[dict] = []
    for patient in patients_raw:
        series_out = []
        lesion_segs = [
            s for s in patient.get("series") or []
            if str(s.get("modality", "")).upper() == "SEG"
            and is_lesion_candidate(str(s.get("series_description", "")), s.get("segment_labels"))
        ]
        types = {s.get("series_instance_uid"): seg_type(s) for s in lesion_segs}
        has_plain_binary = any(
            suggest_lesion_seg(str(s.get("series_description", "")), s.get("segment_labels"),
                               types[s.get("series_instance_uid")], False) == "binary"
            and not (ann_series.get(s.get("series_instance_uid")) or {}).get("discarded")
            for s in lesion_segs
        )
        for series in patient.get("series") or []:
            uid = series.get("series_instance_uid", "")
            ann = ann_series.get(uid) or {}
            modality = str(series.get("modality", "")).upper()
            item = dict(series)
            discarded = bool(ann.get("discarded"))
            item["discarded"] = discarded
            if modality == "MR":
                tag = ann.get("mr_tag")
                if tag not in MR_TAGS:
                    tag = (
                        None
                        if discarded
                        else suggest_mr_tag(str(series.get("series_description", "")))
                    )
                item["mr_tag"] = tag if tag in MR_TAGS else None
                item["is_prostate_regions"] = None
            else:
                flag = ann.get("is_prostate_regions")
                if flag is None and not discarded:
                    flag = suggest_prostate_regions(
                        str(series.get("series_description", "")),
                        list(series.get("segment_labels") or []),
                    )
                item["mr_tag"] = None
                item["is_prostate_regions"] = flag
                item["segmentation_type"] = types.get(uid)
                choice = ann.get("lesion_seg")
                if choice not in LESION_SEG_CHOICES and not discarded:
                    choice = suggest_lesion_seg(
                        str(series.get("series_description", "")),
                        list(series.get("segment_labels") or []),
                        types.get(uid),
                        has_plain_binary,
                    )
                item["lesion_seg"] = choice if choice in LESION_SEG_CHOICES else None
            item["annotation_saved"] = uid in ann_series
            series_out.append(item)
        out.append(
            {
                "patient_id": patient.get("patient_id"),
                "orthanc_patient_ids": patient.get("orthanc_patient_ids") or [],
                "series": series_out,
            }
        )
    return out
