"""Read-only helpers to index ProCAncer-I DICOM studies on disk."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pydicom

from src.series_manifest import suggest_mr_tag
from src.zone_labels import has_zone_supervision, zone_class_for_label

_AXIAL_RE = re.compile(r"(?:^|[^a-z])(?:tra|ax|axial)(?:[^a-z]|$)", re.I)
_FINAL_RE = re.compile(r"_final\b", re.I)
_PROSTATE_RE = re.compile(r"prostate", re.I)
_AUTOSEG_RE = re.compile(r"autoseg|automatic", re.I)


@dataclass
class LocalSeries:
    series_dir: str
    series_instance_uid: str
    modality: str
    series_description: str
    n_instances: int
    mr_tag: str | None = None
    axial: bool = False
    segment_labels: list[str] = field(default_factory=list)
    referenced_series_uids: list[str] = field(default_factory=list)
    has_zone_labels: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_masks_list_name(name: str) -> tuple[str, str]:
    """``PatientID_StudyInstanceUID`` → (patient_id, study_uid)."""
    marker = "_1."
    if marker not in name:
        raise ValueError(f"expected PatientID_StudyUID, got {name}")
    i = name.index(marker)
    return name[:i], name[i + 1 :]


def parse_lesion_list_name(name: str) -> str:
    """Symlink basename is the StudyInstanceUID."""
    return name


def resolve_study_dir(dicom_root: Path, patient_id: str, study_uid: str) -> Path:
    return dicom_root / patient_id / study_uid


def _first_dcm(series_dir: Path) -> Path | None:
    try:
        files = sorted(series_dir.glob("*.dcm"))
        if files:
            return files[0]
        files = sorted(p for p in series_dir.iterdir() if p.is_file())
    except (PermissionError, OSError):
        return None
    return files[0] if files else None


def _ref_series_uids(ds: pydicom.Dataset) -> list[str]:
    out: list[str] = []
    for item in getattr(ds, "ReferencedSeriesSequence", []) or []:
        uid = str(getattr(item, "SeriesInstanceUID", "") or "")
        if uid and uid not in out:
            out.append(uid)
    return out


def scan_series_dir(series_dir: Path) -> LocalSeries | None:
    first = _first_dcm(series_dir)
    if first is None:
        return None
    try:
        ds = pydicom.dcmread(str(first), stop_before_pixels=True, force=True)
    except Exception:  # noqa: BLE001
        return None
    modality = str(getattr(ds, "Modality", "") or "").upper()
    desc = str(getattr(ds, "SeriesDescription", "") or "")
    uid = str(getattr(ds, "SeriesInstanceUID", "") or series_dir.name)
    try:
        n = sum(1 for p in series_dir.iterdir() if p.is_file())
    except (PermissionError, OSError):
        n = 0
    labels: list[str] = []
    refs: list[str] = []
    if modality == "SEG":
        labels = [
            str(getattr(s, "SegmentLabel", "") or getattr(s, "SegmentDescription", "") or "")
            for s in getattr(ds, "SegmentSequence", []) or []
        ]
        refs = _ref_series_uids(ds)
    tag = suggest_mr_tag(desc) if modality == "MR" else None
    return LocalSeries(
        series_dir=str(series_dir),
        series_instance_uid=uid,
        modality=modality,
        series_description=desc,
        n_instances=n,
        mr_tag=tag,
        axial=bool(tag == "t2" and _AXIAL_RE.search(desc)),
        segment_labels=labels,
        referenced_series_uids=refs,
        has_zone_labels=has_zone_supervision(labels),
    )


def scan_study(study_dir: Path) -> list[LocalSeries]:
    if not study_dir.is_dir():
        return []
    out: list[LocalSeries] = []
    try:
        series_dirs = sorted(study_dir.iterdir())
    except (PermissionError, OSError):
        return []
    for ser in series_dirs:
        if not ser.is_dir():
            continue
        try:
            meta = scan_series_dir(ser)
        except (PermissionError, OSError):
            continue
        if meta is not None:
            out.append(meta)
    return out


def zone_seg_rank(series: LocalSeries) -> tuple[int, int, int]:
    """Higher is better: Final prostate > prostate > other zone SEG > autoseg."""
    desc = series.series_description
    auto = 0 if _AUTOSEG_RE.search(desc) else 1
    final = 1 if _FINAL_RE.search(desc) else 0
    prostate = 1 if _PROSTATE_RE.search(desc) else 0
    return (auto, final, prostate)


def pick_zone_seg(series_list: list[LocalSeries]) -> LocalSeries | None:
    cands = [s for s in series_list if s.modality == "SEG" and s.has_zone_labels]
    if not cands:
        return None
    return max(cands, key=zone_seg_rank)


def pick_axial_t2(series_list: list[LocalSeries], preferred_uids: list[str] | None = None) -> LocalSeries | None:
    preferred = set(preferred_uids or [])
    t2s = [s for s in series_list if s.modality == "MR" and s.mr_tag == "t2"]
    axial = [s for s in t2s if s.axial]
    pool = axial or t2s
    if not pool:
        return None
    if preferred:
        hit = [s for s in pool if s.series_instance_uid in preferred]
        if hit:
            return max(hit, key=lambda s: (s.axial, s.n_instances))
    return max(pool, key=lambda s: (s.axial, s.n_instances))


def iter_masks_studies(masks_list: Path, dicom_root: Path):
    """Yield (patient_id, study_uid, study_dir, list_symlink)."""
    for link in sorted(masks_list.iterdir()):
        if not link.is_symlink() and not link.is_file():
            continue
        try:
            patient_id, study_uid = parse_masks_list_name(link.name)
        except ValueError:
            # fallback: symlink target .../PCa-.../StudyUID
            target = Path(os.readlink(link)) if link.is_symlink() else link
            parts = target.parts
            i = next(i for i, x in enumerate(parts) if x.startswith("PCa-"))
            patient_id, study_uid = parts[i], parts[i + 1]
        yield patient_id, study_uid, resolve_study_dir(dicom_root, patient_id, study_uid), link


def _patient_study_from_lesion_link(link: Path) -> tuple[str, str]:
    """Resolve PatientID + StudyUID from a studies_with_lesion symlink."""
    study_uid = parse_lesion_list_name(link.name)
    if link.is_symlink():
        target = Path(os.readlink(link))
        parts = target.parts
        try:
            i = next(i for i, x in enumerate(parts) if x.startswith("PCa-"))
            return parts[i], parts[i + 1]
        except (StopIteration, IndexError):
            pass
    return "", study_uid


def iter_lesion_studies(lesion_list: Path, dicom_root: Path):
    """Yield (patient_id, study_uid, study_dir, list_symlink)."""
    for link in sorted(lesion_list.iterdir()):
        if not link.is_symlink() and not link.is_file():
            continue
        patient_id, study_uid = _patient_study_from_lesion_link(link)
        if not patient_id:
            # Last resort: scan dicom for this StudyUID (slow; rare).
            study_dir = None
            for patient in dicom_root.iterdir():
                cand = patient / study_uid
                if cand.is_dir():
                    patient_id = patient.name
                    study_dir = cand
                    break
            if study_dir is None:
                yield "", study_uid, dicom_root / "_missing_" / study_uid, link
                continue
            yield patient_id, study_uid, study_dir, link
            continue
        yield patient_id, study_uid, resolve_study_dir(dicom_root, patient_id, study_uid), link


def is_true_lesion_seg(series: LocalSeries) -> bool:
    """True for SEG series that carry lesion annotations (not misnamed zone SEGs)."""
    if series.modality != "SEG":
        return False
    labels = series.segment_labels
    desc = series.series_description
    blob = " ".join([desc or ""] + list(labels or [])).lower()
    if not any(tok in blob for tok in ("lesion", "neoplasm", "cspca", "cancer", "tumor", "tumour", "heatmap")):
        return False
    if "broad" in blob or "core roi" in blob:
        return False
    # Zone-only SEGs sometimes include "_Lesions" in the description.
    has_lesion_label = any("lesion" in (lab or "").lower() or "neoplasm" in (lab or "").lower() for lab in labels)
    if series.has_zone_labels and not has_lesion_label:
        return False
    return True


def lesion_seg_rank(series: LocalSeries) -> tuple[int, int, int, int]:
    """Higher is better: Final > lesion-labelled > non-autoseg > more instances."""
    desc = series.series_description
    labels = series.segment_labels
    has_lesion_label = any("lesion" in (lab or "").lower() for lab in labels)
    auto = 0 if _AUTOSEG_RE.search(desc) else 1
    final = 1 if _FINAL_RE.search(desc) else 0
    return (final, int(has_lesion_label), auto, series.n_instances)


def pick_lesion_seg(series_list: list[LocalSeries]) -> LocalSeries | None:
    cands = [s for s in series_list if is_true_lesion_seg(s)]
    if not cands:
        return None
    return max(cands, key=lesion_seg_rank)


def pick_mr_tag(
    series_list: list[LocalSeries],
    tag: str,
    *,
    preferred_uids: list[str] | None = None,
    axial_preferred: bool = False,
) -> LocalSeries | None:
    preferred = set(preferred_uids or [])
    pool = [s for s in series_list if s.modality == "MR" and s.mr_tag == tag]
    if not pool:
        return None
    if preferred:
        hit = [s for s in pool if s.series_instance_uid in preferred]
        if hit:
            pool = hit
    if axial_preferred:
        axial = [s for s in pool if s.axial]
        if axial:
            pool = axial
    return max(pool, key=lambda s: (s.axial if axial_preferred else False, s.n_instances))
