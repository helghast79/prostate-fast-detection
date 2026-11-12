#!/usr/bin/env python3
"""Build a PatientID / MR+SEG series catalog from Orthanc (PCa-* patients).

Writes:
  - data/series_manifest.txt  (human-editable catalog)
  - data/series_manifest.json (UI catalog with Orthanc IDs + SEG references)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bootstrap import sanitize_sys_path

sanitize_sys_path()

from dotenv import load_dotenv
from tqdm import tqdm

from src.orthanc_client import OrthancClient
from src.series_manifest import PatientEntry, SeriesEntry, write_json_manifest, write_text_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-txt",
        type=Path,
        default=ROOT / "data" / "series_manifest.txt",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "data" / "series_manifest.json",
    )
    parser.add_argument(
        "--patient-prefix",
        default="PCa-",
        help="PatientID prefix filter (case-sensitive match on stored IDs)",
    )
    parser.add_argument(
        "--modalities",
        default="MR,SEG",
        help="Comma-separated modalities to keep",
    )
    parser.add_argument("--limit-patients", type=int, default=None)
    parser.add_argument(
        "--with-seg-only",
        action="store_true",
        help="Keep only patients that have at least one SEG series",
    )
    parser.add_argument(
        "--skip-seg-refs",
        action="store_true",
        help="Do not resolve ReferencedSeriesSequence for SEG instances",
    )
    return parser.parse_args()


def _main_tags(resource: dict[str, Any]) -> dict[str, Any]:
    return resource.get("MainDicomTags") or {}


def discover_patients(client: OrthancClient, prefix: str) -> dict[str, list[str]]:
    """Return PatientID -> Orthanc patient IDs for IDs starting with prefix."""
    query_prefix = prefix if prefix.endswith("*") else f"{prefix}*"
    try:
        matches = client.find("Patient", {"PatientID": query_prefix}, expand=True)
    except Exception:
        matches = []

    resources: list[Any]
    if matches and isinstance(matches[0], dict):
        resources = matches
    elif matches and isinstance(matches[0], str):
        resources = [client.get_patient(pid) for pid in matches]
    else:
        resources = [client.get_patient(pid) for pid in client.get_json("/patients")]

    grouped: dict[str, list[str]] = {}
    prefix_l = prefix.rstrip("*")
    for resource in resources:
        if isinstance(resource, str):
            resource = client.get_patient(resource)
        tags = _main_tags(resource)
        patient_id = str(tags.get("PatientID", ""))
        if not patient_id.startswith(prefix_l):
            continue
        orthanc_id = resource.get("ID")
        if not orthanc_id:
            continue
        grouped.setdefault(patient_id, [])
        if orthanc_id not in grouped[patient_id]:
            grouped[patient_id].append(orthanc_id)
    return grouped


def discover_patients_with_seg(
    client: OrthancClient,
    prefix: str,
    all_patients: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Subset of patients that have at least one SEG series (via Orthanc find)."""
    query_prefix = prefix if prefix.endswith("*") else f"{prefix}*"
    prefix_l = prefix.rstrip("*")
    try:
        seg_ids = client.find("Series", {"PatientID": query_prefix, "Modality": "SEG"})
    except Exception:
        seg_ids = []

    wanted: set[str] = set()
    for series_id in tqdm(seg_ids, desc="SEG patients"):
        try:
            series = client.get_series(series_id)
            study = client.get_study(series["ParentStudy"])
            patient_id = str((study.get("PatientMainDicomTags") or {}).get("PatientID", ""))
        except Exception:
            continue
        if patient_id.startswith(prefix_l):
            wanted.add(patient_id)

    return {pid: all_patients[pid] for pid in sorted(wanted) if pid in all_patients}


def _preview_instance_id(series: dict[str, Any]) -> str:
    instances = series.get("Instances") or []
    if not instances:
        return ""
    return str(instances[len(instances) // 2])


def _referenced_series_and_labels(
    client: OrthancClient,
    series: dict[str, Any],
) -> tuple[list[str], list[str]]:
    instances = series.get("Instances") or []
    if not instances:
        return [], []
    tags = client.get_instance_tags(instances[0], simplify=True)
    ref_uids: list[str] = []
    rss = tags.get("ReferencedSeriesSequence")
    items = rss if isinstance(rss, list) else ([rss] if isinstance(rss, dict) else [])
    for item in items:
        if not isinstance(item, dict):
            continue
        uid = str(item.get("SeriesInstanceUID") or "").strip()
        if uid and uid not in ref_uids:
            ref_uids.append(uid)

    labels: list[str] = []
    seg_seq = tags.get("SegmentSequence")
    seg_items = (
        seg_seq if isinstance(seg_seq, list) else ([seg_seq] if isinstance(seg_seq, dict) else [])
    )
    for item in seg_items:
        if not isinstance(item, dict):
            continue
        label = str(item.get("SegmentLabel") or item.get("SegmentDescription") or "").strip()
        if label and label not in labels:
            labels.append(label)
    return ref_uids, labels


def collect_patient_series(
    client: OrthancClient,
    patient_id: str,
    orthanc_patient_ids: list[str],
    modalities: set[str],
    *,
    resolve_seg_refs: bool,
) -> PatientEntry:
    entry = PatientEntry(patient_id=patient_id, orthanc_patient_ids=list(orthanc_patient_ids))
    seen_uids: set[str] = set()

    for orthanc_patient_id in orthanc_patient_ids:
        patient = client.get_patient(orthanc_patient_id)
        for study_id in patient.get("Studies") or []:
            study = client.get_study(study_id)
            for series_id in study.get("Series") or []:
                series = client.get_series(series_id)
                tags = _main_tags(series)
                modality = str(tags.get("Modality", "")).upper()
                if modality not in modalities:
                    continue
                uid = str(tags.get("SeriesInstanceUID", "")).strip()
                if not uid or uid in seen_uids:
                    continue
                seen_uids.add(uid)

                referenced: list[str] = []
                labels: list[str] = []
                if modality == "SEG" and resolve_seg_refs:
                    try:
                        referenced, labels = _referenced_series_and_labels(client, series)
                    except Exception:
                        referenced, labels = [], []

                entry.series.append(
                    SeriesEntry(
                        modality=modality,
                        series_description=str(tags.get("SeriesDescription", "")).strip(),
                        series_instance_uid=uid,
                        orthanc_series_id=str(series_id),
                        orthanc_study_id=str(study_id),
                        preview_instance_id=_preview_instance_id(series),
                        series_number=str(tags.get("SeriesNumber", "")).strip(),
                        referenced_series_uids=referenced,
                        segment_labels=labels,
                    )
                )

    entry.series.sort(
        key=lambda s: (
            0 if s.modality == "MR" else 1,
            s.series_description.lower(),
            s.series_instance_uid,
        )
    )
    return entry


def main() -> int:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    modalities = {m.strip().upper() for m in args.modalities.split(",") if m.strip()}
    if not modalities:
        print("No modalities selected", file=sys.stderr)
        return 1

    client = OrthancClient()
    print(f"Connecting to {client.base_url} ...")
    grouped = discover_patients(client, args.patient_prefix)
    print(f"Found {len(grouped)} PatientID(s) with prefix {args.patient_prefix!r}")
    if args.with_seg_only:
        grouped = discover_patients_with_seg(client, args.patient_prefix, grouped)
        print(f"Patients with SEG: {len(grouped)}")
    patient_ids = sorted(grouped)
    if args.limit_patients is not None:
        patient_ids = patient_ids[: args.limit_patients]
    print(f"Scanning {len(patient_ids)} patient(s); modalities={sorted(modalities)}")

    patients: list[PatientEntry] = []
    for patient_id in tqdm(patient_ids, desc="Patients"):
        entry = collect_patient_series(
            client,
            patient_id,
            grouped[patient_id],
            modalities,
            resolve_seg_refs=not args.skip_seg_refs,
        )
        if not entry.series:
            continue
        patients.append(entry)

    write_text_manifest(args.output_txt, patients)
    write_json_manifest(args.output_json, patients)

    n_series = sum(len(p.series) for p in patients)
    n_seg = sum(1 for p in patients for s in p.series if s.modality == "SEG")
    n_mr = n_series - n_seg
    print(f"Wrote {len(patients)} patients / {n_series} series ({n_mr} MR, {n_seg} SEG)")
    print(f"Text: {args.output_txt}")
    print(f"JSON: {args.output_json}")
    return 0 if patients else 1


if __name__ == "__main__":
    raise SystemExit(main())
