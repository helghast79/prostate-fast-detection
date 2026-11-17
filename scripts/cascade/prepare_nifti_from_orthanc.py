#!/usr/bin/env python3
"""Build cascade NIfTI dataset from Orthanc T2 + prostate-regions SEG pairs.

Output layout (Stage 1):
  data/cascade/
    images/<patient_id>.nii.gz
    masks/<patient_id>.nii.gz
    meta/<patient_id>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.bootstrap import sanitize_sys_path

sanitize_sys_path()

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from src.cascade.io_dicom import DicomSlice, sort_slices_by_ipp, stack_volume
from src.cascade.io_nifti import dataset_paths, save_nifti, save_patient_meta
from src.cascade.normalize import percentile_normalize
from src.geometry import resample_mask_to_geometry
from src.orthanc_client import OrthancClient
from src.seg_masks import extract_prostate_frames, referenced_series_uids
from src.selection import select_t2_prostate_pairs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "data" / "cascade")
    p.add_argument("--manifest", type=Path, default=ROOT / "data" / "series_manifest.json")
    p.add_argument(
        "--annotations",
        type=Path,
        default=ROOT / "data" / "series_annotations.json",
    )
    p.add_argument("--match-tol-mm", type=float, default=1.5)
    p.add_argument("--limit-patients", type=int, default=None)
    p.add_argument("--patient-id", type=str, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--raw-intensity",
        action="store_true",
        help="Store rescaled DICOM values instead of volume-percentile [0,1]",
    )
    p.add_argument(
        "--exclude-seminal-vesicles",
        action="store_true",
        help="Drop SV segments so masks contain prostate only",
    )
    return p.parse_args()


def load_mr_slices(client: OrthancClient, series_uid: str) -> list[DicomSlice]:
    series_id = client.find_series_by_uid(series_uid)
    if not series_id:
        return []
    series = client.get_series(series_id)
    out: list[DicomSlice] = []
    for iid in series.get("Instances") or []:
        raw = client.download_instance_file(iid)
        # Local adapter without going through folder I/O
        from src.cascade.io_dicom import read_dicom_bytes

        out.append(read_dicom_bytes(raw))
    return sort_slices_by_ipp(out)


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


def main() -> int:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    pairs = select_t2_prostate_pairs(args.manifest, args.annotations)
    if args.patient_id:
        pairs = [p for p in pairs if p.patient_id == args.patient_id]
    if args.limit_patients is not None:
        pairs = pairs[: args.limit_patients]
    if not pairs:
        print("No t2 + prostate-regions SEG pairs found")
        return 1

    client = OrthancClient()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "images").mkdir(exist_ok=True)
    (args.output / "masks").mkdir(exist_ok=True)
    (args.output / "meta").mkdir(exist_ok=True)

    index_path = args.output / "index.jsonl"
    errors: list[str] = []
    written = 0

    with index_path.open("w", encoding="utf-8") as index_f:
        for pair in tqdm(pairs, desc="Patients"):
            safe_pid = pair.patient_id.replace("/", "_")
            paths = dataset_paths(args.output, safe_pid)
            if paths["image"].exists() and paths["mask"].exists() and not args.force:
                index_f.write(
                    json.dumps({"patient_id": safe_pid, "status": "skipped_exists"}) + "\n"
                )
                continue

            if not pair.seg_orthanc_series_id:
                errors.append(f"{pair.patient_id}: missing SEG orthanc id")
                continue

            seg_series = client.get_series(pair.seg_orthanc_series_id)
            plane_masks: dict[str, dict] = {}
            ref_series_uids: list[str] = []
            used_segments: list[int] = []

            for instance_id in seg_series.get("Instances") or []:
                try:
                    seg_bytes = client.download_instance_file(instance_id)
                    frames, seg_ds = extract_prostate_frames(
                        seg_bytes,
                        exclude_seminal_vesicles=args.exclude_seminal_vesicles,
                    )
                    if frames:
                        used_segments = frames[0].segment_numbers
                    for frame in frames:
                        prev = plane_masks.get(frame.merge_key)
                        if prev is None:
                            plane_masks[frame.merge_key] = {
                                "mask": frame.mask,
                                "geometry": frame.geometry,
                                "sop": frame.referenced_sop_instance_uid,
                            }
                        else:
                            prev["mask"] = (
                                (prev["mask"].astype(bool) | frame.mask.astype(bool)).astype(
                                    np.uint8
                                )
                            )
                    if seg_ds is not None:
                        ref_series_uids = referenced_series_uids(seg_ds) or ref_series_uids
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{pair.patient_id} SEG {instance_id}: {exc}")

            if not plane_masks:
                errors.append(f"{pair.patient_id}: no prostate SEG frames")
                continue

            mr_uid = pair.t2_series_instance_uid
            mr_slices = load_mr_slices(client, mr_uid)
            if not mr_slices and ref_series_uids:
                mr_uid = ref_series_uids[0]
                mr_slices = load_mr_slices(client, mr_uid)
            if not mr_slices:
                errors.append(f"{pair.patient_id}: T2 not found")
                continue

            vol, affine, dicom_meta = stack_volume(mr_slices)
            mask_vol = np.zeros(vol.shape, dtype=np.uint8)
            by_sop = {s.sop_instance_uid: s for s in mr_slices}
            sop_to_z = {s.sop_instance_uid: i for i, s in enumerate(mr_slices)}
            unmatched = 0
            matched = 0

            for plane in plane_masks.values():
                target, match_mode = find_mr_target(
                    plane["sop"],
                    plane["geometry"].ipp,
                    by_sop,
                    mr_slices,
                    args.match_tol_mm,
                )
                if target is None:
                    unmatched += 1
                    continue
                aligned = resample_mask_to_geometry(
                    plane["mask"], plane["geometry"], target.geometry
                )
                z = sop_to_z[target.sop_instance_uid]
                mask_vol[z] = np.maximum(mask_vol[z], aligned.astype(np.uint8))
                matched += 1

            if matched == 0:
                errors.append(f"{pair.patient_id}: no masks aligned to T2")
                continue

            if args.raw_intensity:
                vol_n = vol.astype(np.float32)
                norm_meta = {"method": "raw_rescaled"}
            else:
                vol_n, norm_meta = percentile_normalize(vol)
            save_nifti(paths["image"], vol_n, affine, dtype=np.float32)
            save_nifti(paths["mask"], mask_vol, affine, dtype=np.uint8)

            meta = {
                "patient_id": safe_pid,
                "modality": "t2",
                "t2_series_instance_uid": pair.t2_series_instance_uid,
                "t2_series_description": pair.t2_description,
                "seg_series_instance_uid": pair.seg_series_instance_uid,
                "seg_series_description": pair.seg_description,
                "segment_labels": pair.segment_labels,
                "segment_numbers": used_segments,
                "referenced_series_uids": ref_series_uids or [mr_uid],
                "shape_zyx": list(vol_n.shape),
                "affine": affine.tolist(),
                "normalization": norm_meta,
                "dicom": dicom_meta,
                "n_mask_slices": int((mask_vol.reshape(mask_vol.shape[0], -1).sum(1) > 0).sum()),
                "mask_voxels": int(mask_vol.sum()),
                "unmatched_seg_planes": unmatched,
                "matched_seg_planes": matched,
            }
            save_patient_meta(paths["meta"], meta)
            index_f.write(
                json.dumps(
                    {
                        "patient_id": safe_pid,
                        "status": "ok",
                        "shape_zyx": meta["shape_zyx"],
                        "n_mask_slices": meta["n_mask_slices"],
                        "mask_voxels": meta["mask_voxels"],
                    }
                )
                + "\n"
            )
            written += 1

    print(f"Wrote {written} patient volume(s) → {args.output}")
    if errors:
        err_path = args.output / "prepare_errors.txt"
        err_path.write_text("\n".join(errors) + "\n", encoding="utf-8")
        print(f"{len(errors)} warning(s) → {err_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
