#!/usr/bin/env python3
"""Download T2 MR + prostate-regions SEG pairs from Orthanc into PNG pairs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.bootstrap import sanitize_sys_path

sanitize_sys_path()

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from src.geometry import (
    describe_alignment,
    geometry_from_tags,
    resample_mask_to_geometry,
)
from src.mr_preprocess import dicom_bytes_to_percentile_png
from src.orthanc_client import OrthancClient
from src.seg_masks import (
    extract_prostate_frames,
    referenced_series_uids,
    resize_mask,
    save_mask_png,
)
from src.selection import select_t2_prostate_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "series_manifest.json")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ROOT / "data" / "series_annotations.json",
    )
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--skip-empty", action="store_true", default=True)
    parser.add_argument("--keep-empty", action="store_true")
    parser.add_argument("--limit-patients", type=int, default=None)
    parser.add_argument("--patient-id", type=str, default=None)
    parser.add_argument("--match-tol-mm", type=float, default=1.5)
    return parser.parse_args()


def load_mr_instances(client: OrthancClient, series_uid: str) -> list[dict]:
    series_id = client.find_series_by_uid(series_uid)
    if not series_id:
        return []
    series = client.get_series(series_id)
    instances: list[dict] = []
    for iid in series.get("Instances") or []:
        tags = client.get_json(f"/instances/{iid}/simplified-tags")
        geom = geometry_from_tags(tags)
        try:
            instance_number = int(float(tags.get("InstanceNumber", 0)))
        except (TypeError, ValueError):
            instance_number = 0
        instances.append(
            {
                "orthanc_id": iid,
                "sop": str(tags["SOPInstanceUID"]),
                "ipp": geom.ipp,
                "geometry": geom,
                "instance_number": instance_number,
            }
        )
    return instances


def find_mr_target(
    frame_sop: str | None,
    seg_ipp: np.ndarray,
    by_sop: dict[str, dict],
    mr_instances: list[dict],
    match_tol_mm: float,
) -> tuple[dict | None, str]:
    if frame_sop and frame_sop in by_sop:
        return by_sop[frame_sop], "sop"
    if not mr_instances:
        return None, "none"
    normal = mr_instances[0]["geometry"].normal
    seg_z = float(np.asarray(seg_ipp) @ normal)
    zs = np.array([float(inst["ipp"] @ normal) for inst in mr_instances], dtype=np.float64)
    best_i = int(np.argmin(np.abs(zs - seg_z)))
    best_dz = float(abs(zs[best_i] - seg_z))
    if best_dz > match_tol_mm:
        return None, f"geom_miss:{best_dz:.3f}"
    return mr_instances[best_i], f"geom_z:{best_dz:.3f}"


def main() -> int:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    skip_empty = False if args.keep_empty else bool(args.skip_empty)
    image_size = args.image_size or int(os.getenv("IMAGE_SIZE", "192"))

    pairs = select_t2_prostate_pairs(args.manifest, args.annotations)
    if args.patient_id:
        pairs = [p for p in pairs if p.patient_id == args.patient_id]
    if args.limit_patients is not None:
        pairs = pairs[: args.limit_patients]
    if not pairs:
        print("No t2 + prostate-regions SEG pairs found in annotations")
        return 1

    client = OrthancClient()
    print(f"Connecting to {client.base_url} ...")
    print(f"Downloading {len(pairs)} patient pair(s); image_size={image_size}")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.jsonl"

    total_pairs = 0
    skipped_empty = 0
    errors: list[str] = []

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for pair in tqdm(pairs, desc="Patients"):
            safe_pid = pair.patient_id.replace("/", "_")
            out_dir = args.output / safe_pid
            img_dir = out_dir / "images"
            mask_dir = out_dir / "masks"
            img_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)

            if not pair.seg_orthanc_series_id:
                errors.append(f"{pair.patient_id}: missing SEG orthanc id")
                continue
            seg_series = client.get_series(pair.seg_orthanc_series_id)
            instance_ids = seg_series.get("Instances") or []
            if not instance_ids:
                errors.append(f"{pair.patient_id}: SEG has no instances")
                continue

            plane_masks: dict[str, dict] = {}
            used_segments: list[int] = []
            ref_series_uids: list[str] = []

            for instance_id in instance_ids:
                try:
                    seg_bytes = client.download_instance_file(instance_id)
                    frames, seg_ds = extract_prostate_frames(seg_bytes)
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
                            if prev["sop"] is None and frame.referenced_sop_instance_uid:
                                prev["sop"] = frame.referenced_sop_instance_uid
                    if seg_ds is not None:
                        ref_series_uids = referenced_series_uids(seg_ds) or ref_series_uids
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{pair.patient_id} SEG instance {instance_id}: {exc}")

            if not plane_masks:
                errors.append(f"{pair.patient_id}: no prostate SEG frames")
                continue

            mr_uid = pair.t2_series_instance_uid
            if ref_series_uids and mr_uid not in ref_series_uids:
                # Still try annotated t2 first; fall back to referenced if needed.
                pass
            mr_instances = load_mr_instances(client, mr_uid)
            if not mr_instances and ref_series_uids:
                mr_uid = ref_series_uids[0]
                mr_instances = load_mr_instances(client, mr_uid)
            if not mr_instances:
                errors.append(f"{pair.patient_id}: T2 series not found: {pair.t2_series_instance_uid}")
                continue

            by_sop = {inst["sop"]: inst for inst in mr_instances}
            mr_masks: dict[str, dict] = {}
            unmatched = 0
            for plane in plane_masks.values():
                target, match_mode = find_mr_target(
                    plane["sop"],
                    plane["geometry"].ipp,
                    by_sop,
                    mr_instances,
                    args.match_tol_mm,
                )
                if target is None:
                    unmatched += 1
                    continue
                aligned = resample_mask_to_geometry(
                    plane["mask"], plane["geometry"], target["geometry"]
                )
                alignment = describe_alignment(plane["geometry"], target["geometry"])
                alignment["match_mode"] = match_mode
                prev = mr_masks.get(target["sop"])
                if prev is None:
                    mr_masks[target["sop"]] = {
                        "mask": aligned,
                        "orthanc_id": target["orthanc_id"],
                        "alignment": alignment,
                        "source_seg_sop": plane["sop"],
                        "instance_number": target["instance_number"],
                    }
                else:
                    prev["mask"] = (
                        (prev["mask"].astype(bool) | aligned.astype(bool)).astype(np.uint8)
                    )

            if unmatched:
                errors.append(
                    f"{pair.patient_id}: {unmatched}/{len(plane_masks)} SEG planes unmatched"
                )
            if not mr_masks:
                errors.append(f"{pair.patient_id}: no MR masks after alignment")
                continue

            for sop_uid, info in tqdm(mr_masks.items(), desc=f"  {safe_pid[:24]}", leave=False):
                try:
                    dicom_bytes = client.download_instance_file(info["orthanc_id"])
                    png_bytes, intensity_meta = dicom_bytes_to_percentile_png(
                        dicom_bytes, size=image_size
                    )
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{pair.patient_id}: MR render failed {sop_uid}: {exc}")
                    continue

                mask_resized = resize_mask(info["mask"], image_size)
                if skip_empty and mask_resized.max() == 0:
                    skipped_empty += 1
                    continue

                stem = sop_uid.replace(".", "_")
                image_path = img_dir / f"{stem}.png"
                mask_path = mask_dir / f"{stem}.png"
                from PIL import Image
                import io

                Image.open(io.BytesIO(png_bytes)).save(image_path)
                save_mask_png(mask_resized, str(mask_path))

                record = {
                    "patient_id": pair.patient_id,
                    "study_split": "raw",
                    "t2_series_instance_uid": pair.t2_series_instance_uid,
                    "t2_series_description": pair.t2_description,
                    "seg_series_instance_uid": pair.seg_series_instance_uid,
                    "seg_series_description": pair.seg_description,
                    "segment_labels": pair.segment_labels,
                    "segment_numbers": used_segments,
                    "referenced_series_uids": ref_series_uids or [mr_uid],
                    "sop_instance_uid": sop_uid,
                    "source_seg_sop_instance_uid": info["source_seg_sop"],
                    "instance_number": info.get("instance_number"),
                    "alignment": info["alignment"],
                    "image_size": image_size,
                    "image_source": "mr_percentile",
                    "intensity_meta": intensity_meta,
                    "image_path": str(image_path.relative_to(args.output)),
                    "mask_path": str(mask_path.relative_to(args.output)),
                    "mask_foreground_pixels": int(mask_resized.sum()),
                }
                manifest.write(json.dumps(record) + "\n")
                total_pairs += 1

    selection_path = args.output / "selected_pairs.json"
    selection_path.write_text(
        json.dumps(
            [
                {
                    "patient_id": p.patient_id,
                    "t2_series_instance_uid": p.t2_series_instance_uid,
                    "seg_series_instance_uid": p.seg_series_instance_uid,
                    "segment_labels": p.segment_labels,
                }
                for p in pairs
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {total_pairs} pairs to {args.output}")
    print(f"Manifest: {manifest_path}")
    print(f"Selection: {selection_path}")
    if skipped_empty:
        print(f"Skipped empty masks: {skipped_empty}")
    if errors:
        err_path = args.output / "download_errors.log"
        err_path.write_text("\n".join(errors) + "\n", encoding="utf-8")
        print(f"{len(errors)} warnings/errors logged to {err_path}")
    return 0 if total_pairs else 1


if __name__ == "__main__":
    raise SystemExit(main())
