#!/usr/bin/env python3
"""Download T2 + zone SEG pairs as class-index masks (gland_non_pz=1, PZ=2)."""

from __future__ import annotations

import argparse
import io
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
from PIL import Image
from tqdm import tqdm

from src.geometry import describe_alignment, geometry_from_tags, resample_mask_to_geometry
from src.mr_preprocess import dicom_bytes_to_percentile_png
from src.orthanc_client import OrthancClient
from src.seg_masks import extract_zone_frames, referenced_series_uids, resize_mask, save_mask_png
from src.selection import select_t2_zone_pairs
from src.zone_labels import CLASS_GLAND, CLASS_NAMES, CLASS_PZ


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "zones_raw")
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


def merge_index_masks(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Merge class-index masks; PZ wins over gland on conflicts."""
    out = a.copy()
    gland = b == CLASS_GLAND
    pz = b == CLASS_PZ
    out[gland & (out == 0)] = CLASS_GLAND
    out[pz] = CLASS_PZ
    return out


def main() -> int:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    skip_empty = False if args.keep_empty else bool(args.skip_empty)
    image_size = args.image_size or int(os.getenv("IMAGE_SIZE", "192"))

    pairs = select_t2_zone_pairs(args.manifest, args.annotations)
    if args.limit_patients is not None:
        pairs = pairs[: args.limit_patients]
    if not pairs:
        print("No t2 + zone-labeled prostate SEG pairs found")
        return 1

    client = OrthancClient()
    print(f"Connecting to {client.base_url} ...")
    print(f"Downloading {len(pairs)} zone patient(s); classes={CLASS_NAMES}; size={image_size}")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.jsonl"

    total = 0
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

            seg_series = client.get_series(pair.seg_orthanc_series_id)
            instance_ids = seg_series.get("Instances") or []
            if not instance_ids:
                errors.append(f"{pair.patient_id}: SEG has no instances")
                continue

            plane_masks: dict[str, dict] = {}
            ref_series_uids: list[str] = []
            for instance_id in instance_ids:
                try:
                    seg_bytes = client.download_instance_file(instance_id)
                    frames, seg_ds = extract_zone_frames(seg_bytes)
                    for frame in frames:
                        prev = plane_masks.get(frame.merge_key)
                        if prev is None:
                            plane_masks[frame.merge_key] = {
                                "mask": frame.mask,
                                "geometry": frame.geometry,
                                "sop": frame.referenced_sop_instance_uid,
                            }
                        else:
                            prev["mask"] = merge_index_masks(prev["mask"], frame.mask)
                            if prev["sop"] is None and frame.referenced_sop_instance_uid:
                                prev["sop"] = frame.referenced_sop_instance_uid
                    if seg_ds is not None:
                        ref_series_uids = referenced_series_uids(seg_ds) or ref_series_uids
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{pair.patient_id} SEG {instance_id}: {exc}")

            if not plane_masks:
                errors.append(f"{pair.patient_id}: no zone frames extracted")
                continue

            mr_uid = pair.t2_series_instance_uid
            mr_instances = load_mr_instances(client, mr_uid)
            if not mr_instances and ref_series_uids:
                mr_uid = ref_series_uids[0]
                mr_instances = load_mr_instances(client, mr_uid)
            if not mr_instances:
                errors.append(f"{pair.patient_id}: T2 not found")
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
                # Resample each class binary separately for correct nearest labels.
                index = plane["mask"]
                aligned = np.zeros(
                    (target["geometry"].rows, target["geometry"].cols), dtype=np.uint8
                )
                for cls in (CLASS_GLAND, CLASS_PZ):
                    binary = (index == cls).astype(np.uint8)
                    if binary.max() == 0:
                        continue
                    warped = resample_mask_to_geometry(
                        binary, plane["geometry"], target["geometry"]
                    )
                    if cls == CLASS_GLAND:
                        aligned[(warped > 0) & (aligned == 0)] = CLASS_GLAND
                    else:
                        aligned[warped > 0] = CLASS_PZ
                alignment = describe_alignment(plane["geometry"], target["geometry"])
                alignment["match_mode"] = match_mode
                prev = mr_masks.get(target["sop"])
                if prev is None:
                    mr_masks[target["sop"]] = {
                        "mask": aligned,
                        "orthanc_id": target["orthanc_id"],
                        "alignment": alignment,
                        "instance_number": target["instance_number"],
                    }
                else:
                    prev["mask"] = merge_index_masks(prev["mask"], aligned)

            if unmatched:
                errors.append(
                    f"{pair.patient_id}: {unmatched}/{len(plane_masks)} planes unmatched"
                )
            if not mr_masks:
                errors.append(f"{pair.patient_id}: no MR masks")
                continue

            for sop_uid, info in tqdm(mr_masks.items(), desc=f"  {safe_pid[:24]}", leave=False):
                try:
                    dicom_bytes = client.download_instance_file(info["orthanc_id"])
                    png_bytes, intensity_meta = dicom_bytes_to_percentile_png(
                        dicom_bytes, size=image_size
                    )
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{pair.patient_id}: MR failed {sop_uid}: {exc}")
                    continue

                mask_resized = resize_mask(info["mask"], image_size)
                n_gland = int((mask_resized == CLASS_GLAND).sum())
                n_pz = int((mask_resized == CLASS_PZ).sum())
                if skip_empty and (n_gland + n_pz) == 0:
                    skipped_empty += 1
                    continue

                stem = sop_uid.replace(".", "_")
                image_path = img_dir / f"{stem}.png"
                mask_path = mask_dir / f"{stem}.png"
                Image.open(io.BytesIO(png_bytes)).save(image_path)
                save_mask_png(mask_resized, str(mask_path))

                record = {
                    "patient_id": pair.patient_id,
                    "task": "zones",
                    "num_classes": 3,
                    "class_names": list(CLASS_NAMES),
                    "t2_series_instance_uid": pair.t2_series_instance_uid,
                    "seg_series_instance_uid": pair.seg_series_instance_uid,
                    "segment_labels": pair.segment_labels,
                    "sop_instance_uid": sop_uid,
                    "instance_number": info.get("instance_number"),
                    "alignment": info["alignment"],
                    "image_size": image_size,
                    "image_path": str(image_path.relative_to(args.output)),
                    "mask_path": str(mask_path.relative_to(args.output)),
                    "mask_gland_pixels": n_gland,
                    "mask_pz_pixels": n_pz,
                    "mask_foreground_pixels": n_gland + n_pz,
                    "intensity_meta": intensity_meta,
                }
                manifest.write(json.dumps(record) + "\n")
                total += 1

    (args.output / "selected_pairs.json").write_text(
        json.dumps(
            [
                {
                    "patient_id": p.patient_id,
                    "segment_labels": p.segment_labels,
                    "t2": p.t2_series_instance_uid,
                    "seg": p.seg_series_instance_uid,
                }
                for p in pairs
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {total} slices to {args.output}")
    if skipped_empty:
        print(f"Skipped empty: {skipped_empty}")
    if errors:
        err = args.output / "download_errors.log"
        err.write_text("\n".join(errors) + "\n", encoding="utf-8")
        print(f"{len(errors)} warnings -> {err}")
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
