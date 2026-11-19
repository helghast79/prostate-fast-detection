#!/usr/bin/env python3
"""Build lesion_v3 training volumes from the DICOM lesion catalog.

Reads data/procancer_lesion_dicom (symlinks + annotations). Skips discarded /
incomplete cases. For each kept study writes under data/lesion_v3 (never into
ProCAncer-I):

  cases/<patient_id>.npz   — t2/adc/dwi pixels+geometry, lesion mask, prostate mask
  cases/<patient_id>.json
  index.json
  splits/{train,validation,test}.txt

ADC/DWI multi-b volumes keep the darkest image per slice position (highest b).
Prostate mask comes from the study's zone SEG (TZ+CZ/PZ → binary gland) when
present; training uses its centroid to place the CROP_MM box (localizer-v3
fallback is applied at train time if missing).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.bootstrap import sanitize_sys_path

sanitize_sys_path()

import numpy as np
from tqdm import tqdm

from src.cascade.io_dicom import load_dicom_folder
from src.cascade.lesion import Series
from src.cascade.orthanc_volumes import keep_lowest_signal_per_position
from src.procancer_dicom import LocalSeries, pick_zone_seg
from src.procancer_lesion import lesion_mask_from_seg_dir
from src.procancer_zones import zone_mask_from_seg_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, default=ROOT / "data" / "procancer_lesion_dicom")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "lesion_v3")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def load_annotations(catalog: Path) -> dict:
    path = catalog / "annotations.json"
    if not path.is_file():
        return {"cases": {}}
    return json.loads(path.read_text())


def to_series(slices) -> Series:
    g = slices[0].geometry
    return Series(
        pixels=np.stack([s.pixels for s in slices]).astype(np.float32),
        ipp=np.stack([s.geometry.ipp for s in slices]).astype(np.float64),
        iop=np.asarray(g.iop, dtype=np.float64),
        spacing=np.asarray(g.spacing, dtype=np.float64),
    )


def series_from_meta(entry: dict) -> LocalSeries:
    return LocalSeries(
        series_dir=entry["series_dir"],
        series_instance_uid=entry["series_instance_uid"],
        modality=entry["modality"],
        series_description=entry.get("series_description") or "",
        n_instances=int(entry.get("n_instances") or 0),
        mr_tag=entry.get("mr_tag"),
        axial=bool(entry.get("axial")),
        segment_labels=list(entry.get("segment_labels") or []),
        referenced_series_uids=list(entry.get("referenced_series_uids") or []),
        has_zone_labels=bool(entry.get("has_zone_labels")),
    )


def label_source(desc: str) -> str:
    return "manual" if "_final" in (desc or "").lower() else "automatic"


def slice_mm(ipp: np.ndarray) -> float:
    if len(ipp) < 2:
        return 0.0
    return float(np.linalg.norm(ipp[-1] - ipp[0]) / (len(ipp) - 1))


def split_patients(metas: list[dict], seed: int, val_frac: float, test_frac: float) -> dict[str, list[str]]:
    rng = random.Random(seed)
    by_src: dict[str, list[str]] = {"manual": [], "automatic": []}
    for m in metas:
        by_src.setdefault(m["label_source"], []).append(m["patient_id"])
    train, val, test = [], [], []
    for pids in by_src.values():
        rng.shuffle(pids)
        n = len(pids)
        n_test = max(1, int(round(n * test_frac))) if n >= 10 else max(0, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac))) if n >= 10 else max(0, int(round(n * val_frac)))
        test.extend(pids[:n_test])
        val.extend(pids[n_test : n_test + n_val])
        train.extend(pids[n_test + n_val :])
    return {"train": sorted(train), "validation": sorted(val), "test": sorted(test)}


def build_one(case_dir: Path, meta: dict) -> tuple[dict, dict]:
    t2_sl = load_dicom_folder(case_dir / "t2")
    adc_sl = keep_lowest_signal_per_position(load_dicom_folder(case_dir / "adc"))
    dwi_sl = keep_lowest_signal_per_position(load_dicom_folder(case_dir / "dwi"))
    if not t2_sl or not adc_sl or not dwi_sl:
        raise ValueError("missing T2/ADC/DWI slices")

    lesion, lesion_info = lesion_mask_from_seg_dir(case_dir / "lesion_seg", t2_sl)

    prostate = np.zeros((len(t2_sl), *t2_sl[0].pixels.shape), dtype=np.uint8)
    zone_info: dict = {"matched_planes": 0, "unmatched_planes": 0}
    zone_desc = None
    zone = pick_zone_seg([series_from_meta(s) for s in meta.get("series") or []])
    if zone is not None:
        zone_desc = zone.series_description
        zone_mask, zone_info = zone_mask_from_seg_dir(Path(zone.series_dir), t2_sl)
        prostate = (zone_mask > 0).astype(np.uint8)

    t2, adc, dwi = to_series(t2_sl), to_series(adc_sl), to_series(dwi_sl)
    dz = slice_mm(t2.ipp)
    lesion_vox = int(lesion.sum())
    out_meta = {
        "patient_id": meta["patient_id"],
        "study_instance_uid": meta["study_instance_uid"],
        "label_source": label_source((meta.get("lesion_seg") or {}).get("series_description") or ""),
        "t2": (meta.get("t2") or {}).get("series_description"),
        "adc": (meta.get("adc") or {}).get("series_description"),
        "dwi": (meta.get("dwi") or {}).get("series_description"),
        "lesion_seg": (meta.get("lesion_seg") or {}).get("series_description"),
        "zone_seg": zone_desc,
        "shape_t2": list(t2.pixels.shape),
        "shape_adc": list(adc.pixels.shape),
        "shape_dwi": list(dwi.pixels.shape),
        "spacing_yx_mm": [float(t2.spacing[0]), float(t2.spacing[1])],
        "slice_mm": dz,
        "lesion_voxels": lesion_vox,
        "lesion_ml": round(float(lesion_vox) * float(t2.spacing[0]) * float(t2.spacing[1]) * dz / 1000.0, 3),
        "lesion_slices": int((lesion.reshape(len(lesion), -1).sum(1) > 0).sum()),
        "prostate_voxels": int(prostate.sum()),
        "has_prostate_mask": bool(prostate.any()),
        "lesion_matched_planes": lesion_info["matched_planes"],
        "lesion_unmatched_planes": lesion_info["unmatched_planes"],
        "zone_matched_planes": zone_info.get("matched_planes"),
        "zone_unmatched_planes": zone_info.get("unmatched_planes"),
    }
    arrays = {
        "mask": lesion.astype(np.uint8),
        "prostate": prostate.astype(np.uint8),
        **t2.to_npz("t2"),
        **adc.to_npz("adc"),
        **dwi.to_npz("dwi"),
    }
    return out_meta, arrays


def main() -> int:
    args = parse_args()
    summary = json.loads((args.catalog / "index.json").read_text())
    ann = load_annotations(args.catalog)["cases"]
    cases_root = args.out / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)

    selected = []
    for c in summary.get("cases") or []:
        if not c.get("ready"):
            continue
        key = f"{c['patient_id']}/{c['study_instance_uid']}"
        if (ann.get(key) or {}).get("discarded"):
            continue
        selected.append(c)
    if args.limit is not None:
        selected = selected[: args.limit]

    index: list[dict] = []
    for c in tqdm(selected, desc="Lesion v3"):
        pid = c["patient_id"]
        npz_path = cases_root / f"{pid}.npz"
        json_path = cases_root / f"{pid}.json"
        if npz_path.is_file() and json_path.is_file() and not args.force:
            index.append(json.loads(json_path.read_text()))
            continue
        case_dir = args.catalog / "cases" / pid / c["study_instance_uid"]
        meta_full = json.loads((case_dir / "meta.json").read_text())
        try:
            out_meta, arrays = build_one(case_dir, meta_full)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {pid[:24]}: {type(exc).__name__}: {exc}")
            continue
        if out_meta["lesion_voxels"] <= 0:
            print(f"SKIP {pid[:24]}: empty lesion mask")
            continue
        np.savez_compressed(npz_path, **arrays)
        json_path.write_text(json.dumps(out_meta, indent=2) + "\n")
        index.append(out_meta)

    index.sort(key=lambda m: m["patient_id"])
    splits = split_patients(index, args.seed, args.val_frac, args.test_frac)
    (args.out / "splits").mkdir(parents=True, exist_ok=True)
    for name, pids in splits.items():
        (args.out / "splits" / f"{name}.txt").write_text("\n".join(pids) + ("\n" if pids else ""))
    (args.out / "index.json").write_text(
        json.dumps(
            {
                "catalog": str(args.catalog),
                "n_cases": len(index),
                "n_with_prostate": sum(1 for m in index if m.get("has_prostate_mask")),
                "splits": {k: len(v) for k, v in splits.items()},
                "cases": index,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"wrote {args.out}: {len(index)} cases, "
        f"prostate {sum(1 for m in index if m.get('has_prostate_mask'))}, "
        f"splits { {k: len(v) for k, v in splits.items()} }"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
