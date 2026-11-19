#!/usr/bin/env python3
"""Build cascade v3 dataset from the ProCAncer DICOM zones catalog.

Reads data/procancer_zones_dicom (symlinks + annotations). Skips discarded
cases. Writes under data/cascade_v3 (never into ProCAncer-I):

  cases/<patient_id>.npz  — image float16 (Z,H,W), mask uint8 (0/1/2)
  cases/<patient_id>.json
  index.json
  splits/{train,validation,test}.txt
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

from src.procancer_zones import load_t2_and_zones


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, default=ROOT / "data" / "procancer_zones_dicom")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "cascade_v3")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def kept_cases(catalog: Path) -> list[dict]:
    summary = json.loads((catalog / "index.json").read_text())
    ann = {}
    ap = catalog / "annotations.json"
    if ap.is_file():
        ann = json.loads(ap.read_text()).get("cases") or {}
    out = []
    for c in summary.get("cases") or []:
        if not c.get("ready"):
            continue
        key = f"{c['patient_id']}/{c['study_instance_uid']}"
        if (ann.get(key) or {}).get("discarded"):
            continue
        out.append(c)
    return out


def write_splits(ids: list[str], out: Path, seed: int, val_frac: float, test_frac: float) -> dict:
    rng = random.Random(seed)
    order = ids[:]
    rng.shuffle(order)
    n = len(order)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    test = sorted(order[:n_test])
    val = sorted(order[n_test : n_test + n_val])
    train = sorted(order[n_test + n_val :])
    split_dir = out / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("validation", val), ("test", test)):
        (split_dir / f"{name}.txt").write_text("\n".join(rows) + "\n")
    return {"train": len(train), "validation": len(val), "test": len(test)}


def main() -> int:
    args = parse_args()
    cases = kept_cases(args.catalog)
    if args.limit is not None:
        cases = cases[: args.limit]
    print(f"{len(cases)} kept ready cases from {args.catalog}")
    cases_dir = args.out / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    index: list[dict] = []
    for c in tqdm(cases, desc="Prepare v3"):
        pid = c["patient_id"]
        sid = c["study_instance_uid"]
        npz_path = cases_dir / f"{pid}.npz"
        json_path = cases_dir / f"{pid}.json"
        if npz_path.is_file() and json_path.is_file() and not args.force:
            meta = json.loads(json_path.read_text())
            index.append(meta)
            continue
        t2 = args.catalog / "cases" / pid / sid / "t2"
        seg = args.catalog / "cases" / pid / sid / "zone_seg"
        try:
            slices, mask, info = load_t2_and_zones(t2, seg)
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP {pid}: {type(exc).__name__}: {exc}")
            continue
        image = np.stack([s.pixels for s in slices], axis=0).astype(np.float32)
        sp = slices[0].geometry.spacing
        spacing_yx = (float(sp[0]), float(sp[1]))
        if len(slices) >= 2:
            slice_mm = float(np.linalg.norm(slices[1].geometry.ipp - slices[0].geometry.ipp))
        else:
            slice_mm = float(getattr(slices[0].geometry, "thickness", 3.0) or 3.0)
        np.savez_compressed(
            npz_path,
            image=image.astype(np.float16),
            mask=mask.astype(np.uint8),
        )
        meta = {
            "patient_id": pid,
            "study_instance_uid": sid,
            "t2_description": (c.get("t2") or {}).get("series_description"),
            "seg_description": (c.get("zone_seg") or {}).get("series_description"),
            "shape_zyx": list(mask.shape),
            "spacing_yx_mm": list(spacing_yx),
            "slice_mm": slice_mm,
            "gland_voxels": int((mask == 1).sum()),
            "pz_voxels": int((mask == 2).sum()),
            "prostate_voxels": int((mask > 0).sum()),
            "prostate_slices": int((mask.reshape(len(mask), -1).sum(1) > 0).sum()),
            "matched_planes": info.get("matched_planes"),
            "unmatched_planes": info.get("unmatched_planes"),
        }
        json_path.write_text(json.dumps(meta, indent=2) + "\n")
        index.append(meta)

    index.sort(key=lambda m: m["patient_id"])
    ids = [m["patient_id"] for m in index]
    split_counts = write_splits(ids, args.out, args.seed, args.val_frac, args.test_frac)
    summary = {
        "n_cases": len(index),
        "catalog": str(args.catalog),
        "splits": split_counts,
        "cases": index,
    }
    (args.out / "index.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.out}: {summary['n_cases']} cases, splits={split_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
