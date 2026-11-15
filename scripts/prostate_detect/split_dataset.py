#!/usr/bin/env python3
"""Split downloaded raw pairs into train / val / holdout patient folders."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.bootstrap import sanitize_sys_path

sanitize_sys_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "train")
    parser.add_argument("--val", type=Path, default=ROOT / "data" / "val")
    parser.add_argument("--holdout", type=Path, default=ROOT / "data" / "holdout")
    parser.add_argument("--n-holdout", type=int, default=6)
    parser.add_argument("--n-val", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacing existing train/val/holdout directories",
    )
    return parser.parse_args()


def _load_rows(manifest: Path) -> list[dict]:
    rows: list[dict] = []
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_manifest(path: Path, rows: list[dict], split_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            out = dict(row)
            out["study_split"] = split_name
            # Paths in raw are relative to raw/; rewrite to patient-local paths.
            pid = out["patient_id"].replace("/", "_")
            img = Path(out["image_path"]).name
            mask = Path(out["mask_path"]).name
            out["image_path"] = f"{pid}/images/{img}"
            out["mask_path"] = f"{pid}/masks/{mask}"
            f.write(json.dumps(out) + "\n")


def _prepare_dest(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise SystemExit(f"Refusing to overwrite existing {path} (pass --force)")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    src_manifest = args.src / "manifest.jsonl"
    if not src_manifest.is_file():
        print(f"Missing manifest: {src_manifest}")
        return 1

    rows = _load_rows(src_manifest)
    patients = sorted({r["patient_id"] for r in rows})
    if len(patients) < 3:
        print(f"Need at least 3 patients; found {len(patients)}")
        return 1

    rng = random.Random(args.seed)
    shuffled = patients[:]
    rng.shuffle(shuffled)

    n_hold = min(max(1, args.n_holdout), len(patients) - 2)
    n_val = min(max(1, args.n_val), len(patients) - n_hold - 1)
    holdout_patients = sorted(shuffled[:n_hold])
    val_patients = sorted(shuffled[n_hold : n_hold + n_val])
    train_patients = sorted(shuffled[n_hold + n_val :])
    hold_set, val_set, train_set = set(holdout_patients), set(val_patients), set(train_patients)

    for dest in (args.train, args.val, args.holdout):
        _prepare_dest(dest, args.force)

    def move_patients(pids: list[str], dest: Path) -> None:
        for pid in pids:
            safe = pid.replace("/", "_")
            src = args.src / safe
            if src.is_dir():
                shutil.copytree(src, dest / safe)

    move_patients(train_patients, args.train)
    move_patients(val_patients, args.val)
    move_patients(holdout_patients, args.holdout)

    train_rows = [r for r in rows if r["patient_id"] in train_set]
    val_rows = [r for r in rows if r["patient_id"] in val_set]
    hold_rows = [r for r in rows if r["patient_id"] in hold_set]
    _write_manifest(args.train / "manifest.jsonl", train_rows, "train")
    _write_manifest(args.val / "manifest.jsonl", val_rows, "val")
    _write_manifest(args.holdout / "manifest.jsonl", hold_rows, "holdout")

    split_info = {
        "seed": args.seed,
        "n_holdout": len(holdout_patients),
        "n_val": len(val_patients),
        "n_train": len(train_patients),
        "holdout_patients": holdout_patients,
        "val_patients": val_patients,
        "train_patients": train_patients,
        "n_train_slices": len(train_rows),
        "n_val_slices": len(val_rows),
        "n_holdout_slices": len(hold_rows),
        "src": str(args.src),
    }
    split_path = ROOT / "data" / "split.json"
    split_path.write_text(json.dumps(split_info, indent=2) + "\n", encoding="utf-8")
    for dest in (args.train, args.val, args.holdout):
        (dest / "split.json").write_text(json.dumps(split_info, indent=2) + "\n", encoding="utf-8")
        (dest / "patients.txt").write_text(
            "\n".join(
                holdout_patients
                if dest == args.holdout
                else val_patients
                if dest == args.val
                else train_patients
            )
            + "\n",
            encoding="utf-8",
        )

    print(
        f"Split seed={args.seed}: "
        f"train={len(train_patients)}/{len(train_rows)}, "
        f"val={len(val_patients)}/{len(val_rows)}, "
        f"holdout={len(holdout_patients)}/{len(hold_rows)}"
    )
    print(f"Holdout patients: {holdout_patients}")
    print(f"Wrote {split_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
