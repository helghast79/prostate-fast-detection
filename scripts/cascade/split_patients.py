#!/usr/bin/env python3
"""Patient-level train / validation / test split for the cascade NIfTI dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.bootstrap import sanitize_sys_path

sanitize_sys_path()

from src.cascade.splits import list_patients, patient_split, write_split_files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=ROOT / "data" / "cascade")
    p.add_argument("--n-test", type=int, default=3)
    p.add_argument("--n-val", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    patients = list_patients(args.dataset)
    if not patients:
        print(f"No patients under {args.dataset}/images")
        return 1
    train, val, test = patient_split(
        patients, n_test=args.n_test, n_val=args.n_val, seed=args.seed
    )
    write_split_files(args.dataset, train, val, test)
    print(
        f"patients={len(patients)}  train={len(train)}  "
        f"val={len(val)}  test={len(test)}  → {args.dataset / 'splits'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
