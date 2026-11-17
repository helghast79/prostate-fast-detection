"""Patient-level train/val/test split helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence


def list_patients(dataset_root: Path) -> list[str]:
    images = Path(dataset_root) / "images"
    if not images.is_dir():
        return []
    return sorted(p.name.replace(".nii.gz", "") for p in images.glob("*.nii.gz"))


def write_split_files(
    dataset_root: Path,
    train: Sequence[str],
    validation: Sequence[str],
    test: Sequence[str],
) -> dict[str, Path]:
    splits_dir = Path(dataset_root) / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "train.txt": list(train),
        "validation.txt": list(validation),
        "test.txt": list(test),
    }
    paths: dict[str, Path] = {}
    for name, ids in mapping.items():
        path = splits_dir / name
        path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
        paths[name] = path
    meta = {
        "n_train": len(train),
        "n_validation": len(validation),
        "n_test": len(test),
        "train": list(train),
        "validation": list(validation),
        "test": list(test),
    }
    (splits_dir / "split.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return paths


def read_split_file(path: Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def patient_split(
    patients: Sequence[str],
    *,
    n_test: int,
    n_val: int,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    ids = list(patients)
    rng = random.Random(seed)
    rng.shuffle(ids)
    if n_test + n_val > len(ids):
        raise ValueError(
            f"n_test({n_test})+n_val({n_val}) exceeds n_patients({len(ids)})"
        )
    test = sorted(ids[:n_test])
    validation = sorted(ids[n_test : n_test + n_val])
    train = sorted(ids[n_test + n_val :])
    return train, validation, test
