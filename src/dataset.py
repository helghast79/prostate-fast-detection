"""PyTorch dataset over downloaded T2/mask PNG pairs."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset


def load_manifest(data_root: Path) -> list[dict]:
    manifest_path = data_root / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    rows: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def filter_existing_pairs(data_root: Path, rows: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for row in rows:
        if (data_root / row["image_path"]).is_file() and (data_root / row["mask_path"]).is_file():
            kept.append(row)
    return kept


def filter_nonempty_masks(rows: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for row in rows:
        fg = row.get("mask_foreground_pixels")
        if fg is None or int(fg) > 0:
            kept.append(row)
    return kept


def patient_split(
    rows: list[dict],
    n_val_patients: int = 8,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[str]]:
    patients = sorted({r["patient_id"] for r in rows})
    rng = random.Random(seed)
    shuffled = patients[:]
    rng.shuffle(shuffled)
    n_val = min(max(0, n_val_patients), max(0, len(patients) - 1))
    val_ids = sorted(shuffled[:n_val])
    val_set = set(val_ids)
    train_rows = [r for r in rows if r["patient_id"] not in val_set]
    val_rows = [r for r in rows if r["patient_id"] in val_set]
    if not train_rows:
        return rows, [], []
    return train_rows, val_rows, val_ids


class ProstateSegDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        rows: list[dict],
        augment: bool = False,
        seed: int = 0,
        multiclass: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.rows = rows
        self.augment = augment
        self.rng = random.Random(seed)
        self.multiclass = bool(multiclass)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[idx]
        image = np.asarray(
            Image.open(self.data_root / row["image_path"]).convert("L"), dtype=np.float32
        ) / 255.0
        mask_u8 = np.asarray(
            Image.open(self.data_root / row["mask_path"]).convert("L"), dtype=np.uint8
        )
        if self.multiclass:
            # Class-index mask stored as 0/1/2 (not scaled to 255).
            mask = mask_u8.astype(np.int64)
            if mask.max() > 10:
                # Safety: if a binary 0/255 slipped in, collapse to gland-only.
                mask = (mask > 127).astype(np.int64)
        else:
            mask = (mask_u8 > 127).astype(np.float32)

        if self.augment:
            if self.rng.random() < 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
            if self.rng.random() < 0.5:
                k = self.rng.randint(1, 3)
                image = np.rot90(image, k).copy()
                mask = np.rot90(mask, k).copy()
            if self.rng.random() < 0.7:
                factor = self.rng.uniform(0.85, 1.15)
                image = np.clip(image * factor, 0.0, 1.0)
            if self.rng.random() < 0.3:
                image = ndimage.gaussian_filter(image, sigma=self.rng.uniform(0.3, 0.8))
                image = np.clip(image, 0.0, 1.0).astype(np.float32)

        if self.multiclass:
            mask_t = torch.from_numpy(mask.astype(np.int64))
        else:
            mask_t = torch.from_numpy(mask.astype(np.float32)[None, ...])

        return {
            "image": torch.from_numpy(image[None, ...]),
            "mask": mask_t,
            "patient_id": row["patient_id"],
            "sop_instance_uid": row.get("sop_instance_uid", ""),
        }
