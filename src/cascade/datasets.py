"""PyTorch datasets for localization and ROI segmentation stages."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .decisions import CASCADE_DECISIONS
from .io_nifti import dataset_paths, load_nifti
from .labels import normalize_center, volume_slice_labels
from .roi import compute_square_roi, extract_roi, extract_roi_mask, jitter_center
from .splits import read_split_file


def _resize2d(arr: np.ndarray, size: int, *, nearest: bool = False) -> np.ndarray:
    if arr.dtype != np.uint8:
        if nearest:
            u8 = (arr > 0).astype(np.uint8) * 255
            out = Image.fromarray(u8, mode="L").resize(
                (size, size), resample=Image.Resampling.NEAREST
            )
            return (np.asarray(out) > 127).astype(np.float32)
        u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        out = Image.fromarray(u8, mode="L").resize(
            (size, size), resample=Image.Resampling.BILINEAR
        )
        return np.asarray(out, dtype=np.float32) / 255.0
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    out = Image.fromarray(arr, mode="L").resize((size, size), resample=resample)
    if nearest:
        return (np.asarray(out) > 127).astype(np.float32)
    return np.asarray(out, dtype=np.float32) / 255.0


class _VolumeCache:
    """Lazy per-patient image/mask cache (shared within a dataset instance)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._images: dict[str, np.ndarray] = {}
        self._masks: dict[str, np.ndarray] = {}

    def image(self, patient_id: str) -> np.ndarray:
        if patient_id not in self._images:
            paths = dataset_paths(self.root, patient_id)
            img, _ = load_nifti(paths["image"])
            self._images[patient_id] = np.asarray(img, dtype=np.float32)
        return self._images[patient_id]

    def mask(self, patient_id: str) -> np.ndarray:
        if patient_id not in self._masks:
            paths = dataset_paths(self.root, patient_id)
            msk, _ = load_nifti(paths["mask"])
            self._masks[patient_id] = np.asarray(msk)
        return self._masks[patient_id]


class LocalizerSliceDataset(Dataset):
    """Full-slice 128×128 samples with presence + normalized centre."""

    def __init__(
        self,
        dataset_root: Path,
        patient_ids: Sequence[str],
        *,
        image_size: int | None = None,
        augment: bool = False,
        neg_ratio: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.root = Path(dataset_root)
        self.image_size = image_size or CASCADE_DECISIONS.localizer_input_size
        self.augment = augment
        self.cache = _VolumeCache(self.root)
        self.samples: list[dict] = []
        rng = np.random.default_rng(seed)

        positives: list[dict] = []
        negatives: list[dict] = []
        for pid in patient_ids:
            image = self.cache.image(pid)
            mask = self.cache.mask(pid)
            labels = volume_slice_labels(mask)
            rows, cols = image.shape[1], image.shape[2]
            for lab in labels:
                item = {
                    "patient_id": pid,
                    "z": lab.z,
                    "present": lab.present,
                    "center_yx": lab.center_yx,
                    "rows": rows,
                    "cols": cols,
                }
                if lab.present:
                    positives.append(item)
                else:
                    negatives.append(item)

        n_pos = len(positives)
        n_neg_keep = (
            min(len(negatives), max(1, int(n_pos * neg_ratio))) if n_pos else len(negatives)
        )
        if negatives and n_neg_keep < len(negatives):
            idx = rng.choice(len(negatives), size=n_neg_keep, replace=False)
            negatives = [negatives[i] for i in idx]
        self.samples = positives + negatives
        rng.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int | bool]:
        item = dict(self.samples[index])
        sl = self.cache.image(item["patient_id"])[item["z"]].astype(np.float32).copy()
        center_yx = item["center_yx"]

        if self.augment:
            if np.random.rand() < 0.5:
                sl = np.fliplr(sl).copy()
                if center_yx is not None:
                    cy, cx = center_yx
                    center_yx = (cy, (item["cols"] - 1) - cx)
            gain = float(np.random.uniform(0.9, 1.1))
            bias = float(np.random.uniform(-0.05, 0.05))
            sl = np.clip(sl * gain + bias, 0.0, 1.0)

        sl_r = _resize2d(sl, self.image_size, nearest=False)
        present = bool(item["present"])
        if present and center_yx is not None:
            cy, cx = normalize_center(center_yx, item["rows"], item["cols"])
            center = np.array([cy, cx], dtype=np.float32)
        else:
            center = np.zeros(2, dtype=np.float32)

        return {
            "image": torch.from_numpy(sl_r[None, ...]),
            "present": torch.tensor(1.0 if present else 0.0, dtype=torch.float32),
            "center": torch.from_numpy(center),
            "patient_id": item["patient_id"],
            "z": int(item["z"]),
        }


class SegmenterRoiDataset(Dataset):
    """ROI crops for binary segmentation (GT centres + optional jitter)."""

    def __init__(
        self,
        dataset_root: Path,
        patient_ids: Sequence[str],
        *,
        crop_size: int | None = None,
        augment: bool = False,
        jitter_centers: bool | None = None,
        positives_only: bool = True,
        seed: int = 0,
    ) -> None:
        self.root = Path(dataset_root)
        self.crop_size = crop_size or CASCADE_DECISIONS.roi_crop_size
        self.augment = augment
        self.jitter_centers = (
            CASCADE_DECISIONS.roi_train_with_noisy_centers
            if jitter_centers is None
            else jitter_centers
        )
        self.rng = np.random.default_rng(seed)
        self.cache = _VolumeCache(self.root)
        self.samples: list[dict] = []

        for pid in patient_ids:
            mask = self.cache.mask(pid)
            labels = volume_slice_labels(mask)
            for lab in labels:
                if positives_only and not lab.present:
                    continue
                self.samples.append(
                    {
                        "patient_id": pid,
                        "z": lab.z,
                        "center_yx": lab.center_yx,
                        "present": lab.present,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        item = self.samples[index]
        sl = self.cache.image(item["patient_id"])[item["z"]].astype(np.float32)
        mk = (self.cache.mask(item["patient_id"])[item["z"]] > 0).astype(np.uint8)
        rows, cols = sl.shape

        center = item["center_yx"]
        if center is None:
            center = (rows / 2.0, cols / 2.0)
        if self.jitter_centers and self.augment and item["present"]:
            center = jitter_center(center, self.crop_size, rng=self.rng)

        crop = compute_square_roi(center, rows, cols, size=self.crop_size)
        img_roi = extract_roi(sl, crop)
        msk_roi = extract_roi_mask(mk, crop)

        if self.augment and np.random.rand() < 0.5:
            img_roi = np.fliplr(img_roi).copy()
            msk_roi = np.fliplr(msk_roi).copy()

        return {
            "image": torch.from_numpy(img_roi[None, ...].astype(np.float32)),
            "mask": torch.from_numpy(msk_roi[None, ...].astype(np.float32)),
            "patient_id": item["patient_id"],
            "z": int(item["z"]),
        }


def load_split_patients(dataset_root: Path, split: str) -> list[str]:
    name = {
        "train": "train.txt",
        "val": "validation.txt",
        "validation": "validation.txt",
        "test": "test.txt",
        "holdout": "test.txt",
    }[split]
    return read_split_file(Path(dataset_root) / "splits" / name)
