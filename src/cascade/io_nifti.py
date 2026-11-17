"""NIfTI read/write with cascade conventions.

Volumes are stored as ``(Z, Y, X)`` = ``(slices, rows, cols)``.
The affine maps voxel ``(i=col, j=row, k=slice)`` into patient LPS mm.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


def affine_from_dicom_stack(
    ipps: np.ndarray,
    iop: np.ndarray,
    spacing_yx: tuple[float, float],
) -> np.ndarray:
    """
    Build a 4×4 affine for a sorted axial stack.

    Parameters
    ----------
    ipps:
        ``(Z, 3)`` ImagePositionPatient per slice (already IPP-sorted).
    iop:
        Length-6 ImageOrientationPatient of the first slice.
    spacing_yx:
        ``(row_spacing, col_spacing)`` = PixelSpacing.
    """
    ipps = np.asarray(ipps, dtype=np.float64)
    iop = np.asarray(iop, dtype=np.float64).reshape(6)
    row_dir = iop[:3]
    col_dir = iop[3:]
    row_sp, col_sp = float(spacing_yx[0]), float(spacing_yx[1])

    if len(ipps) >= 2:
        slice_vec = ipps[1] - ipps[0]
    else:
        normal = np.cross(row_dir, col_dir)
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        slice_vec = normal  # 1 mm fallback thickness

    # NIfTI voxel axes: i→columns (X), j→rows (Y), k→slices (Z)
    affine = np.eye(4, dtype=np.float64)
    affine[:3, 0] = col_dir * col_sp
    affine[:3, 1] = row_dir * row_sp
    affine[:3, 2] = slice_vec
    affine[:3, 3] = ipps[0]
    return affine


def save_nifti(
    path: Path,
    volume: np.ndarray,
    affine: np.ndarray,
    *,
    dtype: np.dtype | type = np.float32,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # nibabel expects array shaped (X, Y, Z) = (cols, rows, slices)
    arr = np.asarray(volume, dtype=dtype)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {arr.shape}")
    xyz = np.transpose(arr, (2, 1, 0))  # (Z,Y,X) → (X,Y,Z)
    img = nib.Nifti1Image(xyz, affine=np.asarray(affine, dtype=np.float64))
    nib.save(img, str(path))


def load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(volume ZYX, affine)``."""
    img = nib.load(str(path))
    xyz = np.asanyarray(img.dataobj)
    if xyz.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got shape {xyz.shape}")
    zyx = np.transpose(xyz, (2, 1, 0))
    return np.asarray(zyx), np.asarray(img.affine, dtype=np.float64)


def save_patient_meta(path: Path, meta: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def load_patient_meta(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dataset_paths(root: Path, patient_id: str) -> dict[str, Path]:
    root = Path(root)
    return {
        "image": root / "images" / f"{patient_id}.nii.gz",
        "mask": root / "masks" / f"{patient_id}.nii.gz",
        "meta": root / "meta" / f"{patient_id}.json",
    }
