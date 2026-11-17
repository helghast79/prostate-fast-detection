"""DICOM series loading adapter (sorted by ImagePositionPatient)."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pydicom

from src.geometry import SliceGeometry, geometry_from_tags, parse_float_vector


@dataclass
class DicomSlice:
    sop_instance_uid: str
    instance_number: int
    geometry: SliceGeometry
    pixels: np.ndarray  # float32, slope/intercept applied
    tags: dict[str, Any]


def _tags_from_dataset(ds: pydicom.Dataset) -> dict[str, Any]:
    def get(name: str, default=None):
        return getattr(ds, name, default)

    return {
        "SOPInstanceUID": str(get("SOPInstanceUID", "")),
        "InstanceNumber": get("InstanceNumber", 0),
        "ImagePositionPatient": list(get("ImagePositionPatient")),
        "ImageOrientationPatient": list(get("ImageOrientationPatient")),
        "PixelSpacing": list(get("PixelSpacing")),
        "Rows": int(get("Rows")),
        "Columns": int(get("Columns")),
        "RescaleSlope": float(get("RescaleSlope", 1.0) or 1.0),
        "RescaleIntercept": float(get("RescaleIntercept", 0.0) or 0.0),
        "SliceThickness": float(get("SliceThickness", 0.0) or 0.0),
        "SeriesDescription": str(get("SeriesDescription", "")),
    }


def read_dicom_bytes(data: bytes) -> DicomSlice:
    ds = pydicom.dcmread(io.BytesIO(data))
    tags = _tags_from_dataset(ds)
    geom = geometry_from_tags(tags)
    arr = np.asarray(ds.pixel_array, dtype=np.float32)
    arr = arr * tags["RescaleSlope"] + tags["RescaleIntercept"]
    try:
        instance_number = int(float(tags["InstanceNumber"]))
    except (TypeError, ValueError):
        instance_number = 0
    return DicomSlice(
        sop_instance_uid=tags["SOPInstanceUID"],
        instance_number=instance_number,
        geometry=geom,
        pixels=arr,
        tags=tags,
    )


def sort_slices_by_ipp(slices: Iterable[DicomSlice]) -> list[DicomSlice]:
    slices = list(slices)
    if not slices:
        return []
    normal = slices[0].geometry.normal
    return sorted(slices, key=lambda s: float(s.geometry.ipp @ normal))


def load_dicom_folder(folder: Path) -> list[DicomSlice]:
    """Load all ``.dcm`` / extensionless DICOM files in a folder, IPP-sorted."""
    folder = Path(folder)
    paths = sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".dcm", ".dicom", ""}
    )
    if not paths:
        paths = sorted(p for p in folder.rglob("*") if p.is_file())
    out: list[DicomSlice] = []
    for path in paths:
        try:
            out.append(read_dicom_bytes(path.read_bytes()))
        except Exception:
            continue
    return sort_slices_by_ipp(out)


def stack_volume(slices: list[DicomSlice]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Stack IPP-sorted slices into ``(Z, Y, X)`` and an affine.

    All slices must share rows/cols/IOP/spacing (enforced softly: first wins).
    """
    from .io_nifti import affine_from_dicom_stack

    if not slices:
        raise ValueError("No DICOM slices to stack")
    slices = sort_slices_by_ipp(slices)
    rows = slices[0].geometry.rows
    cols = slices[0].geometry.cols
    vol = np.stack([s.pixels for s in slices], axis=0).astype(np.float32)
    if vol.shape[1] != rows or vol.shape[2] != cols:
        raise ValueError("Inconsistent slice shapes in DICOM series")
    ipps = np.stack([s.geometry.ipp for s in slices], axis=0)
    affine = affine_from_dicom_stack(
        ipps,
        slices[0].geometry.iop,
        (float(slices[0].geometry.spacing[0]), float(slices[0].geometry.spacing[1])),
    )
    meta = {
        "n_slices": len(slices),
        "rows": rows,
        "cols": cols,
        "pixel_spacing_yx": [
            float(slices[0].geometry.spacing[0]),
            float(slices[0].geometry.spacing[1]),
        ],
        "iop": slices[0].geometry.iop.tolist(),
        "ipp_first": slices[0].geometry.ipp.tolist(),
        "ipp_last": slices[-1].geometry.ipp.tolist(),
        "sop_instance_uids": [s.sop_instance_uid for s in slices],
        "instance_numbers": [s.instance_number for s in slices],
        "series_description": slices[0].tags.get("SeriesDescription", ""),
    }
    return vol, affine, meta
