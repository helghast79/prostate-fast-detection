"""Read-only MetaImage (.mha) helpers for ProCAncer-I nifti volumes."""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np

_DTYPES = {
    "MET_UCHAR": np.uint8,
    "MET_CHAR": np.int8,
    "MET_USHORT": np.uint16,
    "MET_SHORT": np.int16,
    "MET_UINT": np.uint32,
    "MET_INT": np.int32,
    "MET_FLOAT": np.float32,
    "MET_DOUBLE": np.float64,
}


def read_mha_header(path: Path) -> tuple[dict[str, str], int]:
    """Return (header dict, byte offset of pixel data)."""
    header: dict[str, str] = {}
    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"truncated mha header: {path}")
            if b"=" not in line:
                continue
            key, value = line.decode("ascii", "replace").split("=", 1)
            header[key.strip()] = value.strip()
            if key.strip() == "ElementDataFile":
                return header, f.tell()
    raise ValueError(f"missing ElementDataFile: {path}")


def read_mha(path: Path | str) -> tuple[np.ndarray, tuple[float, float, float], dict[str, str]]:
    """Load a (possibly zlib-compressed) MetaImage. Array is z,y,x."""
    path = Path(path)
    header, offset = read_mha_header(path)
    dims = tuple(int(x) for x in header["DimSize"].split())  # x y z
    spacing = tuple(float(x) for x in header.get("ElementSpacing", "1 1 1").split()[:3])
    dtype = _DTYPES[header.get("ElementType", "MET_UCHAR")]
    with path.open("rb") as f:
        f.seek(offset)
        raw = f.read()
    if header.get("CompressedData", "").lower() == "true":
        raw = zlib.decompress(raw)
    arr = np.frombuffer(raw, dtype=dtype).reshape(dims[2], dims[1], dims[0])
    return arr, (spacing[0], spacing[1], spacing[2]), header


def lesion_volume_ml(mask: np.ndarray, spacing: tuple[float, float, float]) -> float:
    return float((mask > 0).sum()) * spacing[0] * spacing[1] * spacing[2] / 1000.0
