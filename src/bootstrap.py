"""Sanitize import path before loading third-party packages.

Shared hosts sometimes inject ~/.local via PYTHONPATH, which breaks
venv-installed onnx/protobuf/pydicom stacks.
"""

from __future__ import annotations

import os
import sys


def sanitize_sys_path() -> None:
    os.environ["PYTHONNOUSERSITE"] = "1"
    # Shared hosts often inject ~/.local via PYTHONPATH; drop those entries.
    raw_pythonpath = os.environ.get("PYTHONPATH", "")
    if raw_pythonpath:
        kept = []
        for entry in raw_pythonpath.split(os.pathsep):
            norm = entry.replace("\\", "/")
            if not entry or "/.local/lib/python" in norm:
                continue
            kept.append(entry)
        if kept:
            os.environ["PYTHONPATH"] = os.pathsep.join(kept)
        else:
            os.environ.pop("PYTHONPATH", None)

    cleaned: list[str] = []
    for entry in sys.path:
        norm = entry.replace("\\", "/")
        if "/.local/lib/python" in norm:
            continue
        cleaned.append(entry)
    sys.path[:] = cleaned


sanitize_sys_path()
