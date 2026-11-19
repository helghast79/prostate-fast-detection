#!/usr/bin/env python3
"""Review the DICOM gland+PZ catalog (discard / keep / per-slice overlays).

Reads data/procancer_zones_dicom/index.json; writes discards to
data/procancer_zones_dicom/annotations.json. Never touches ProCAncer-I.
Slice overlays are cached under data/procancer_zones_dicom/slice_cache/.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bootstrap import sanitize_sys_path

sanitize_sys_path()

from flask import Flask, Response, jsonify, request, send_from_directory

from src.procancer_zones import load_t2_and_zones, overlay_zone_rgb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, default=ROOT / "data" / "procancer_zones_dicom")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8771)
    return p.parse_args()


def load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


def create_app(catalog: Path) -> Flask:
    app = Flask(
        __name__,
        static_folder=str(ROOT / "web_zones" / "static"),
        static_url_path="/static",
    )
    index_path = catalog / "index.json"
    ann_path = catalog / "annotations.json"
    cache_root = catalog / "slice_cache"
    volume_lock = threading.Lock()
    volume_cache: dict[str, dict[str, Any]] = {}

    def annotations() -> dict[str, Any]:
        data = load_json(ann_path, {"cases": {}})
        if not isinstance(data.get("cases"), dict):
            data["cases"] = {}
        return data

    def save_ann(data: dict[str, Any]) -> None:
        ann_path.parent.mkdir(parents=True, exist_ok=True)
        ann_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

    def case_key(pid: str, sid: str) -> str:
        return f"{pid}/{sid}"

    def find_case(pid: str, sid: str) -> dict[str, Any] | None:
        summary = load_json(index_path, {})
        for c in summary.get("cases") or []:
            if c.get("patient_id") == pid and c.get("study_instance_uid") == sid:
                a = annotations()["cases"].get(case_key(pid, sid)) or {}
                return {
                    **c,
                    "discarded": bool(a.get("discarded", c.get("discarded"))),
                    "note": a.get("note", ""),
                }
        return None

    def case_dirs(pid: str, sid: str) -> tuple[Path, Path]:
        case_dir = catalog / "cases" / pid / sid
        return case_dir / "t2", case_dir / "zone_seg"

    def load_volume(pid: str, sid: str) -> dict[str, Any]:
        key = case_key(pid, sid)
        with volume_lock:
            hit = volume_cache.get(key)
            if hit is not None:
                return hit
        t2_dir, seg_dir = case_dirs(pid, sid)
        if not t2_dir.is_dir() or not seg_dir.is_dir():
            raise FileNotFoundError(f"missing t2/zone_seg symlinks for {key}")
        slices, mask, info = load_t2_and_zones(t2_dir, seg_dir)
        vol = np.stack([s.pixels for s in slices], axis=0).astype(np.float32)
        lo, hi = np.percentile(vol, (1, 99))
        if hi <= lo:
            hi = float(lo) + 1.0
        packed = {
            "vol": vol,
            "mask": mask,
            "lo": float(lo),
            "hi": float(hi),
            "info": info,
            "has_mask": (mask.reshape(len(mask), -1).sum(1) > 0).astype(bool).tolist(),
        }
        with volume_lock:
            # Keep a small LRU-ish cache (last few reviewed patients).
            if len(volume_cache) >= 3 and key not in volume_cache:
                volume_cache.pop(next(iter(volume_cache)))
            volume_cache[key] = packed
        return packed

    def ensure_slice_png(pid: str, sid: str, z: int) -> Path:
        out_dir = cache_root / f"{pid}__{sid}"
        out = out_dir / f"{z:03d}.png"
        if out.is_file():
            return out
        packed = load_volume(pid, sid)
        if z < 0 or z >= len(packed["vol"]):
            raise IndexError(f"slice {z} out of range")
        rgb = overlay_zone_rgb(
            packed["vol"][z],
            packed["mask"][z],
            lo=packed["lo"],
            hi=packed["hi"],
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(out)
        return out

    @app.get("/")
    def index() -> Any:
        return send_from_directory(ROOT / "web_zones", "index.html")

    @app.get("/case")
    def case_page() -> Any:
        return send_from_directory(ROOT / "web_zones", "case.html")

    @app.get("/api/catalog")
    def api_catalog() -> Any:
        summary = load_json(index_path, {})
        ann = annotations()["cases"]
        cases = []
        for c in summary.get("cases") or []:
            key = case_key(c["patient_id"], c["study_instance_uid"])
            a = ann.get(key) or {}
            cases.append(
                {
                    **c,
                    "discarded": bool(a.get("discarded", c.get("discarded"))),
                    "note": a.get("note", ""),
                }
            )
        return jsonify(
            {
                "source": summary.get("source_list"),
                "n_studies": summary.get("n_studies"),
                "n_ready": summary.get("n_ready"),
                "cases": cases,
            }
        )

    @app.get("/api/case/<pid>/<sid>")
    def api_case_get(pid: str, sid: str) -> Any:
        meta = find_case(pid, sid)
        if meta is None:
            return jsonify({"ok": False, "error": "case not found"}), 404
        try:
            packed = load_volume(pid, sid)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 500
        return jsonify(
            {
                "ok": True,
                "case": meta,
                "n_slices": int(len(packed["vol"])),
                "has_mask": packed["has_mask"],
                "matched_planes": packed["info"].get("matched_planes"),
                "unmatched_planes": packed["info"].get("unmatched_planes"),
            }
        )

    @app.get("/api/case/<pid>/<sid>/slice/<int:z>.png")
    def api_case_slice(pid: str, sid: str, z: int) -> Any:
        try:
            path = ensure_slice_png(pid, sid, z)
        except FileNotFoundError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404
        except IndexError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 500
        return send_from_directory(path.parent, path.name, mimetype="image/png")

    @app.put("/api/case")
    def api_case() -> Any:
        body = request.get_json(force=True) or {}
        pid, sid = body.get("patient_id"), body.get("study_instance_uid")
        if not pid or not sid:
            return jsonify({"ok": False, "error": "patient_id and study_instance_uid required"}), 400
        key = case_key(pid, sid)
        data = annotations()
        cur = data["cases"].setdefault(key, {})
        if "discarded" in body:
            cur["discarded"] = bool(body["discarded"])
        if "note" in body:
            cur["note"] = str(body["note"])
        save_ann(data)
        return jsonify({"ok": True, "case": cur})

    @app.get("/previews/<path:name>")
    def preview(name: str) -> Any:
        return send_from_directory(catalog / "previews", name)

    return app


def main() -> int:
    args = parse_args()
    if not (args.catalog / "index.json").is_file():
        raise SystemExit(
            f"missing {args.catalog / 'index.json'}; run scripts/zones/build_dicom_catalog.py first"
        )
    app = create_app(args.catalog)
    print(f"zones catalog {args.catalog} on http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
