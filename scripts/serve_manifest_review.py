#!/usr/bin/env python3
"""Local web UI to review Orthanc series thumbnails and tag MR/SEG series."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bootstrap import sanitize_sys_path

sanitize_sys_path()

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_from_directory

from src.cascade.orthanc_volumes import segmentation_type
from src.orthanc_client import OrthancClient
from src.seg_preview import resolve_seg_thumbnail_target
from src.series_manifest import (
    LESION_SEG_CHOICES,
    MR_TAGS,
    enrich_patients,
    read_json_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "series_manifest.json",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ROOT / "data" / "series_annotations.json",
    )
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    return parser.parse_args()


def load_annotations(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"series": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"series": {}}
    if not isinstance(data, dict):
        return {"series": {}}
    series = data.get("series")
    if not isinstance(series, dict):
        data["series"] = {}
    return data


def save_annotations(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def create_app(manifest_path: Path, annotations_path: Path) -> Flask:
    app = Flask(
        __name__,
        static_folder=str(ROOT / "web" / "static"),
        static_url_path="/static",
    )
    client = OrthancClient()
    seg_types: dict[str, str | None] = {}

    def seg_type(series: dict[str, Any]) -> str | None:
        """SegmentationType (BINARY / FRACTIONAL) of a SEG series, cached."""
        uid = str(series.get("series_instance_uid", ""))
        if uid not in seg_types:
            seg_types[uid] = segmentation_type(client, series)
        return seg_types[uid]

    @app.get("/")
    def index() -> Any:
        return send_from_directory(ROOT / "web", "index.html")

    @app.get("/api/health")
    def health() -> Any:
        return jsonify(
            {
                "ok": True,
                "orthanc": client.base_url,
                "manifest": str(manifest_path),
                "annotations": str(annotations_path),
            }
        )

    @app.get("/api/manifest")
    def api_manifest() -> Any:
        if not manifest_path.is_file():
            return jsonify({"error": f"Missing manifest: {manifest_path}"}), 404
        patients = read_json_manifest(manifest_path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        annotations = load_annotations(annotations_path)
        enriched = enrich_patients(payload.get("patients") or [], annotations, seg_type)
        return jsonify(
            {
                "generated_at": payload.get("generated_at"),
                "patient_count": len(patients),
                "series_count": sum(len(p.series) for p in patients),
                "mr_tags": list(MR_TAGS),
                "patients": enriched,
            }
        )

    @app.get("/api/annotations")
    def get_annotations() -> Any:
        return jsonify(load_annotations(annotations_path))

    @app.put("/api/annotations")
    def put_annotations() -> Any:
        body = request.get_json(force=True, silent=True) or {}
        series = body.get("series")
        if not isinstance(series, dict):
            return jsonify({"error": "Body must contain object 'series'"}), 400
        cleaned: dict[str, Any] = {"series": {}}
        for uid, value in series.items():
            if not isinstance(value, dict):
                continue
            item: dict[str, Any] = {}
            discarded = bool(value.get("discarded"))
            if discarded:
                item["discarded"] = True
            mr_tag = value.get("mr_tag")
            if mr_tag in MR_TAGS:
                item["mr_tag"] = mr_tag
            elif mr_tag in (None, "", "null"):
                item["mr_tag"] = None
            if "is_prostate_regions" in value:
                flag = value.get("is_prostate_regions")
                if flag is True:
                    item["is_prostate_regions"] = True
                elif flag is False:
                    item["is_prostate_regions"] = False
                else:
                    item["is_prostate_regions"] = None
            if value.get("lesion_seg") in LESION_SEG_CHOICES:
                item["lesion_seg"] = value["lesion_seg"]
            # Keep entries that are discarded and/or tagged.
            if item:
                cleaned["series"][str(uid)] = item
        save_annotations(annotations_path, cleaned)
        return jsonify({"ok": True, "count": len(cleaned["series"])})

    @app.get("/api/thumbnail/<series_id>")
    def thumbnail(series_id: str) -> Any:
        size = request.args.get("size", default=128, type=int) or 128
        size = max(32, min(size, 512))
        instance_id = request.args.get("instance_id") or ""
        frame = request.args.get("frame", default=None, type=int)
        modality = (request.args.get("modality") or "").upper()
        try:
            if modality == "SEG" and not instance_id:
                target = resolve_seg_thumbnail_target(client.base_url, series_id)
                instance_id = str(target["instance_id"])
                frame = target.get("frame_index")
            elif not instance_id:
                series = client.get_series(series_id)
                instances = series.get("Instances") or []
                if not instances:
                    return Response(status=404)
                modality_tags = (series.get("MainDicomTags") or {}).get("Modality", "")
                if str(modality_tags).upper() == "SEG":
                    target = resolve_seg_thumbnail_target(client.base_url, series_id)
                    instance_id = str(target["instance_id"])
                    frame = target.get("frame_index")
                else:
                    instance_id = instances[len(instances) // 2]

            if frame is not None:
                png = client.get_bytes(
                    f"/instances/{instance_id}/frames/{int(frame)}/rendered",
                    {"width": size, "height": size},
                )
            else:
                png = client.download_rendered(instance_id, width=size, height=size)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 502
        return Response(png, mimetype="image/png")

    return app


def main() -> int:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    host = args.host or os.getenv("MANIFEST_REVIEW_HOST", "0.0.0.0")
    port = args.port or int(os.getenv("MANIFEST_REVIEW_PORT", "8765"))
    app = create_app(args.manifest, args.annotations)
    print(f"Manifest: {args.manifest}")
    print(f"Annotations: {args.annotations}")
    print(f"Open http://{host}:{port}/")
    app.run(host=host, port=port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
