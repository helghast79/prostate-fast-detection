#!/usr/bin/env python3
"""Review the ProCAncer-I lesion symlink catalog (discard / keep).

Reads data/procancer_lesion/index.json and writes discards to
data/procancer_lesion/annotations.json. Never touches the ProCAncer-I tree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bootstrap import sanitize_sys_path

sanitize_sys_path()

from flask import Flask, jsonify, request, send_from_directory


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, default=ROOT / "data" / "procancer_lesion")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8770)
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
        static_folder=str(ROOT / "web_procancer" / "static"),
        static_url_path="/static",
    )
    index_path = catalog / "index.json"
    ann_path = catalog / "annotations.json"

    def annotations() -> dict[str, Any]:
        data = load_json(ann_path, {"cases": {}})
        if not isinstance(data.get("cases"), dict):
            data["cases"] = {}
        return data

    def save_ann(data: dict[str, Any]) -> None:
        ann_path.parent.mkdir(parents=True, exist_ok=True)
        ann_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

    @app.get("/")
    def index() -> Any:
        return send_from_directory(ROOT / "web_procancer", "index.html")

    @app.get("/api/catalog")
    def api_catalog() -> Any:
        summary = load_json(index_path, {})
        ann = annotations()["cases"]
        cases = []
        for c in summary.get("cases") or []:
            key = f"{c['patient_id']}/{c['study_instance_uid']}"
            a = ann.get(key) or {}
            cases.append({**c, "discarded": bool(a.get("discarded", c.get("discarded"))), "note": a.get("note", "")})
        return jsonify(
            {
                "source": summary.get("source"),
                "n_studies": summary.get("n_studies"),
                "n_complete": summary.get("n_complete"),
                "n_incomplete": summary.get("n_incomplete"),
                "cases": cases,
            }
        )

    @app.put("/api/case")
    def api_case() -> Any:
        body = request.get_json(force=True) or {}
        pid, sid = body.get("patient_id"), body.get("study_instance_uid")
        if not pid or not sid:
            return jsonify({"ok": False, "error": "patient_id and study_instance_uid required"}), 400
        key = f"{pid}/{sid}"
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
        raise SystemExit(f"missing {args.catalog / 'index.json'}; run scripts/lesion/build_procancer_catalog.py first")
    app = create_app(args.catalog)
    print(f"catalog {args.catalog} on http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
