#!/usr/bin/env python3
"""Build a DICOM-only gland + peripheral-zone training catalog.

Uses ProCAncer-I ``subsets/studies_with_masks`` (symlink names =
``PatientID_StudyInstanceUID``) to locate studies under ``dicom/``, then
selects:

  - best zone SEG (TZ+CZ / PZ labels): prefer ``*_Prostate_Final`` over
    ``*_Prostate`` over ``*autoseg*``
  - axial T2 referenced by that SEG when possible, else the largest axial T2

Writes ONLY under ``--out`` (default ``data/procancer_zones_dicom``). Never
modifies the ProCAncer-I tree. Series folders are symlinked, not copied.

  cases/<patient>/<study>/
    t2/          -> dicom series dir
    zone_seg/    -> dicom SEG series dir
    meta.json
  previews/<patient>__<study>.png   # T2 mid-lesion-slice with gland/PZ overlay
  index.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.bootstrap import sanitize_sys_path

sanitize_sys_path()

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.procancer_dicom import (
    iter_masks_studies,
    pick_axial_t2,
    pick_zone_seg,
    scan_study,
)
from src.procancer_zones import load_t2_and_zones


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--procancer-root",
        type=Path,
        default=Path("/usr/backups/datasets/ProCAncer-I"),
    )
    p.add_argument("--out", type=Path, default=ROOT / "data" / "procancer_zones_dicom")
    p.add_argument("--previews", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def assert_outside(out: Path, procancer: Path) -> None:
    out_r, pro_r = out.resolve(), procancer.resolve()
    if out_r == pro_r or pro_r in out_r.parents:
        raise SystemExit(f"refusing to write inside ProCAncer-I: {out_r}")


def safe_symlink(src: Path, dst: Path, force: bool) -> None:
    if dst.is_symlink() or dst.exists():
        if not force:
            return
        if dst.is_dir() and not dst.is_symlink():
            raise SystemExit(f"refusing to replace real directory {dst}")
        dst.unlink()
    dst.symlink_to(src.resolve())


def overlay_zones(t2_zyx: np.ndarray, mask: np.ndarray, out_png: Path) -> int:
    """Draw gland (teal) + PZ (amber) on the slice with most zone voxels."""
    counts = (mask > 0).reshape(len(mask), -1).sum(1)
    z = int(np.argmax(counts)) if counts.any() else len(mask) // 2
    sl = t2_zyx[z].astype(np.float32)
    lo, hi = np.percentile(sl, (1, 99))
    if hi <= lo:
        hi = lo + 1.0
    gray = np.clip((sl - lo) / (hi - lo), 0, 1)
    rgb = np.stack([gray, gray, gray], axis=-1)
    g = mask[z] == 1
    p = mask[z] == 2
    rgb[g, 1] = np.maximum(rgb[g, 1], 0.75)
    rgb[g, 2] = np.maximum(rgb[g, 2], 0.55)
    rgb[p, 0] = np.maximum(rgb[p, 0], 0.95)
    rgb[p, 1] = np.maximum(rgb[p, 1], 0.55)
    rgb[p, 2] *= 0.25
    Image.fromarray((rgb * 255).astype(np.uint8)).save(out_png)
    return z


def build_one(
    patient_id: str,
    study_uid: str,
    study_dir: Path,
    case_dir: Path,
    preview_path: Path | None,
    *,
    force: bool,
    make_preview: bool,
) -> dict:
    series = scan_study(study_dir) if study_dir.is_dir() else []
    seg = pick_zone_seg(series)
    t2 = pick_axial_t2(series, seg.referenced_series_uids if seg else None)
    ready = bool(seg and t2)

    case_dir.mkdir(parents=True, exist_ok=True)
    if t2:
        safe_symlink(Path(t2.series_dir), case_dir / "t2", force)
    if seg:
        safe_symlink(Path(seg.series_dir), case_dir / "zone_seg", force)

    meta: dict = {
        "patient_id": patient_id,
        "study_instance_uid": study_uid,
        "dicom_study_path": str(study_dir) if study_dir.is_dir() else None,
        "ready": ready,
        "t2": t2.to_dict() if t2 else None,
        "zone_seg": seg.to_dict() if seg else None,
        "n_series_in_study": len(series),
        "series": [s.to_dict() for s in series],
        "preview": None,
        "preview_slice": None,
        "gland_voxels": None,
        "pz_voxels": None,
        "matched_planes": None,
        "unmatched_planes": None,
        "discarded": False,
    }

    if ready and make_preview and preview_path is not None:
        try:
            slices, mask, info = load_t2_and_zones(Path(t2.series_dir), Path(seg.series_dir))
            vol = np.stack([s.pixels for s in slices], axis=0)
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            if force or not preview_path.is_file():
                z = overlay_zones(vol, mask, preview_path)
            else:
                z = None
            meta["preview"] = str(preview_path) if preview_path.is_file() else None
            meta["preview_slice"] = z
            meta["gland_voxels"] = int((mask == 1).sum())
            meta["pz_voxels"] = int((mask == 2).sum())
            meta["matched_planes"] = info["matched_planes"]
            meta["unmatched_planes"] = info["unmatched_planes"]
            meta["shape_zyx"] = list(mask.shape)
        except (PermissionError, OSError, ValueError, RuntimeError) as exc:
            meta["preview_error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001
            meta["preview_error"] = f"{type(exc).__name__}: {exc}"

    (case_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    # Slim index entry (drop full series list to keep index manageable).
    slim = {k: v for k, v in meta.items() if k != "series"}
    slim["n_series_in_study"] = meta["n_series_in_study"]
    return slim


def main() -> int:
    args = parse_args()
    assert_outside(args.out, args.procancer_root)
    masks_list = args.procancer_root / "subsets" / "studies_with_masks"
    dicom_root = args.procancer_root / "dicom"
    if not masks_list.is_dir():
        raise SystemExit(f"missing {masks_list}")

    cases_root = args.out / "cases"
    previews_root = args.out / "previews"
    cases_root.mkdir(parents=True, exist_ok=True)

    items = list(iter_masks_studies(masks_list, dicom_root))
    if args.limit is not None:
        items = items[: args.limit]

    index: list[dict] = []
    for patient_id, study_uid, study_dir, _link in tqdm(items, desc="Zones DICOM"):
        case_dir = cases_root / patient_id / study_uid
        preview = previews_root / f"{patient_id}__{study_uid}.png"
        try:
            entry = build_one(
                patient_id,
                study_uid,
                study_dir,
                case_dir,
                preview if args.previews else None,
                force=args.force,
                make_preview=args.previews,
            )
        except Exception as exc:  # noqa: BLE001
            entry = {
                "patient_id": patient_id,
                "study_instance_uid": study_uid,
                "dicom_study_path": str(study_dir),
                "ready": False,
                "error": f"{type(exc).__name__}: {exc}",
                "discarded": False,
            }
        index.append(entry)

    index.sort(key=lambda m: (not m["ready"], m["patient_id"]))
    summary = {
        "source_list": str(masks_list),
        "dicom_root": str(dicom_root),
        "out": str(args.out),
        "n_studies": len(index),
        "n_ready": sum(1 for m in index if m["ready"]),
        "n_missing_dicom": sum(1 for m in index if not m.get("dicom_study_path")),
        "n_preview_ok": sum(1 for m in index if m.get("preview")),
        "n_preview_error": sum(1 for m in index if m.get("preview_error")),
        "cases": index,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"wrote {args.out / 'index.json'}: {summary['n_ready']}/{summary['n_studies']} ready, "
        f"{summary['n_preview_ok']} previews"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
