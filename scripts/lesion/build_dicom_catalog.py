#!/usr/bin/env python3
"""Build a DICOM-only lesion training catalog.

Uses ProCAncer-I ``subsets/studies_with_lesion`` (symlink names =
``StudyInstanceUID``, targets under nifti_new) to locate studies under
``dicom/<PatientID>/<StudyUID>/``, then selects:

  - lesion SEG (prefer ``*_Lesions_Final``, lesion-labelled, non-autoseg)
  - axial T2 referenced by that SEG when possible
  - ADC and DWI (largest / most instances)

Writes ONLY under ``--out`` (default ``data/procancer_lesion_dicom``). Never
modifies the ProCAncer-I tree. Series folders are symlinked, not copied.

  cases/<patient>/<study>/
    t2/          -> dicom series dir
    adc/         -> dicom series dir
    dwi/         -> dicom series dir
    lesion_seg/  -> dicom SEG series dir
    meta.json
  previews/<patient>__<study>.png   # T2 mid-lesion-slice with magenta overlay
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
    iter_lesion_studies,
    pick_axial_t2,
    pick_lesion_seg,
    pick_mr_tag,
    scan_study,
)
from src.procancer_lesion import load_t2_and_lesion


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--procancer-root",
        type=Path,
        default=Path("/usr/backups/datasets/ProCAncer-I"),
    )
    p.add_argument("--out", type=Path, default=ROOT / "data" / "procancer_lesion_dicom")
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


def overlay_lesion(t2_zyx: np.ndarray, mask: np.ndarray, out_png: Path) -> int:
    """Draw magenta lesion on the slice with most lesion voxels."""
    counts = (mask > 0).reshape(len(mask), -1).sum(1)
    z = int(np.argmax(counts)) if counts.any() else len(mask) // 2
    sl = t2_zyx[z].astype(np.float32)
    lo, hi = np.percentile(sl, (1, 99))
    if hi <= lo:
        hi = lo + 1.0
    gray = np.clip((sl - lo) / (hi - lo), 0, 1)
    rgb = np.stack([gray, gray, gray], axis=-1)
    m = mask[z] > 0
    rgb[m, 0] = np.maximum(rgb[m, 0], 0.95)
    rgb[m, 1] *= 0.25
    rgb[m, 2] = np.maximum(rgb[m, 2], 0.85)
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
    seg = pick_lesion_seg(series)
    refs = seg.referenced_series_uids if seg else None
    t2 = pick_axial_t2(series, refs)
    adc = pick_mr_tag(series, "adc")
    dwi = pick_mr_tag(series, "dwi")
    ready = bool(seg and t2 and adc and dwi)

    case_dir.mkdir(parents=True, exist_ok=True)
    if t2:
        safe_symlink(Path(t2.series_dir), case_dir / "t2", force)
    if adc:
        safe_symlink(Path(adc.series_dir), case_dir / "adc", force)
    if dwi:
        safe_symlink(Path(dwi.series_dir), case_dir / "dwi", force)
    if seg:
        safe_symlink(Path(seg.series_dir), case_dir / "lesion_seg", force)

    meta: dict = {
        "patient_id": patient_id,
        "study_instance_uid": study_uid,
        "dicom_study_path": str(study_dir) if study_dir.is_dir() else None,
        "ready": ready,
        "t2": t2.to_dict() if t2 else None,
        "adc": adc.to_dict() if adc else None,
        "dwi": dwi.to_dict() if dwi else None,
        "lesion_seg": seg.to_dict() if seg else None,
        "n_series_in_study": len(series),
        "series": [s.to_dict() for s in series],
        "preview": None,
        "preview_slice": None,
        "lesion_voxels": None,
        "matched_planes": None,
        "unmatched_planes": None,
        "discarded": False,
    }

    if seg and t2 and make_preview and preview_path is not None:
        try:
            slices, mask, info = load_t2_and_lesion(Path(t2.series_dir), Path(seg.series_dir))
            vol = np.stack([s.pixels for s in slices], axis=0)
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            if force or not preview_path.is_file():
                z = overlay_lesion(vol, mask, preview_path)
            else:
                z = None
            meta["preview"] = str(preview_path) if preview_path.is_file() else None
            meta["preview_slice"] = z
            meta["lesion_voxels"] = int((mask > 0).sum())
            meta["matched_planes"] = info["matched_planes"]
            meta["unmatched_planes"] = info["unmatched_planes"]
            meta["shape_zyx"] = list(mask.shape)
        except (PermissionError, OSError, ValueError, RuntimeError) as exc:
            meta["preview_error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001
            meta["preview_error"] = f"{type(exc).__name__}: {exc}"

    (case_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    slim = {k: v for k, v in meta.items() if k != "series"}
    slim["n_series_in_study"] = meta["n_series_in_study"]
    return slim


def main() -> int:
    args = parse_args()
    assert_outside(args.out, args.procancer_root)
    lesion_list = args.procancer_root / "subsets" / "studies_with_lesion"
    dicom_root = args.procancer_root / "dicom"
    if not lesion_list.is_dir():
        raise SystemExit(f"missing {lesion_list}")

    cases_root = args.out / "cases"
    previews_root = args.out / "previews"
    cases_root.mkdir(parents=True, exist_ok=True)

    items = list(iter_lesion_studies(lesion_list, dicom_root))
    if args.limit is not None:
        items = items[: args.limit]

    index: list[dict] = []
    for patient_id, study_uid, study_dir, _link in tqdm(items, desc="Lesion DICOM"):
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

    index.sort(key=lambda m: (not m["ready"], m.get("patient_id") or ""))
    summary = {
        "source_list": str(lesion_list),
        "dicom_root": str(dicom_root),
        "out": str(args.out),
        "n_studies": len(index),
        "n_ready": sum(1 for m in index if m["ready"]),
        "n_missing_dicom": sum(1 for m in index if not m.get("dicom_study_path")),
        "n_preview_ok": sum(1 for m in index if m.get("preview")),
        "n_preview_error": sum(1 for m in index if m.get("preview_error")),
        "n_has_t2_lesion": sum(
            1 for m in index if m.get("t2") and m.get("lesion_seg")
        ),
        "cases": index,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"wrote {args.out / 'index.json'}: {summary['n_ready']}/{summary['n_studies']} ready "
        f"(T2+ADC+DWI+lesion), {summary['n_preview_ok']} previews"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
