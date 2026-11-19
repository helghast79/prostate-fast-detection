#!/usr/bin/env python3
"""Build a lesion training catalog from ProCAncer-I (read-only).

Scans subsets/studies_with_lesion (symlink names = StudyInstanceUID) and resolves
each study under dicom/<PatientID>/<StudyInstanceUID>. The lesion list itself
points at retrospective/nifti_new volumes that already contain:

  image_T2.mha, image_ADC.mha, image_DWI.mha, mask_lesion.mha

This script NEVER writes inside the ProCAncer-I tree. It only creates, under
--out (default data/procancer_lesion):

  index.json
  cases/<patient_id>/<study_uid>/   # symlinks to the .mha files (+ meta.json)
  previews/<patient_id>__<study_uid>.png   # optional mid-slice T2+lesion overlay

Incomplete studies (missing a modality in nifti) are still indexed and keep a
dicom_study_path so they can be recovered later without copying.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.bootstrap import sanitize_sys_path

sanitize_sys_path()

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.procancer_mha import lesion_volume_ml, read_mha

NEEDED = ("image_T2.mha", "image_ADC.mha", "image_DWI.mha", "mask_lesion.mha")
OPTIONAL = ("mask_PZ.mha", "mask_TZ+CZ.mha", "mask_SV.mha", "predicted_prostate_mask.mha")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--procancer-root",
        type=Path,
        default=Path("/usr/backups/datasets/ProCAncer-I"),
    )
    p.add_argument("--out", type=Path, default=ROOT / "data" / "procancer_lesion")
    p.add_argument("--previews", action="store_true", help="Write mid-slice PNG overlays")
    p.add_argument("--force", action="store_true", help="Recreate symlinks / meta even if present")
    return p.parse_args()


def resolve_symlink(link: Path) -> tuple[str, str, Path]:
    target = Path(os.readlink(link))
    if not target.is_absolute():
        target = (link.parent / target).resolve()
    parts = target.parts
    try:
        i = next(i for i, x in enumerate(parts) if x.startswith("PCa-"))
    except StopIteration as exc:
        raise ValueError(f"no PCa- patient id in {target}") from exc
    return parts[i], parts[i + 1], target


def assert_outside_procancer(out: Path, procancer: Path) -> None:
    out_r, pro_r = out.resolve(), procancer.resolve()
    if out_r == pro_r or pro_r in out_r.parents:
        raise SystemExit(f"refusing to write inside ProCAncer-I: {out_r}")


def safe_symlink(src: Path, dst: Path, force: bool) -> None:
    if dst.is_symlink() or dst.exists():
        if not force:
            return
        dst.unlink()
    dst.symlink_to(src.resolve())


def overlay_preview(t2: np.ndarray, mask: np.ndarray, out_png: Path) -> None:
    z = int(np.argmax(mask.reshape(len(mask), -1).sum(1))) if mask.any() else len(mask) // 2
    sl = t2[z].astype(np.float32)
    lo, hi = np.percentile(sl, (1, 99))
    if hi <= lo:
        hi = lo + 1
    gray = np.clip((sl - lo) / (hi - lo), 0, 1)
    rgb = np.stack([gray, gray, gray], axis=-1)
    m = mask[z] > 0
    rgb[m, 0] = np.maximum(rgb[m, 0], 0.85)
    rgb[m, 1] *= 0.35
    rgb[m, 2] *= 0.35
    Image.fromarray((rgb * 255).astype(np.uint8)).save(out_png)


def build_case(
    patient_id: str,
    study_uid: str,
    nifti_dir: Path,
    dicom_dir: Path,
    case_dir: Path,
    preview_path: Path | None,
    *,
    force: bool,
    make_preview: bool,
) -> dict:
    case_dir.mkdir(parents=True, exist_ok=True)
    present = {name: (nifti_dir / name).is_file() for name in (*NEEDED, *OPTIONAL)}
    for name, ok in present.items():
        if ok:
            safe_symlink(nifti_dir / name, case_dir / name, force)

    complete = all(present[n] for n in NEEDED)
    lesion_ml = lesion_voxels = lesion_slices = None
    shape = spacing = None
    if present["mask_lesion.mha"]:
        mask, spacing, _ = read_mha(nifti_dir / "mask_lesion.mha")
        lesion_voxels = int((mask > 0).sum())
        lesion_slices = int((mask.reshape(len(mask), -1).sum(1) > 0).sum())
        lesion_ml = round(lesion_volume_ml(mask, spacing), 3)
        shape = list(mask.shape)
        if make_preview and present["image_T2.mha"] and preview_path is not None:
            t2, _, _ = read_mha(nifti_dir / "image_T2.mha")
            # resample-free overlay requires same shape; skip if mismatch
            if t2.shape == mask.shape:
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                if force or not preview_path.is_file():
                    overlay_preview(t2, mask, preview_path)

    meta = {
        "patient_id": patient_id,
        "study_instance_uid": study_uid,
        "complete": complete,
        "present": present,
        "missing": [n for n in NEEDED if not present[n]],
        "nifti_dir": str(nifti_dir),
        "dicom_study_path": str(dicom_dir) if dicom_dir.is_dir() else None,
        "case_dir": str(case_dir),
        "preview": str(preview_path) if preview_path and preview_path.is_file() else None,
        "shape_zyx": shape,
        "spacing_xyz_mm": list(spacing) if spacing else None,
        "lesion_voxels": lesion_voxels,
        "lesion_slices": lesion_slices,
        "lesion_ml": lesion_ml,
        "discarded": False,
    }
    (case_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> int:
    args = parse_args()
    assert_outside_procancer(args.out, args.procancer_root)
    lesion_list = args.procancer_root / "subsets" / "studies_with_lesion"
    dicom_root = args.procancer_root / "dicom"
    if not lesion_list.is_dir():
        raise SystemExit(f"missing {lesion_list}")

    cases_root = args.out / "cases"
    previews_root = args.out / "previews"
    cases_root.mkdir(parents=True, exist_ok=True)

    links = sorted(p for p in lesion_list.iterdir() if p.is_symlink())
    index: list[dict] = []
    for link in tqdm(links, desc="Catalog"):
        patient_id, study_uid, nifti_dir = resolve_symlink(link)
        case_dir = cases_root / patient_id / study_uid
        preview = previews_root / f"{patient_id}__{study_uid}.png"
        meta = build_case(
            patient_id,
            study_uid,
            nifti_dir,
            dicom_root / patient_id / study_uid,
            case_dir,
            preview if args.previews else None,
            force=args.force,
            make_preview=args.previews,
        )
        index.append(meta)

    index.sort(key=lambda m: (not m["complete"], -(m["lesion_ml"] or 0), m["patient_id"]))
    summary = {
        "source": str(lesion_list),
        "procancer_root": str(args.procancer_root),
        "out": str(args.out),
        "n_studies": len(index),
        "n_complete": sum(1 for m in index if m["complete"]),
        "n_incomplete": sum(1 for m in index if not m["complete"]),
        "n_empty_lesion": sum(1 for m in index if (m["lesion_voxels"] or 0) == 0),
        "cases": index,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"wrote {args.out / 'index.json'}: "
        f"{summary['n_complete']} complete / {summary['n_studies']} studies "
        f"({summary['n_incomplete']} incomplete, {summary['n_empty_lesion']} empty)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
