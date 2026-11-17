#!/usr/bin/env python3
"""Per-case QC of the v2 dataset: mask metrics + contour contact sheets.

Writes <out>/qc.json, <out>/qc.csv and <out>/sheet_XX.png (rows = patients,
columns = slices across the gland with the GT contour in red).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.bootstrap import sanitize_sys_path

sanitize_sys_path()

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from src.cascade.io_nifti import dataset_paths, load_patient_meta
from src.cascade.io_nifti import load_nifti, save_nifti
from src.cascade.v2 import crop_box, full_box, load_patient, preprocess, sample_box, write_json

THUMB = 160
COLS = 7


def metrics(p, meta, v1_voxels: int | None) -> dict:
    Z, H, W = p.image.shape
    sy, sx = p.spacing_yx
    pos = np.nonzero(p.mask.reshape(Z, -1).sum(1))[0]
    out = {
        "patient_id": p.pid,
        "labels": "|".join(meta.get("segment_labels") or []),
        "shape": f"{Z}x{H}x{W}",
        "spacing_mm": f"{sy:.2f}x{sx:.2f}x{p.slice_mm:.1f}",
        "fov_mm": round(H * sy),
        "n_mask_slices": int(len(pos)),
        "unmatched_planes": int(meta.get("unmatched_seg_planes", 0)),
    }
    if len(pos) == 0:
        return {**out, "volume_ml": 0.0}
    out["volume_ml"] = round(float(p.mask.sum()) * sy * sx * p.slice_mm / 1000, 1)
    out["v1_voxel_ratio"] = round(float(p.mask.sum()) / v1_voxels, 3) if v1_voxels else None
    out["z_gaps"] = int((pos[-1] - pos[0] + 1) - len(pos))
    out["z_extent_mm"] = round(len(pos) * p.slice_mm, 1)
    _, n3d = ndimage.label(p.mask)
    out["components_3d"] = int(n3d)
    out["max_components_2d"] = int(max(ndimage.label(p.mask[z])[1] for z in pos))
    yy, xx = np.nonzero(p.mask.any(0))
    out["extent_y_mm"] = round((yy.max() - yy.min() + 1) * sy, 1)
    out["extent_x_mm"] = round((xx.max() - xx.min() + 1) * sx, 1)
    gc = p.gt_center
    out["center_offset_mm"] = round(float(np.hypot((gc[0] - H / 2) * sy, (gc[1] - W / 2) * sx)), 1)
    out["touches_border"] = bool(p.mask[:, [0, -1], :].any() or p.mask[:, :, [0, -1]].any())
    # Gland intensity vs surrounding ring: mislabelled masks look like background.
    z = int(pos[len(pos) // 2])
    m = p.mask[z].astype(bool)
    ring = ndimage.binary_dilation(m, iterations=int(10 / sy)) & ~m
    img = p.image[z]
    out["mid_area_cm2"] = round(float(m.sum()) * sy * sx / 100, 1)
    out["in_vs_ring"] = round(float(img[m].mean() / max(img[ring].mean(), 1e-6)), 2)
    return out


def thumbnail(p, z: int, zoom_mm: float) -> Image.Image:
    H, W = p.image.shape[1:]
    gc = p.gt_center
    box = crop_box(gc, p.spacing_yx, zoom_mm) if gc and zoom_mm > 0 else full_box(H, W)
    x = (preprocess(p.image[z], box) * 255).astype(np.uint8)
    m = sample_box(p.mask[z].astype(np.float32), *box, nearest=True) > 0
    edge = m & ~ndimage.binary_erosion(m)
    rgb = np.stack([x] * 3, -1)
    rgb[edge] = (255, 40, 40)
    return Image.fromarray(rgb).resize((THUMB, THUMB), Image.BILINEAR)


def row_image(p, info: dict, zoom_mm: float) -> Image.Image:
    Z = p.image.shape[0]
    pos = np.nonzero(p.mask.reshape(Z, -1).sum(1))[0]
    if len(pos):
        zs = [max(pos[0] - 1, 0)] + list(np.linspace(pos[0], pos[-1], COLS - 2).round().astype(int))
        zs.append(min(pos[-1] + 1, Z - 1))
    else:
        zs = list(np.linspace(0, Z - 1, COLS).round().astype(int))
    row = Image.new("RGB", (THUMB * COLS, THUMB + 16), (20, 20, 20))
    for i, z in enumerate(zs):
        row.paste(thumbnail(p, int(z), zoom_mm), (i * THUMB, 16))
    ImageDraw.Draw(row).text(
        (4, 2),
        f"{info['patient_id'][:22]}  {info.get('volume_ml')} mL  {info['spacing_mm']}  "
        f"fov {info['fov_mm']}mm  gaps {info.get('z_gaps')}  cc3d {info.get('components_3d')}  "
        f"in/ring {info.get('in_vs_ring')}  [{info['labels'][:40]}]",
        fill=(255, 255, 0),
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "cascade_v2")
    ap.add_argument("--v1-data", type=Path, default=ROOT / "data" / "cascade")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "prostate_detect" / "cascade_v2" / "qc")
    ap.add_argument("--per-sheet", type=int, default=9)
    ap.add_argument("--zoom-mm", type=float, default=110.0, help="0 = full field of view")
    ap.add_argument("--keep-largest", action="store_true",
                    help="Rewrite masks keeping only the largest 3D connected component")
    ap.add_argument("--exclude", type=Path, default=None,
                    help="'<patient_id>  # reason' per line; written splits omit these")
    ap.add_argument("--v1-splits", type=Path, default=ROOT / "data" / "cascade" / "splits")
    args = ap.parse_args()

    pids = sorted(f.name[: -len(".json")] for f in (args.data / "meta").glob("*.json"))
    rows, images = [], []
    for pid in pids:
        if args.keep_largest:
            paths = dataset_paths(args.data, pid)
            mask, affine = load_nifti(paths["mask"])
            lab, n = ndimage.label(np.asarray(mask) > 0)
            if n > 1:
                sizes = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1))
                keep = (lab == 1 + int(np.argmax(sizes))).astype(np.uint8)
                save_nifti(paths["mask"], keep, affine, dtype=np.uint8)
                print(f"{pid}: kept largest of {n} components "
                      f"(removed {int((np.asarray(mask) > 0).sum() - keep.sum())} voxels)")
        p = load_patient(args.data, pid)
        meta = load_patient_meta(dataset_paths(args.data, pid)["meta"])
        v1_meta = dataset_paths(args.v1_data, pid)["meta"]
        v1_vox = load_patient_meta(v1_meta).get("mask_voxels") if v1_meta.exists() else None
        info = metrics(p, meta, v1_vox)
        rows.append(info)
        images.append(row_image(p, info, args.zoom_mm))
        print(info)

    if args.exclude:
        excluded = {
            line.split("#")[0].strip()
            for line in args.exclude.read_text().splitlines()
            if line.split("#")[0].strip()
        }
        split_dir = args.data / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        for name in ("train", "validation", "test"):
            ids = [
                s.strip()
                for s in (args.v1_splits / f"{name}.txt").read_text().splitlines()
                if s.strip() and s.strip() not in excluded and s.strip() in pids
            ]
            (split_dir / f"{name}.txt").write_text("\n".join(ids) + "\n")
            print(f"split {name}: {len(ids)} patients")

    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "qc.json", rows)
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with (args.out / "qc.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    for s in range(0, len(images), args.per_sheet):
        chunk = images[s : s + args.per_sheet]
        sheet = Image.new("RGB", (chunk[0].width, sum(r.height for r in chunk)))
        y = 0
        for r in chunk:
            sheet.paste(r, (0, y))
            y += r.height
        sheet.save(args.out / f"sheet_{s // args.per_sheet:02d}.png")
    print(f"QC → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
