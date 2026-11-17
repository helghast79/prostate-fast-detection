#!/usr/bin/env python3
"""Reference end-to-end cascade inference (localizer → robust band → ENet).

Use this as the ground-truth behaviour the frontend should match.
Writes overlay PNGs + a JSON summary for one or more patients.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from PIL import Image

from src.cascade.decisions import CASCADE_DECISIONS
from src.cascade.frontend_policy import SliceScore, plan_series
from src.cascade.io_nifti import dataset_paths, load_nifti, load_patient_meta
from src.cascade.labels import volume_slice_labels
from src.cascade.roi import compute_square_roi, extract_roi, paste_roi_into_full
from src.cascade.splits import read_split_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=ROOT / "data" / "cascade")
    p.add_argument(
        "--localizer",
        type=Path,
        default=ROOT
        / "outputs/prostate_detect/cascade/localizer-v1/prostate_localizer_128.onnx",
    )
    p.add_argument(
        "--segmenter",
        type=Path,
        default=ROOT
        / "outputs/prostate_detect/cascade/seg-enet-v2/prostate_seg_enet_160.onnx",
    )
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--patient-id", type=str, default=None)
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/prostate_detect/cascade/cascade_infer_previews",
    )
    p.add_argument("--presence-thr-naive", type=float, default=0.5)
    p.add_argument("--seg-thr", type=float, default=0.5)
    p.add_argument("--n-preview", type=int, default=8)
    return p.parse_args()


def resize_localizer(sl: np.ndarray, size: int) -> np.ndarray:
    """Match training: float[0,1] → uint8 → bilinear → float[0,1]."""
    u8 = np.clip(sl * 255.0, 0, 255).astype(np.uint8)
    out = Image.fromarray(u8, mode="L").resize((size, size), resample=Image.Resampling.BILINEAR)
    return np.asarray(out, dtype=np.float32) / 255.0


def run_localizer_volume(
    session: ort.InferenceSession, volume: np.ndarray, size: int
) -> list[SliceScore]:
    rows, cols = volume.shape[1], volume.shape[2]
    name = session.get_inputs()[0].name
    out: list[SliceScore] = []
    for z in range(volume.shape[0]):
        inp = resize_localizer(volume[z], size)[None, None, ...].astype(np.float32)
        outs = session.run(None, {name: inp})
        logit = float(np.asarray(outs[0]).reshape(-1)[0])
        cy_n, cx_n = [float(x) for x in np.asarray(outs[1]).reshape(-1)[:2]]
        presence = 1.0 / (1.0 + np.exp(-logit))
        out.append(
            SliceScore(
                z=z,
                presence=presence,
                center_yx=(cy_n * max(rows - 1, 1), cx_n * max(cols - 1, 1)),
            )
        )
    return out


def run_segmenter_slice(
    session: ort.InferenceSession,
    sl: np.ndarray,
    center_yx: tuple[float, float],
    crop_size: int,
    seg_thr: float,
) -> tuple[np.ndarray, dict]:
    rows, cols = sl.shape
    crop = compute_square_roi(center_yx, rows, cols, size=crop_size)
    roi = extract_roi(sl, crop)
    name = session.get_inputs()[0].name
    logits = session.run(
        None, {name: roi[None, None, ...].astype(np.float32)}
    )[0]
    prob = 1.0 / (1.0 + np.exp(-logits[0, 0]))
    roi_mask = (prob > seg_thr).astype(np.uint8)
    full = paste_roi_into_full((rows, cols), crop, roi_mask)
    return full, crop.to_dict()


def dice(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = float((a & b).sum())
    return float((2 * inter + eps) / (a.sum() + b.sum() + eps))


def main() -> int:
    args = parse_args()
    split_file = {
        "train": "train.txt",
        "validation": "validation.txt",
        "test": "test.txt",
    }[args.split]
    patients = (
        [args.patient_id]
        if args.patient_id
        else read_split_file(args.dataset / "splits" / split_file)
    )
    args.output.mkdir(parents=True, exist_ok=True)

    loc = ort.InferenceSession(str(args.localizer), providers=["CPUExecutionProvider"])
    seg = ort.InferenceSession(str(args.segmenter), providers=["CPUExecutionProvider"])
    loc_size = CASCADE_DECISIONS.localizer_input_size
    crop_size = CASCADE_DECISIONS.roi_crop_size

    summaries = []
    for pid in patients:
        paths = dataset_paths(args.dataset, pid)
        volume, _ = load_nifti(paths["image"])
        mask, _ = load_nifti(paths["mask"])
        meta = load_patient_meta(paths["meta"]) if paths["meta"].is_file() else {}
        # Approximate slice spacing from affine column 2 length if present.
        affine = np.asarray(meta.get("affine") or np.eye(4), dtype=np.float64)
        spacing_z = float(np.linalg.norm(affine[:3, 2])) or 3.0

        scores = run_localizer_volume(loc, volume, loc_size)
        naive_z = [s.z for s in scores if s.presence >= args.presence_thr_naive]
        plan = plan_series(
            scores,
            rows=volume.shape[1],
            cols=volume.shape[2],
            slice_spacing_mm=spacing_z,
        )
        labels = volume_slice_labels(mask)
        gt_pos = {lab.z for lab in labels if lab.present}

        pred_vol = np.zeros_like(mask, dtype=np.uint8)
        if plan is not None:
            for z in plan.active_z:
                full, _ = run_segmenter_slice(
                    seg,
                    volume[z],
                    plan.centers_yx[z],
                    crop_size,
                    args.seg_thr,
                )
                pred_vol[z] = full

        # Dice on GT-positive slices only + volume dice.
        slice_dices = []
        for lab in labels:
            if not lab.present:
                continue
            slice_dices.append(dice(pred_vol[lab.z], mask[lab.z] > 0))
        vol_dice = dice(pred_vol, mask > 0)

        summary = {
            "patient_id": pid,
            "n_slices": int(volume.shape[0]),
            "naive_pred_positive": len(naive_z),
            "robust_active": 0 if plan is None else len(plan.active_z),
            "robust_band": None if plan is None else list(plan.band),
            "peak_z": None if plan is None else plan.peak_z,
            "peak_presence": None if plan is None else plan.peak_presence,
            "plan": None if plan is None else plan.reason,
            "gt_positive": len(gt_pos),
            "mean_dice_gt_slices": float(np.mean(slice_dices)) if slice_dices else 0.0,
            "volume_dice": vol_dice,
            "scores": [
                {
                    "z": s.z,
                    "presence": s.presence,
                    "center_yx": list(s.center_yx),
                    "gt_present": s.z in gt_pos,
                    "active_robust": bool(plan and s.z in plan.active_z),
                    "active_naive": s.z in naive_z,
                }
                for s in scores
            ],
        }
        summaries.append(summary)
        (args.output / f"{pid}.json").write_text(json.dumps(summary, indent=2) + "\n")

        # Preview panel: peak ± neighbors.
        if plan is None:
            print(f"{pid}: no prostate band (peak too low)")
            continue
        show = sorted(
            set(
                [plan.peak_z]
                + plan.active_z[:: max(1, len(plan.active_z) // args.n_preview)]
            )
        )[: args.n_preview]
        fig, axes = plt.subplots(1, len(show), figsize=(3.2 * len(show), 3.6), squeeze=False)
        for ax, z in zip(axes[0], show):
            ax.imshow(volume[z], cmap="gray", vmin=0, vmax=1)
            if (mask[z] > 0).any():
                ax.contour(mask[z] > 0, levels=[0.5], colors=["#3dd68c"], linewidths=0.8)
            if pred_vol[z].any():
                ax.contour(pred_vol[z] > 0, levels=[0.5], colors=["#ff4d4d"], linewidths=0.9)
            cy, cx = plan.centers_yx[z]
            ax.plot(cx, cy, "r+", markersize=10, markeredgewidth=1.5)
            sc = next(s for s in scores if s.z == z)
            ax.set_title(
                f"z={z} p={sc.presence:.2f}\n"
                f"{'GT+' if z in gt_pos else 'GT-'}  "
                f"dice={dice(pred_vol[z], mask[z]>0):.2f}",
                fontsize=8,
                family="monospace",
            )
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(
            f"{pid}\nnaive≥{args.presence_thr_naive}: {len(naive_z)} slices  |  "
            f"robust band: {plan.band[0]}–{plan.band[1]} ({len(plan.active_z)})  |  "
            f"vol dice={vol_dice:.3f}",
            fontsize=9,
            family="monospace",
            ha="left",
            x=0.01,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.78])
        fig.savefig(args.output / f"{pid}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(
            f"{pid}: naive+={len(naive_z)} robust+={len(plan.active_z)} "
            f"GT+={len(gt_pos)} vol_dice={vol_dice:.3f}  ({plan.reason})"
        )

    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
