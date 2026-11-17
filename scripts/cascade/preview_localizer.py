#!/usr/bin/env python3
"""Preview localizer ONNX on cascade holdout/test patients.

For each series, writes a panel PNG with a few slices, predicted crosshair,
optional GT centre, and a legend of inference outputs.
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
from matplotlib.patches import Circle
from PIL import Image

from src.cascade.decisions import CASCADE_DECISIONS
from src.cascade.io_nifti import dataset_paths, load_nifti
from src.cascade.labels import volume_slice_labels
from src.cascade.splits import read_split_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=ROOT / "data" / "cascade")
    p.add_argument(
        "--onnx",
        type=Path,
        default=ROOT
        / "outputs"
        / "prostate_detect"
        / "cascade"
        / "localizer-v1"
        / "prostate_localizer_128.onnx",
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Patient split (holdout is test.txt)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: alongside ONNX under previews/)",
    )
    p.add_argument("--n-slices", type=int, default=6, help="Slices shown per patient")
    p.add_argument("--presence-threshold", type=float, default=0.5)
    p.add_argument("--image-size", type=int, default=None)
    return p.parse_args()


def resize_slice(sl: np.ndarray, size: int) -> np.ndarray:
    u8 = np.clip(sl * 255.0, 0, 255).astype(np.uint8)
    out = Image.fromarray(u8, mode="L").resize((size, size), resample=Image.Resampling.BILINEAR)
    return np.asarray(out, dtype=np.float32) / 255.0


def run_localizer(
    session: ort.InferenceSession,
    volume: np.ndarray,
    image_size: int,
) -> list[dict]:
    """Infer presence + centre for every slice. centre_yx is native pixel coords."""
    rows, cols = volume.shape[1], volume.shape[2]
    results: list[dict] = []
    input_name = session.get_inputs()[0].name
    for z in range(volume.shape[0]):
        sl = volume[z].astype(np.float32)
        inp = resize_slice(sl, image_size)[None, None, ...].astype(np.float32)
        outs = session.run(None, {input_name: inp})
        # presence_logit, center_yx
        logit = float(np.asarray(outs[0]).reshape(-1)[0])
        cy_n, cx_n = [float(x) for x in np.asarray(outs[1]).reshape(-1)[:2]]
        presence = 1.0 / (1.0 + np.exp(-logit))
        results.append(
            {
                "z": z,
                "presence": presence,
                "logit": logit,
                "center_norm_yx": (cy_n, cx_n),
                "center_yx": (cy_n * max(rows - 1, 1), cx_n * max(cols - 1, 1)),
            }
        )
    return results


def pick_slice_indices(
    preds: list[dict],
    labels,
    n: int,
    threshold: float,
) -> list[int]:
    """Mix of GT apex/mid/base, top predicted presence, and a clear negative."""
    z_pos = [lab.z for lab in labels if lab.present]
    z_neg = [lab.z for lab in labels if not lab.present]
    chosen: list[int] = []

    if z_pos:
        apex, mid, base = z_pos[0], z_pos[len(z_pos) // 2], z_pos[-1]
        for z in (apex, mid, base):
            if z not in chosen:
                chosen.append(z)

    ranked = sorted(preds, key=lambda r: r["presence"], reverse=True)
    for r in ranked:
        if r["z"] not in chosen:
            chosen.append(r["z"])
        if len(chosen) >= max(n - 1, 1):
            break

    # Prefer a true negative with low predicted presence.
    neg_ranked = sorted(
        (r for r in preds if r["z"] in z_neg),
        key=lambda r: r["presence"],
    )
    if neg_ranked:
        z = neg_ranked[0]["z"]
        if z not in chosen:
            chosen.append(z)

    while len(chosen) < n and len(chosen) < len(preds):
        for r in preds:
            if r["z"] not in chosen:
                chosen.append(r["z"])
                break
        else:
            break

    return sorted(chosen[:n])


def draw_crosshair(ax, y: float, x: float, color: str, label: str | None = None) -> None:
    h, w = ax.get_images()[0].get_array().shape[:2]
    arm = max(8, min(h, w) * 0.04)
    ax.plot([x - arm, x + arm], [y, y], color=color, lw=1.6, solid_capstyle="butt")
    ax.plot([x, x], [y - arm, y + arm], color=color, lw=1.6, solid_capstyle="butt")
    ax.add_patch(Circle((x, y), radius=arm * 0.35, fill=False, ec=color, lw=1.2))
    if label:
        ax.plot([], [], color=color, lw=2, label=label)


def render_patient(
    patient_id: str,
    volume: np.ndarray,
    mask: np.ndarray,
    preds: list[dict],
    labels,
    out_path: Path,
    *,
    n_slices: int,
    threshold: float,
) -> dict:
    idxs = pick_slice_indices(preds, labels, n_slices, threshold)
    label_by_z = {lab.z: lab for lab in labels}
    pred_by_z = {r["z"]: r for r in preds}

    n = len(idxs)
    fig_w = max(3.2 * n, 8)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 4.6), squeeze=False)
    axes = axes[0]

    for ax, z in zip(axes, idxs):
        sl = volume[z]
        ax.imshow(sl, cmap="gray", vmin=0, vmax=1)
        mk = mask[z] > 0
        if mk.any():
            ax.contour(mk.astype(np.float32), levels=[0.5], colors=["#3dd68c"], linewidths=0.9)

        pred = pred_by_z[z]
        lab = label_by_z[z]
        present = pred["presence"] >= threshold
        cy, cx = pred["center_yx"]
        if present:
            draw_crosshair(ax, cy, cx, color="#ff4d4d", label="pred centre")
        else:
            # Dim marker so negatives are still inspectable.
            draw_crosshair(ax, cy, cx, color="#ff9f1a")

        if lab.present and lab.center_yx is not None:
            gy, gx = lab.center_yx
            ax.plot(gx, gy, marker="+", markersize=10, markeredgewidth=1.8, color="#3db8ff")

        title = (
            f"z={z}\n"
            f"p={pred['presence']:.2f} {'POS' if present else 'neg'}\n"
            f"pred=({cy:.0f},{cx:.0f})"
        )
        if lab.present and lab.center_yx is not None:
            title += f"\nGT=({lab.center_yx[0]:.0f},{lab.center_yx[1]:.0f})"
        else:
            title += "\nGT=absent"
        ax.set_title(title, fontsize=9, family="monospace")
        ax.set_xticks([])
        ax.set_yticks([])

    # Shared legend under the panel.
    n_pos_pred = sum(1 for r in preds if r["presence"] >= threshold)
    n_pos_gt = sum(1 for lab in labels if lab.present)
    best = max(preds, key=lambda r: r["presence"])
    legend = (
        f"{patient_id}\n"
        f"red crosshair = predicted centre (shown when p≥{threshold:.2f}; orange if below)\n"
        f"cyan + = GT centre   green contour = GT mask\n"
        f"pred-positive slices: {n_pos_pred}/{len(preds)}   "
        f"GT-positive slices: {n_pos_gt}/{len(labels)}   "
        f"peak p={best['presence']:.3f} @ z={best['z']}"
    )
    fig.suptitle(legend, fontsize=10, x=0.01, ha="left", family="monospace")
    fig.tight_layout(rect=[0, 0, 1, 0.78])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "patient_id": patient_id,
        "n_slices_shown": n,
        "n_pred_positive": n_pos_pred,
        "n_gt_positive": n_pos_gt,
        "peak_presence": best["presence"],
        "peak_z": best["z"],
        "preview": str(out_path),
        "slice_indices": idxs,
    }


def write_index_html(out_dir: Path, rows: list[dict]) -> Path:
    cards = []
    for r in rows:
        rel = Path(r["preview"]).name
        cards.append(
            f"""
            <section class="card">
              <h2>{r["patient_id"]}</h2>
              <p class="meta">
                pred+ {r["n_pred_positive"]} &nbsp;|&nbsp;
                GT+ {r["n_gt_positive"]} &nbsp;|&nbsp;
                peak p={r["peak_presence"]:.3f} @ z={r["peak_z"]}
              </p>
              <img src="{rel}" alt="{r["patient_id"]}" />
            </section>
            """
        )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Localizer holdout preview</title>
<style>
  body {{ margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
         background: #0f1419; color: #e7eef7; }}
  h1 {{ padding: 1.2rem 1.4rem 0.4rem; font-size: 1.15rem; font-weight: 650; }}
  .sub {{ padding: 0 1.4rem 1rem; color: #8b9bb0; font-size: 0.85rem; }}
  .card {{ padding: 0.6rem 1.4rem 1.6rem; border-top: 1px solid #243040; }}
  .card h2 {{ font-size: 0.95rem; margin: 0 0 0.35rem; }}
  .meta {{ color: #8b9bb0; font-size: 0.8rem; margin: 0 0 0.6rem; }}
  img {{ max-width: 100%; height: auto; background: #000; border-radius: 4px; }}
</style>
</head>
<body>
  <h1>Cascade localizer — holdout preview</h1>
  <p class="sub">Red crosshair = predicted centre &nbsp;·&nbsp; Cyan + = GT centre &nbsp;·&nbsp; Green = GT mask</p>
  {"".join(cards)}
</body>
</html>
"""
    path = out_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    if not args.onnx.is_file():
        print(f"Missing ONNX: {args.onnx}")
        return 1

    split_name = {
        "train": "train.txt",
        "validation": "validation.txt",
        "test": "test.txt",
    }[args.split]
    patients = read_split_file(args.dataset / "splits" / split_name)
    if not patients:
        print(f"No patients in {args.dataset / 'splits' / split_name}")
        return 1

    out_dir = args.output or (args.onnx.parent / "previews" / args.split)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_size = args.image_size or CASCADE_DECISIONS.localizer_input_size

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    print(f"ONNX={args.onnx}  patients={len(patients)}  size={image_size}  → {out_dir}")

    summaries: list[dict] = []
    for pid in patients:
        paths = dataset_paths(args.dataset, pid)
        volume, _ = load_nifti(paths["image"])
        mask, _ = load_nifti(paths["mask"])
        labels = volume_slice_labels(mask)
        preds = run_localizer(session, volume, image_size)
        png = out_dir / f"{pid}.png"
        summary = render_patient(
            pid,
            volume,
            mask,
            preds,
            labels,
            png,
            n_slices=args.n_slices,
            threshold=args.presence_threshold,
        )
        # Per-patient JSON with all slice predictions for debugging.
        (out_dir / f"{pid}.json").write_text(
            json.dumps(
                {
                    **summary,
                    "slices": [
                        {
                            **{k: v for k, v in r.items() if k != "center_norm_yx"},
                            "center_norm_yx": list(r["center_norm_yx"]),
                            "center_yx": list(r["center_yx"]),
                            "gt_present": bool(labels[r["z"]].present),
                            "gt_center_yx": (
                                list(labels[r["z"]].center_yx)
                                if labels[r["z"]].center_yx
                                else None
                            ),
                        }
                        for r in preds
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        summaries.append(summary)
        print(
            f"  {pid[:28]}…  pred+={summary['n_pred_positive']}  "
            f"GT+={summary['n_gt_positive']}  peak={summary['peak_presence']:.3f}"
        )

    index = write_index_html(out_dir, summaries)
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"Index: {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
