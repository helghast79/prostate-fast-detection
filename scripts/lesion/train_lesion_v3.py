#!/usr/bin/env python3
"""Train lesion_detection-v3: ENet, 3ch (T2/ADC/DWI) at 192², CROP_MM box.

Crop centre priority:
  1. Prostate gland mask centroid from zone SEG (saved in lesion_v3 npz)
  2. Else localizer-v3 series centre on T2

Same physical crop as cascade v3 (default 110 mm). Trains on every T2 slice.
Eval reports raw Dice / detection and the same metrics after restricting
predictions to the prostate mask (SEG if present, else prostate_detection-v3).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.bootstrap import sanitize_sys_path

sanitize_sys_path()

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage

from src.cascade.enet import ENetBinary
from src.cascade.lesion import (
    lesion_input,
    lesion_metrics,
    load_case,
    prostate_center_yx,
    restrict_to_prostate,
)
from src.cascade.v2 import (
    SIZE,
    LocalizerHead,
    SegmenterHead,
    crop_box,
    fit,
    full_box,
    keep_largest_slice_run,
    paste_back,
    preprocess,
    sample_box,
    series_center,
    write_json,
)
from src.cascade.v3 import CROP_MM_DEFAULT, DetectionHead
from src.model import TinyUNet
from src.train_monitor import MetricsStore, start_dashboard

OUT = ROOT / "outputs" / "prostate_detect"
LESION_AUG = {"scale": 0.10, "shift": 0.05, "rot_deg": 10.0}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "lesion_v3")
    p.add_argument("--out", type=Path, default=OUT / "lesion" / "lesion_detection-v3")
    p.add_argument(
        "--loc",
        type=Path,
        default=OUT / "cascade_v3" / "localizer-v3" / "best.pt",
    )
    p.add_argument(
        "--det",
        type=Path,
        default=OUT / "cascade_v3" / "prostate_detection-v3" / "best.pt",
        help="fallback prostate mask when case has no zone SEG",
    )
    p.add_argument("--crop-mm", type=float, default=CROP_MM_DEFAULT)
    p.add_argument("--epochs", type=int, default=75)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pos-repeat", type=int, default=4)
    p.add_argument("--qc-only", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--dashboard-port", type=int, default=8769)
    p.add_argument("--no-dashboard", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def read_split(data: Path) -> dict[str, list[str]]:
    out = {}
    for name in ("train", "validation", "test"):
        path = data / "splits" / f"{name}.txt"
        out[name] = [s.strip() for s in path.read_text().splitlines() if s.strip()] if path.is_file() else []
    return out


@torch.no_grad()
def localizer_center(t2: np.ndarray, loc_head: LocalizerHead, device: torch.device) -> tuple[float, float]:
    Z, H, W = t2.shape
    x = np.stack([preprocess(t2[z], full_box(H, W)) for z in range(Z)])[:, None]
    _, mass, center, _ = loc_head(torch.from_numpy(x).to(device))
    return series_center(mass.cpu().numpy(), center.cpu().numpy(), H, W) or (H / 2, W / 2)


@torch.no_grad()
def predict_prostate(
    t2: np.ndarray,
    box: tuple[float, float, float, float],
    det_head: DetectionHead,
    device: torch.device,
) -> np.ndarray:
    """Binary prostate from 3-class detection head (gland ∪ PZ)."""
    Z, H, W = t2.shape
    xs = np.stack([preprocess(t2[z], box) for z in range(Z)])[:, None]
    prob = det_head(torch.from_numpy(xs).to(device)).cpu().numpy()  # [Z,3,S,S]
    cls = prob.argmax(1)
    binary = (cls > 0).astype(np.float32)
    return keep_largest_slice_run(
        np.stack([paste_back(binary[z], box, H, W) for z in range(Z)])
    )


def resolve_crop_and_prostate(case, loc_head, det_head, crop_mm, device):
    t2 = case.t2.pixels
    spacing = (float(case.t2.spacing[0]), float(case.t2.spacing[1]))
    if case.prostate is not None and case.prostate.any():
        center = prostate_center_yx(case.prostate)
        source = "zone_seg"
        prostate = case.prostate.astype(np.uint8)
    else:
        center = localizer_center(t2, loc_head, device)
        source = "localizer-v3"
        prostate = None
    box = crop_box(center, spacing, crop_mm)
    if prostate is None:
        prostate = predict_prostate(t2, box, det_head, device)
        source = f"{source}+detection-v3"
    return box, prostate, source, center


def summarize(rows: list[dict]) -> dict:
    return {
        "mean_dice3d": round(float(np.mean([r["dice3d"] for r in rows])), 4) if rows else None,
        "lesions_detected": f"{sum(r['detected'] for r in rows)}/{sum(r['gt_lesions'] for r in rows)}",
        "fp_components_per_patient": round(
            float(np.mean([r["false_positive_components"] for r in rows])), 2
        )
        if rows
        else None,
    }


def qc_sheet(cases, inputs, crop_sources, path: Path) -> None:
    rows = []
    for pid, c in cases.items():
        x, m = inputs[pid]
        zs = np.nonzero(m.reshape(len(m), -1).sum(1))[0]
        z = int(zs[len(zs) // 2]) if len(zs) else len(m) // 2
        tiles = []
        edge = (m[z, 0] > 0) & ~ndimage.binary_erosion(m[z, 0] > 0)
        for ch in range(3):
            g = (x[z, ch] * 255).astype(np.uint8)
            rgb = np.stack([g] * 3, -1)
            rgb[edge] = (255, 40, 200)
            tiles.append(Image.fromarray(rgb))
        row = Image.new("RGB", (SIZE * 3, SIZE + 14))
        for i, t in enumerate(tiles):
            row.paste(t, (i * SIZE, 14))
        ImageDraw.Draw(row).text(
            (3, 1),
            f"{pid[:22]} {crop_sources[pid]} {c.meta.get('lesion_ml')}mL",
            fill=(255, 255, 0),
        )
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    for s in range(0, len(rows), 8):
        chunk = rows[s : s + 8]
        sheet = Image.new("RGB", (SIZE * 3, sum(r.height for r in chunk)))
        for i, r in enumerate(chunk):
            sheet.paste(r, (0, i * r.height))
        sheet.save(path.parent / f"{path.stem}_{s // 8}.png")


def main() -> int:
    args = parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    split = read_split(args.data)
    pids = sorted({p for ids in split.values() for p in ids})
    if not pids:
        pids = sorted(f.stem for f in (args.data / "cases").glob("*.npz"))
    cases = {pid: load_case(args.data, pid) for pid in pids}

    lck = torch.load(args.loc, map_location=device, weights_only=False)
    loc_net = TinyUNet(1, 1, base=lck.get("base", 8)).to(device)
    loc_net.load_state_dict(lck["state_dict"])
    loc_head = LocalizerHead(loc_net).to(device).eval()

    dck = torch.load(args.det, map_location=device, weights_only=False)
    det_net = ENetBinary(in_channels=1, out_channels=int(dck.get("num_classes", 3))).to(device)
    det_net.load_state_dict(dck["state_dict"])
    det_head = DetectionHead(det_net).to(device).eval()
    crop_mm = float(args.crop_mm)

    inputs, boxes, prostates, crop_sources = {}, {}, {}, {}
    for pid, c in cases.items():
        box, prostate, source, center = resolve_crop_and_prostate(
            c, loc_head, det_head, crop_mm, device
        )
        Z = c.t2.pixels.shape[0]
        x = np.stack([lesion_input(c.t2, c.adc, c.dwi, z, box) for z in range(Z)]).astype(np.float32)
        m = np.stack(
            [sample_box(c.mask[z].astype(np.float32), *box, nearest=True) for z in range(Z)]
        )[:, None]
        inputs[pid], boxes[pid], prostates[pid], crop_sources[pid] = (x, m), box, prostate, source
        y0, x0, h, w = box
        H, W = c.mask.shape[1:]
        inside = (
            ((np.arange(H) + 0.5)[:, None] >= y0)
            & ((np.arange(H) + 0.5)[:, None] < y0 + h)
            & ((np.arange(W) + 0.5)[None, :] >= x0)
            & ((np.arange(W) + 0.5)[None, :] < x0 + w)
        )
        cov = float((c.mask & inside[None]).sum() / max(c.mask.sum(), 1))
        print(
            f"{pid[:24]} src={source} slices={Z} lesion_slices={(m.reshape(Z, -1).sum(1) > 0).sum()} "
            f"lesion_in_crop={cov:.2f} centre=({center[0]:.1f},{center[1]:.1f}) "
            f"adc_nz={(x[:, 1] > 0).mean():.2f} dwi_nz={(x[:, 2] > 0).mean():.2f}"
        )

    qc_sheet(cases, inputs, crop_sources, args.out / "qc_channels.png")
    write_json(args.out / "split.json", split)
    write_json(args.out / "crop_sources.json", crop_sources)
    print({k: len(v) for k, v in split.items()}, "crop", dict(zip(*np.unique(list(crop_sources.values()), return_counts=True))))
    if args.qc_only:
        return 0

    def stack(ids):
        return tuple(torch.from_numpy(np.concatenate([inputs[i][k] for i in ids])).to(device) for k in (0, 1))

    train, val = stack(split["train"]), stack(split["validation"])
    if args.pos_repeat > 1:
        pos = train[1].sum((1, 2, 3)) > 0
        train = tuple(torch.cat([t] + [t[pos]] * (args.pos_repeat - 1)) for t in train)
    print(
        f"slices train {train[0].shape[0]} (lesion {(train[1].sum((1, 2, 3)) > 0).sum().item()}) "
        f"val {val[0].shape[0]}"
    )

    store, history = MetricsStore(), []
    if not args.no_dashboard:
        start_dashboard("0.0.0.0", args.dashboard_port, store)
        print(f"Dashboard: http://0.0.0.0:{args.dashboard_port}/")
    store.update(
        status="running",
        run_name=args.out.name,
        max_epochs=args.epochs,
        patience=args.patience,
        description=f"ENet 3ch lesion @{SIZE}, crop {crop_mm:.0f}mm (zone SEG centre → localizer-v3)",
    )

    def on_epoch(row):
        history.append(row)
        store.update(
            epoch=row["epoch"],
            latest=row,
            history=history,
            best_epoch=row["best_epoch"],
            best_val_loss=row["best_val_loss"],
            epochs_without_improve=row["stale"],
        )

    torch.manual_seed(args.seed)
    net = ENetBinary(in_channels=3, out_channels=1).to(device)
    t0 = time.time()
    if args.eval_only:
        ck = torch.load(args.out / "best.pt", map_location=device, weights_only=False)
        net.load_state_dict(ck["state_dict"])
        best = ck["best"]
    else:
        net, best = fit(
            net,
            train,
            val,
            aug=LESION_AUG,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            on_epoch=on_epoch,
        )
        torch.save(
            {
                "state_dict": net.state_dict(),
                "best": best,
                "crop_mm": crop_mm,
                "size": SIZE,
                "in_channels": 3,
                "channels": ["t2", "adc", "dwi"],
                "pos_repeat": args.pos_repeat,
                "crop_priority": ["zone_seg_centroid", "localizer-v3"],
            },
            args.out / "best.pt",
        )

    head = SegmenterHead(net).eval()
    results = {}
    with torch.no_grad():
        for split_name in ("validation", "test"):
            rows = []
            for pid in split[split_name]:
                c = cases[pid]
                x, _ = inputs[pid]
                prob = head(torch.from_numpy(x).to(device)).cpu().numpy()[:, 0]
                Z, H, W = c.mask.shape
                pred = np.stack([paste_back(prob[z], boxes[pid], H, W) for z in range(Z)])
                in_prostate = restrict_to_prostate(pred, prostates[pid])
                r = {
                    "patient_id": pid,
                    "label_source": c.meta.get("label_source"),
                    "crop_source": crop_sources[pid],
                    "gt_ml": c.meta.get("lesion_ml"),
                    **lesion_metrics(pred, c.mask),
                    "in_prostate": lesion_metrics(in_prostate, c.mask),
                }
                rows.append(r)
                print(split_name, r)
            results[split_name] = {
                "raw": summarize(rows),
                "in_prostate": summarize([r["in_prostate"] for r in rows]),
                "patients": rows,
            }
    stats = {
        "best": best,
        "crop_mm": crop_mm,
        "split": split,
        "crop_sources": crop_sources,
        "train_seconds": round(time.time() - t0),
        "results": results,
    }
    write_json(args.out / "stats.json", stats)
    store.update(status="done", stats={k: v["raw"] for k, v in results.items()})
    print(json.dumps({k: {"raw": v["raw"], "in_prostate": v["in_prostate"]} for k, v in results.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
