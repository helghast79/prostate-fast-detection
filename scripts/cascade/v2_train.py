#!/usr/bin/env python3
"""Train cascade v2 stages (both 192×192, same preprocessing).

  --stage localizer : TinyUNet coarse mask on the full slice.
                      K-fold out-of-fold (OOF) predictions for train patients,
                      final model predicts validation/test → predictions.json
  --stage segmenter : ENet on a CROP_MM box around the *localizer* series
                      centre (OOF for train, final model for val/test), all
                      slices including empty ones; then end-to-end evaluation.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.bootstrap import sanitize_sys_path

sanitize_sys_path()

import json

import numpy as np
import torch

from src.cascade.enet import ENetBinary
from src.cascade.v2 import (
    SIZE,
    LocalizerHead,
    SegmenterHead,
    center_error_mm,
    dice3d,
    fit,
    load_patient,
    localizer_tensors,
    run_cascade,
    run_localizer,
    segmenter_tensors,
    series_center,
    write_json,
)
from src.model import TinyUNet
from src.train_monitor import MetricsStore, start_dashboard

OUT = ROOT / "outputs" / "prostate_detect" / "cascade_v2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["localizer", "segmenter"], required=True)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "cascade_v2")
    p.add_argument("--loc-dir", type=Path, default=OUT / "localizer-v2")
    p.add_argument("--seg-dir", type=Path, default=OUT / "seg-enet-v3")
    p.add_argument("--crop-mm", type=float, default=110.0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=75)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--val-loss-target", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dashboard-port", type=int, default=8769)
    p.add_argument("--no-dashboard", action="store_true")
    p.add_argument("--eval-only", action="store_true", help="segmenter: reuse best.pt, only evaluate")
    return p.parse_args()


def read_split(data: Path, name: str) -> list[str]:
    return [s.strip() for s in (data / "splits" / f"{name}.txt").read_text().splitlines() if s.strip()]


def to_dev(arrs, device):
    return tuple(torch.from_numpy(np.concatenate(a)).to(device) for a in arrs)


class Dashboard:
    def __init__(self, args) -> None:
        self.store = MetricsStore()
        self.history: list[dict] = []
        if not args.no_dashboard:
            start_dashboard("0.0.0.0", args.dashboard_port, self.store)
            print(f"Dashboard: http://0.0.0.0:{args.dashboard_port}/")
        self.max_epochs = args.epochs
        self.patience = args.patience

    def start(self, name: str, desc: str) -> None:
        self.history = []
        self.store.update(status="running", run_name=name, description=desc, history=[],
                          max_epochs=self.max_epochs, patience=self.patience, epoch=0)

    def epoch(self, row: dict) -> None:
        self.history.append(row)
        self.store.update(
            epoch=row["epoch"], latest=row, history=self.history,
            best_epoch=row["best_epoch"], best_val_loss=row["best_val_loss"],
            best_val_dice=row["val_dice"] if row["best"] else self.store.snapshot().get("best_val_dice"),
            epochs_without_improve=row["stale"],
        )


def new_localizer(device):
    return TinyUNet(in_channels=1, out_channels=1, base=8).to(device)


def new_segmenter(device):
    return ENetBinary(in_channels=1, out_channels=1).to(device)


LOC_AUG = {"scale": 0.15, "shift": 0.10, "rot_deg": 10.0}
SEG_AUG = {"scale": 0.10, "shift": 0.05, "rot_deg": 10.0}


def localizer_summary(p, loc: dict, source: str) -> dict:
    Z, H, W = p.image.shape
    center = series_center(loc["mass"], loc["center_frac"], H, W)
    gt_pos = p.mask.reshape(Z, -1).any(1)
    pres = loc["presence"]
    return {
        "source": source,
        "center": list(center) if center else None,
        "gt_center": list(p.gt_center),
        "center_err_mm": center_error_mm(center, p.gt_center, p.spacing_yx),
        "slice_recall@0.5": float((pres[gt_pos] >= 0.5).mean()) if gt_pos.any() else None,
        "slice_fp@0.5": float((pres[~gt_pos] >= 0.5).mean()) if (~gt_pos).any() else None,
        "presence": [round(float(v), 4) for v in pres],
        "mass": [round(float(v), 6) for v in loc["mass"]],
    }


def describe(errs: list[float]) -> dict:
    a = np.asarray([e for e in errs if e is not None], dtype=float)
    if a.size == 0:
        return {}
    return {"n": int(a.size), "median": round(float(np.median(a)), 2),
            "p90": round(float(np.percentile(a, 90)), 2), "max": round(float(a.max()), 2)}


def train_localizer(args, device, dash: Dashboard) -> None:
    train_ids = read_split(args.data, "train")
    val_ids = read_split(args.data, "validation")
    test_ids = read_split(args.data, "test")
    patients = {pid: load_patient(args.data, pid) for pid in train_ids + val_ids + test_ids}
    tensors = {pid: localizer_tensors(p) for pid, p in patients.items()}
    val = to_dev(zip(*[tensors[pid] for pid in val_ids]), device)
    args.loc_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    order = train_ids[:]
    rng.shuffle(order)
    folds = [order[k :: args.folds] for k in range(args.folds)]
    preds: dict[str, dict] = {}

    runs = [(f"fold{k}", [i for i in train_ids if i not in folds[k]], folds[k]) for k in range(args.folds)]
    runs.append(("final", train_ids, val_ids + test_ids))
    for name, fit_ids, pred_ids in runs:
        torch.manual_seed(args.seed)
        net = new_localizer(device)
        train = to_dev(zip(*[tensors[pid] for pid in fit_ids]), device)
        dash.start(f"localizer-v2 {name}", f"TinyUNet base8 @{SIZE} full slice; fit {len(fit_ids)} pts")
        print(f"== localizer {name}: fit {len(fit_ids)} patients, predict {len(pred_ids)}")
        net, best = fit(net, train, val, aug=LOC_AUG, epochs=args.epochs, patience=args.patience,
                        batch_size=args.batch_size, on_epoch=dash.epoch)
        head = LocalizerHead(net).to(device)
        for pid in pred_ids:
            src = "oof" if name != "final" else ("validation" if pid in val_ids else "test")
            preds[pid] = localizer_summary(patients[pid], run_localizer(head, patients[pid], device), src)
        if name == "final":
            torch.save({"state_dict": net.state_dict(), "best": best, "base": 8, "size": SIZE},
                       args.loc_dir / "best.pt")
            final_best = best
        del train

    stats = {"best": final_best, "folds": args.folds}
    for src in ("oof", "validation", "test"):
        rows = [v for v in preds.values() if v["source"] == src]
        stats[src] = {
            "center_err_mm": describe([r["center_err_mm"] for r in rows]),
            "no_center": sum(r["center"] is None for r in rows),
            "slice_recall@0.5": round(float(np.mean([r["slice_recall@0.5"] for r in rows])), 3),
            "slice_fp@0.5": round(float(np.mean([r["slice_fp@0.5"] for r in rows if r["slice_fp@0.5"] is not None])), 3),
        }
    write_json(args.loc_dir / "predictions.json", preds)
    write_json(args.loc_dir / "stats.json", stats)
    dash.store.update(status="done", stats=stats)
    print(json.dumps(stats, indent=2))


def train_segmenter(args, device, dash: Dashboard) -> None:
    preds = json.loads((args.loc_dir / "predictions.json").read_text())
    train_ids = read_split(args.data, "train")
    val_ids = read_split(args.data, "validation")
    test_ids = read_split(args.data, "test")
    patients = {pid: load_patient(args.data, pid) for pid in train_ids + val_ids + test_ids}

    def center_for(pid):
        c = preds[pid]["center"]
        if c is None:
            H, W = patients[pid].image.shape[1:]
            return (H / 2, W / 2)
        return tuple(c)

    def seg_arrays(ids):
        out = [segmenter_tensors(patients[pid], center_for(pid), args.crop_mm)[:2] for pid in ids]
        return to_dev(zip(*out), device)

    net = new_segmenter(device)
    if args.eval_only:
        ck = torch.load(args.seg_dir / "best.pt", map_location=device)
        net.load_state_dict(ck["state_dict"])
        best = ck["best"]
    else:
        train = seg_arrays(train_ids)
        val = seg_arrays(val_ids)
        print(f"segmenter slices: train {train[0].shape[0]} val {val[0].shape[0]} "
              f"(positive train {(train[1].sum((1, 2, 3)) > 0).sum().item()})")
        torch.manual_seed(args.seed)
        dash.start("seg-enet-v3", f"ENet @{SIZE} crop {args.crop_mm:.0f}mm around localizer-v2 centre")
        net, best = fit(net, train, val, aug=SEG_AUG, epochs=args.epochs, patience=args.patience,
                        batch_size=args.batch_size, val_loss_target=args.val_loss_target,
                        on_epoch=dash.epoch)
        args.seg_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": net.state_dict(), "best": best, "crop_mm": args.crop_mm,
                    "size": SIZE}, args.seg_dir / "best.pt")

    ck = torch.load(args.loc_dir / "best.pt", map_location=device)
    loc_net = new_localizer(device)
    loc_net.load_state_dict(ck["state_dict"])
    loc_head = LocalizerHead(loc_net).to(device)
    seg_head = SegmenterHead(net).to(device)

    results = {}
    for split, ids in (("validation", val_ids), ("test", test_ids)):
        rows = []
        for pid in ids:
            p = patients[pid]
            r = run_cascade(loc_head, seg_head, p, args.crop_mm, device)
            g = run_cascade(loc_head, seg_head, p, args.crop_mm, device, center_override=p.gt_center)
            Z, H, W = p.image.shape
            coverage = None
            if r["box"] is not None:
                y0, x0, h, w = r["box"]
                yy = (np.arange(H) + 0.5)[:, None]
                xx = (np.arange(W) + 0.5)[None, :]
                inside = (yy >= y0) & (yy < y0 + h) & (xx >= x0) & (xx < x0 + w)
                coverage = float((p.mask & inside[None]).sum() / max(p.mask.sum(), 1))
            rows.append({
                "patient_id": pid,
                "dice3d": round(dice3d(r["pred"], p.mask), 4),
                "dice3d_gt_center": round(dice3d(g["pred"], p.mask), 4),
                "center_err_mm": center_error_mm(r["center"], p.gt_center, p.spacing_yx),
                "gt_in_crop": coverage,
                "pred_ml": round(float(r["pred"].sum()) * p.spacing_yx[0] * p.spacing_yx[1] * p.slice_mm / 1000, 1),
                "gt_ml": round(float(p.mask.sum()) * p.spacing_yx[0] * p.spacing_yx[1] * p.slice_mm / 1000, 1),
            })
            print(split, rows[-1])
        results[split] = {
            "mean_dice3d": round(float(np.mean([r["dice3d"] for r in rows])), 4),
            "mean_dice3d_gt_center": round(float(np.mean([r["dice3d_gt_center"] for r in rows])), 4),
            "patients": rows,
        }
    stats = {"best": best, "crop_mm": args.crop_mm, "end_to_end": results}
    write_json(args.seg_dir / "stats.json", stats)
    dash.store.update(status="done", stats={k: v["mean_dice3d"] for k, v in results.items()})
    print(json.dumps({k: (v["mean_dice3d"], v["mean_dice3d_gt_center"]) for k, v in results.items()}))


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    dash = Dashboard(args)
    t0 = time.time()
    if args.stage == "localizer":
        train_localizer(args, device, dash)
    else:
        train_segmenter(args, device, dash)
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
