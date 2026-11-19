#!/usr/bin/env python3
"""Train cascade v3: localizer-v3 then prostate_detection-v3 (gland + PZ).

  --stage localizer : TinyUNet binary prostate on full T2 → localizer-v3
  --stage detection : ENet 3-class (bg/gland/PZ) on CROP_MM around localizer
                      series centre → prostate_detection-v3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.bootstrap import sanitize_sys_path

sanitize_sys_path()

import numpy as np
import torch

from src.cascade.enet import ENetBinary
from src.cascade.v3 import (
    CROP_MM_DEFAULT,
    NUM_CLASSES,
    SIZE,
    DetectionHead,
    LocalizerHead,
    center_error_mm,
    detection_tensors,
    fit_binary,
    fit_multiclass,
    load_patient,
    localizer_tensors,
    run_cascade,
    run_localizer,
    series_center,
    write_json,
    zone_dice3d,
)
from src.model import TinyUNet
from src.train_monitor import MetricsStore, start_dashboard

OUT = ROOT / "outputs" / "prostate_detect" / "cascade_v3"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["localizer", "detection"], required=True)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "cascade_v3")
    p.add_argument("--loc-dir", type=Path, default=OUT / "localizer-v3")
    p.add_argument("--det-dir", type=Path, default=OUT / "prostate_detection-v3")
    p.add_argument("--crop-mm", type=float, default=CROP_MM_DEFAULT)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dashboard-port", type=int, default=8769)
    p.add_argument("--no-dashboard", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
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
        self.store.update(
            status="running",
            run_name=name,
            description=desc,
            history=[],
            max_epochs=self.max_epochs,
            patience=self.patience,
            epoch=0,
        )

    def epoch(self, row: dict) -> None:
        self.history.append(row)
        self.store.update(
            epoch=row["epoch"],
            latest=row,
            history=self.history,
            best_epoch=row["best_epoch"],
            best_val_loss=row["best_val_loss"],
            best_val_dice=row["val_dice"] if row["best"] else self.store.snapshot().get("best_val_dice"),
            epochs_without_improve=row["stale"],
        )


LOC_AUG = {"scale": 0.15, "shift": 0.10, "rot_deg": 10.0}
DET_AUG = {"scale": 0.10, "shift": 0.05, "rot_deg": 10.0}


def localizer_summary(p, loc: dict, source: str) -> dict:
    Z, H, W = p.image.shape
    center = series_center(loc["mass"], loc["center_frac"], H, W)
    gt_pos = p.binary_mask.reshape(Z, -1).any(1)
    pres = loc["presence"]
    return {
        "source": source,
        "center": list(center) if center else None,
        "gt_center": list(p.gt_center) if p.gt_center else None,
        "center_err_mm": center_error_mm(center, p.gt_center, p.spacing_yx),
        "slice_recall@0.5": float((pres[gt_pos] >= 0.5).mean()) if gt_pos.any() else None,
        "slice_fp@0.5": float((pres[~gt_pos] >= 0.5).mean()) if (~gt_pos).any() else None,
        "presence": [round(float(v), 4) for v in pres],
        "mass": [round(float(v), 6) for v in loc["mass"]],
    }


def describe(errs: list) -> dict:
    a = np.asarray([e for e in errs if e is not None], dtype=float)
    if a.size == 0:
        return {}
    return {
        "n": int(a.size),
        "median": round(float(np.median(a)), 2),
        "p90": round(float(np.percentile(a, 90)), 2),
        "max": round(float(a.max()), 2),
    }


def train_localizer(args, device, dash: Dashboard) -> None:
    train_ids = read_split(args.data, "train")
    val_ids = read_split(args.data, "validation")
    test_ids = read_split(args.data, "test")
    print(f"loading {len(train_ids)+len(val_ids)+len(test_ids)} patients…")
    patients = {pid: load_patient(args.data, pid) for pid in train_ids + val_ids + test_ids}
    print("building localizer tensors…")
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
    final_best = None
    for name, fit_ids, pred_ids in runs:
        torch.manual_seed(args.seed)
        net = TinyUNet(in_channels=1, out_channels=1, base=8).to(device)
        train = to_dev(zip(*[tensors[pid] for pid in fit_ids]), device)
        dash.start(f"localizer-v3 {name}", f"TinyUNet base8 @{SIZE} full slice; fit {len(fit_ids)} pts")
        print(f"== localizer-v3 {name}: fit {len(fit_ids)} patients, predict {len(pred_ids)}")
        net, best = fit_binary(
            net,
            train,
            val,
            aug=LOC_AUG,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            on_epoch=dash.epoch,
        )
        head = LocalizerHead(net).to(device)
        for pid in pred_ids:
            src = "oof" if name != "final" else ("validation" if pid in val_ids else "test")
            preds[pid] = localizer_summary(patients[pid], run_localizer(head, patients[pid], device), src)
        if name == "final":
            torch.save(
                {"state_dict": net.state_dict(), "best": best, "base": 8, "size": SIZE, "version": "localizer-v3"},
                args.loc_dir / "best.pt",
            )
            final_best = best
        del train

    stats = {"version": "localizer-v3", "best": final_best, "folds": args.folds}
    for src in ("oof", "validation", "test"):
        rows = [v for v in preds.values() if v["source"] == src]
        if not rows:
            continue
        stats[src] = {
            "center_err_mm": describe([r["center_err_mm"] for r in rows]),
            "no_center": sum(r["center"] is None for r in rows),
            "slice_recall@0.5": round(float(np.mean([r["slice_recall@0.5"] for r in rows if r["slice_recall@0.5"] is not None])), 3),
            "slice_fp@0.5": round(
                float(np.mean([r["slice_fp@0.5"] for r in rows if r["slice_fp@0.5"] is not None])), 3
            ),
        }
    write_json(args.loc_dir / "predictions.json", preds)
    write_json(args.loc_dir / "stats.json", stats)
    dash.store.update(status="done", stats=stats)
    print(json.dumps(stats, indent=2))


def train_detection(args, device, dash: Dashboard) -> None:
    preds = json.loads((args.loc_dir / "predictions.json").read_text())
    train_ids = read_split(args.data, "train")
    val_ids = read_split(args.data, "validation")
    test_ids = read_split(args.data, "test")
    print(f"loading {len(train_ids)+len(val_ids)+len(test_ids)} patients…")
    patients = {pid: load_patient(args.data, pid) for pid in train_ids + val_ids + test_ids}

    def center_for(pid):
        c = preds[pid]["center"]
        if c is None:
            H, W = patients[pid].image.shape[1:]
            return (H / 2, W / 2)
        return tuple(c)

    def det_arrays(ids):
        out = [detection_tensors(patients[pid], center_for(pid), args.crop_mm)[:2] for pid in ids]
        return to_dev(zip(*out), device)

    net = ENetBinary(in_channels=1, out_channels=NUM_CLASSES).to(device)
    if args.eval_only:
        ck = torch.load(args.det_dir / "best.pt", map_location=device, weights_only=False)
        net.load_state_dict(ck["state_dict"])
        best = ck["best"]
    else:
        print("building detection tensors…")
        train = det_arrays(train_ids)
        val = det_arrays(val_ids)
        print(
            f"detection slices: train {train[0].shape[0]} val {val[0].shape[0]} "
            f"(positive train {(train[1].reshape(train[1].shape[0], -1).sum(1) > 0).sum().item()})"
        )
        torch.manual_seed(args.seed)
        dash.start(
            "prostate_detection-v3",
            f"ENet {NUM_CLASSES}-class @{SIZE} crop {args.crop_mm:.0f}mm around localizer-v3",
        )
        net, best = fit_multiclass(
            net,
            train,
            val,
            aug=DET_AUG,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=max(8, args.batch_size // 2),
            on_epoch=dash.epoch,
        )
        args.det_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": net.state_dict(),
                "best": best,
                "crop_mm": args.crop_mm,
                "size": SIZE,
                "num_classes": NUM_CLASSES,
                "version": "prostate_detection-v3",
            },
            args.det_dir / "best.pt",
        )

    ck = torch.load(args.loc_dir / "best.pt", map_location=device, weights_only=False)
    loc_net = TinyUNet(in_channels=1, out_channels=1, base=8).to(device)
    loc_net.load_state_dict(ck["state_dict"])
    loc_head = LocalizerHead(loc_net).to(device)
    det_head = DetectionHead(net).to(device)

    results = {}
    for split, ids in (("validation", val_ids), ("test", test_ids)):
        rows = []
        for pid in ids:
            p = patients[pid]
            r = run_cascade(loc_head, det_head, p, args.crop_mm, device)
            g = run_cascade(loc_head, det_head, p, args.crop_mm, device, center_override=p.gt_center)
            scores = zone_dice3d(r["pred"], p.mask)
            scores_gt = zone_dice3d(g["pred"], p.mask)
            Z, H, W = p.image.shape
            coverage = None
            if r["box"] is not None:
                y0, x0, h, w = r["box"]
                yy = (np.arange(H) + 0.5)[:, None]
                xx = (np.arange(W) + 0.5)[None, :]
                inside = (yy >= y0) & (yy < y0 + h) & (xx >= x0) & (xx < x0 + w)
                coverage = float(((p.mask > 0) & inside[None]).sum() / max((p.mask > 0).sum(), 1))
            rows.append(
                {
                    "patient_id": pid,
                    **{k: round(v, 4) for k, v in scores.items()},
                    **{f"{k}_gt_center": round(v, 4) for k, v in scores_gt.items()},
                    "center_err_mm": center_error_mm(r["center"], p.gt_center, p.spacing_yx),
                    "gt_in_crop": coverage,
                }
            )
            print(split, rows[-1])
        results[split] = {
            "mean_dice_prostate": round(float(np.mean([r["dice_prostate"] for r in rows])), 4),
            "mean_dice_gland": round(float(np.mean([r["dice_gland"] for r in rows])), 4),
            "mean_dice_pz": round(float(np.mean([r["dice_pz"] for r in rows])), 4),
            "mean_dice_prostate_gt_center": round(
                float(np.mean([r["dice_prostate_gt_center"] for r in rows])), 4
            ),
            "patients": rows,
        }
    stats = {
        "version": "prostate_detection-v3",
        "best": best,
        "crop_mm": args.crop_mm,
        "num_classes": NUM_CLASSES,
        "end_to_end": results,
    }
    write_json(args.det_dir / "stats.json", stats)
    dash.store.update(
        status="done",
        stats={k: v["mean_dice_prostate"] for k, v in results.items()},
    )
    print(
        json.dumps(
            {
                k: (v["mean_dice_prostate"], v["mean_dice_gland"], v["mean_dice_pz"])
                for k, v in results.items()
            },
            indent=2,
        )
    )


def main() -> int:
    args = parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    np.random.seed(args.seed)
    dash = Dashboard(args)
    t0 = time.time()
    if args.stage == "localizer":
        train_localizer(args, device, dash)
    else:
        train_detection(args, device, dash)
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
