#!/usr/bin/env python3
"""Train multi-class TinyUNet: background / gland_non_pz / peripheral_zone."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.bootstrap import sanitize_sys_path
from src.paths import TASK_PROSTATE_DETECT, ensure_task_dirs, task_outputs

sanitize_sys_path()
ensure_task_dirs(TASK_PROSTATE_DETECT)

import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from src.dataset import (
    ProstateSegDataset,
    filter_existing_pairs,
    filter_nonempty_masks,
    load_manifest,
)
from src.model import TinyUNet, ce_multiclass_dice_loss, multiclass_metrics
from src.train_monitor import MetricsStore, start_dashboard
from src.zone_labels import CLASS_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "zones_train")
    parser.add_argument("--val", type=Path, default=ROOT / "data" / "zones_val")
    parser.add_argument("--holdout", type=Path, default=ROOT / "data" / "zones_holdout")
    parser.add_argument("--output", type=Path, default=task_outputs(TASK_PROSTATE_DETECT))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=5e-5)
    parser.add_argument("--dashboard-host", type=str, default="0.0.0.0")
    parser.add_argument("--dashboard-port", type=int, default=8768)
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--description", type=str, default="")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    metrics_acc = {"dice": 0.0, "iou": 0.0, "dice_c1": 0.0, "dice_c2": 0.0}
    n = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = ce_multiclass_dice_loss(logits, masks, num_classes=num_classes)
        m = multiclass_metrics(logits, masks, num_classes=num_classes)
        bs = images.shape[0]
        total_loss += float(loss.item()) * bs
        for k in metrics_acc:
            metrics_acc[k] += float(m.get(k, 0.0)) * bs
        n += bs
    if n == 0:
        return {"loss": float("nan"), **{k: float("nan") for k in metrics_acc}}
    return {"loss": total_loss / n, **{k: v / n for k, v in metrics_acc.items()}}


def train_one_epoch(model, loader, optimizer, device, num_classes: int) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = ce_multiclass_dice_loss(logits, masks, num_classes=num_classes)
        loss.backward()
        optimizer.step()
        bs = images.shape[0]
        total += float(loss.item()) * bs
        n += bs
    return total / max(n, 1)


def write_stats(run_dir: Path, payload: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stats.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# {payload.get('run_name')}",
        "",
        f"- When: {payload.get('started_at')}",
        f"- Description: {payload.get('description')}",
        f"- Classes: {payload.get('class_names')}",
        f"- Epochs: {payload.get('epochs_ran')}/{payload.get('epochs_max')}",
        f"- Best epoch: {payload.get('best_epoch')}",
        f"- Best val loss: {payload.get('best_val_loss')}",
        f"- Best val Dice (mean FG): {payload.get('best_val_dice')}",
        f"- Best val Dice gland: {payload.get('best_val_dice_gland')}",
        f"- Best val Dice PZ: {payload.get('best_val_dice_pz')}",
        f"- Holdout Dice mean/gland/PZ: {payload.get('holdout_dice')} / "
        f"{payload.get('holdout_dice_gland')} / {payload.get('holdout_dice_pz')}",
        f"- Patients train/val/holdout: {payload.get('n_train_patients')}/"
        f"{payload.get('n_val_patients')}/{payload.get('n_holdout_patients')}",
        f"- Params: {payload.get('n_params')}  base={payload.get('base_channels')}",
        "",
    ]
    (run_dir / "stats.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or datetime.now(timezone.utc).strftime("t2-zones-%Y%m%d-%H%M%S")
    run_dir = args.output / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "history.jsonl"

    train_rows = filter_nonempty_masks(
        filter_existing_pairs(args.train, load_manifest(args.train))
    )
    val_rows = filter_nonempty_masks(filter_existing_pairs(args.val, load_manifest(args.val)))
    hold_rows = []
    if args.holdout.exists() and (args.holdout / "manifest.jsonl").exists():
        hold_rows = filter_nonempty_masks(
            filter_existing_pairs(args.holdout, load_manifest(args.holdout))
        )

    train_patients = sorted({r["patient_id"] for r in train_rows})
    val_patients = sorted({r["patient_id"] for r in val_rows})
    hold_patients = sorted({r["patient_id"] for r in hold_rows})
    print(
        f"zones train={len(train_patients)}/{len(train_rows)}  "
        f"val={len(val_patients)}/{len(val_rows)}  "
        f"holdout={len(hold_patients)}/{len(hold_rows)}  device={device}  "
        f"base={args.base_channels} classes={args.num_classes}"
    )
    if not train_rows or not val_rows:
        print("Need non-empty train and val")
        return 1

    train_ds = ProstateSegDataset(
        args.train, train_rows, augment=True, seed=args.seed, multiclass=True
    )
    val_ds = ProstateSegDataset(
        args.val, val_rows, augment=False, seed=args.seed, multiclass=True
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    hold_loader = None
    if hold_rows:
        hold_ds = ProstateSegDataset(
            args.holdout, hold_rows, augment=False, seed=args.seed, multiclass=True
        )
        hold_loader = DataLoader(
            hold_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    model = TinyUNet(
        in_channels=1, out_channels=args.num_classes, base=args.base_channels
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    metrics = MetricsStore()
    dashboard = None
    if not args.no_dashboard:
        dashboard = start_dashboard(args.dashboard_host, args.dashboard_port, metrics)
        print(f"Dashboard: http://{args.dashboard_host}:{args.dashboard_port}/")

    started = datetime.now(timezone.utc).isoformat()
    best_val = float("inf")
    best_epoch = 0
    best_dice = 0.0
    best_dice_gland = 0.0
    best_dice_pz = 0.0
    best_iou = 0.0
    stale = 0
    history_rows: list[dict] = []
    t0 = time.time()
    metrics.update(status="running", run_name=run_name, max_epochs=args.epochs, history=[])

    with history_path.open("w", encoding="utf-8") as history:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, device, args.num_classes
            )
            val_m = evaluate(model, val_loader, device, args.num_classes)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_m["loss"],
                "val_dice": val_m["dice"],
                "val_iou": val_m["iou"],
                "val_dice_gland": val_m["dice_c1"],
                "val_dice_pz": val_m["dice_c2"],
            }
            improved = val_m["loss"] < (best_val - args.min_delta)
            if improved:
                best_val = val_m["loss"]
                best_epoch = epoch
                best_dice = val_m["dice"]
                best_dice_gland = val_m["dice_c1"]
                best_dice_pz = val_m["dice_c2"]
                best_iou = val_m["iou"]
                stale = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "val_loss": best_val,
                        "val_dice": best_dice,
                        "val_dice_gland": best_dice_gland,
                        "val_dice_pz": best_dice_pz,
                        "num_classes": args.num_classes,
                        "class_names": list(CLASS_NAMES),
                        "in_channels": 1,
                        "base_channels": args.base_channels,
                        "image_size": int(os.getenv("IMAGE_SIZE", "192")),
                        "args": vars(args),
                    },
                    run_dir / "best.pt",
                )
                row["best"] = True
            else:
                stale += 1
                row["best"] = False

            history.write(json.dumps(row) + "\n")
            history.flush()
            history_rows.append(row)
            metrics.update(
                status="running",
                epoch=epoch,
                latest=row,
                history=history_rows,
                best_epoch=best_epoch,
                best_val_loss=best_val if best_epoch else None,
                best_dice=best_dice if best_epoch else None,
                epochs_without_improve=stale,
            )
            print(
                f"epoch {epoch:03d}  train={train_loss:.4f}  val={val_m['loss']:.4f}  "
                f"dice={val_m['dice']:.3f}  gland={val_m['dice_c1']:.3f}  "
                f"pz={val_m['dice_c2']:.3f}  stale={stale}"
            )
            if stale >= args.patience:
                print(f"Early stop at epoch {epoch}")
                break

    holdout = {}
    if hold_loader is not None and (run_dir / "best.pt").is_file():
        ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        hold_m = evaluate(model, hold_loader, device, args.num_classes)
        holdout = hold_m
        print(
            f"Holdout: loss={hold_m['loss']:.4f} dice={hold_m['dice']:.3f} "
            f"gland={hold_m['dice_c1']:.3f} pz={hold_m['dice_c2']:.3f}"
        )

    epochs_ran = history_rows[-1]["epoch"] if history_rows else 0
    stats = {
        "run_name": run_name,
        "description": args.description
        or "Multi-class TinyUNet: gland_non_pz + peripheral_zone on zone-labeled SEGs",
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - t0, 1),
        "epochs_max": args.epochs,
        "epochs_ran": epochs_ran,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "best_val_dice": best_dice,
        "best_val_dice_gland": best_dice_gland,
        "best_val_dice_pz": best_dice_pz,
        "best_val_iou": best_iou,
        "holdout_dice": holdout.get("dice"),
        "holdout_dice_gland": holdout.get("dice_c1"),
        "holdout_dice_pz": holdout.get("dice_c2"),
        "holdout_iou": holdout.get("iou"),
        "holdout_loss": holdout.get("loss"),
        "n_train_patients": len(train_patients),
        "n_val_patients": len(val_patients),
        "n_holdout_patients": len(hold_patients),
        "n_train_slices": len(train_rows),
        "n_val_slices": len(val_rows),
        "n_holdout_slices": len(hold_rows),
        "train_patients": train_patients,
        "val_patients": val_patients,
        "holdout_patients": hold_patients,
        "class_names": list(CLASS_NAMES),
        "num_classes": args.num_classes,
        "base_channels": args.base_channels,
        "n_params": n_params,
        "device": str(device),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
    }
    write_stats(run_dir, stats)
    log_path = args.output / "experiment_log.md"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {run_name}\n")
        f.write(f"- **When:** {started}\n")
        f.write(f"- **Description:** {stats['description']}\n")
        f.write(
            f"- **Best val:** loss={best_val:.4f} dice={best_dice:.3f} "
            f"gland={best_dice_gland:.3f} pz={best_dice_pz:.3f} (epoch {best_epoch})\n"
        )
        if holdout:
            f.write(
                f"- **Holdout:** dice={holdout.get('dice'):.3f} "
                f"gland={holdout.get('dice_c1'):.3f} pz={holdout.get('dice_c2'):.3f}\n"
            )
        f.write(f"- **Artifacts:** `{run_dir}`\n")
    print(f"Stats: {run_dir / 'stats.md'}")
    if dashboard is not None:
        print("Dashboard still running.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
