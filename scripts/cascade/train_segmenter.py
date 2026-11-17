#!/usr/bin/env python3
"""Train Stage-4 ROI segmenter (enet | mobilenet_unet) with shared eval."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.bootstrap import sanitize_sys_path

sanitize_sys_path()

import torch
from torch.utils.data import DataLoader

from src.cascade.datasets import SegmenterRoiDataset, load_split_patients
from src.cascade.decisions import CASCADE_DECISIONS
from src.cascade.mobilenet_unet import build_segmenter
from src.cascade.seg_loss import binary_dice, dice_bce_loss
from src.paths import ensure_task_dirs, task_outputs
from src.train_monitor import MetricsStore, start_dashboard


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=ROOT / "data" / "cascade")
    p.add_argument(
        "--arch",
        type=str,
        default="enet",
        choices=["enet", "mobilenet_unet"],
    )
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--epochs", type=int, default=75)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Stop if val loss does not improve for this many epochs",
    )
    p.add_argument(
        "--val-loss-target",
        type=float,
        default=0.25,
        help="Stop early once best val loss reaches this value (set <=0 to disable)",
    )
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--crop-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pretrained-encoder", action="store_true")
    p.add_argument("--no-center-jitter", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dashboard-host", type=str, default="0.0.0.0")
    p.add_argument("--dashboard-port", type=int, default=8769)
    p.add_argument("--no-dashboard", action="store_true")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    n = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = dice_bce_loss(logits, masks)
        dice = binary_dice(logits, masks)
        bs = images.size(0)
        total_loss += float(loss.item()) * bs
        total_dice += dice * bs
        n += bs
    return {
        "loss": total_loss / max(n, 1),
        "dice": total_dice / max(n, 1),
        "iou": 0.0,  # dashboard column; Dice is the primary metric here
    }


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    ensure_task_dirs("prostate_detect")
    crop = args.crop_size or CASCADE_DECISIONS.roi_crop_size
    run_name = args.run_name or datetime.now(timezone.utc).strftime(
        f"seg-{args.arch}-%Y%m%d-%H%M%S"
    )
    out_dir = task_outputs("prostate_detect") / "cascade" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "history.jsonl"

    train_ids = load_split_patients(args.dataset, "train")
    val_ids = load_split_patients(args.dataset, "validation")
    hold_ids = load_split_patients(args.dataset, "test")
    if not train_ids:
        print("Empty train split — run split_patients.py first")
        return 1

    train_ds = SegmenterRoiDataset(
        args.dataset,
        train_ids,
        crop_size=crop,
        augment=True,
        jitter_centers=not args.no_center_jitter,
        seed=args.seed,
    )
    val_ds = SegmenterRoiDataset(
        args.dataset,
        val_ids,
        crop_size=crop,
        augment=False,
        jitter_centers=False,
        seed=args.seed + 1,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_segmenter(
        args.arch, in_channels=1, pretrained=args.pretrained_encoder
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"seg arch={args.arch} params={n_params} crop={crop} "
        f"train={len(train_ds)} val={len(val_ds)} device={device} "
        f"epochs={args.epochs} patience={args.patience} "
        f"val_loss_target={args.val_loss_target}"
    )

    metrics = MetricsStore()
    if not args.no_dashboard:
        start_dashboard(args.dashboard_host, args.dashboard_port, metrics)
        print(f"Dashboard: http://{args.dashboard_host}:{args.dashboard_port}/")

    best_val = float("inf")
    best_dice = -1.0
    best_epoch = 0
    stale = 0
    history_rows: list[dict] = []
    stop_reason = "max_epochs"
    t0 = time.time()
    metrics.update(
        status="running",
        run_name=run_name,
        max_epochs=args.epochs,
        history=[],
        description=(
            f"{args.arch} ROI seg crop={crop}  "
            f"target_val_loss={args.val_loss_target} patience={args.patience}"
        ),
    )

    with history_path.open("w", encoding="utf-8") as history_f:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running = 0.0
            n = 0
            for batch in train_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(images)
                loss = dice_bce_loss(logits, masks)
                loss.backward()
                opt.step()
                running += float(loss.item()) * images.size(0)
                n += images.size(0)
            train_loss = running / max(n, 1)
            val_m = evaluate(model, val_loader, device)

            improved = val_m["loss"] < (best_val - args.min_delta)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_m["loss"],
                "val_dice": val_m["dice"],
                "val_iou": val_m["iou"],
                "stale": stale,
                "best": False,
            }
            if improved:
                best_val = val_m["loss"]
                best_dice = val_m["dice"]
                best_epoch = epoch
                stale = 0
                row["best"] = True
                torch.save(
                    {
                        "model": model.state_dict(),
                        "arch": args.arch,
                        "crop_size": crop,
                        "in_channels": 1,
                        "epoch": epoch,
                        "val": val_m,
                        "args": {
                            "lr": args.lr,
                            "batch_size": args.batch_size,
                            "jitter": not args.no_center_jitter,
                        },
                    },
                    out_dir / "best.pt",
                )
            else:
                stale += 1

            history_f.write(json.dumps(row) + "\n")
            history_f.flush()
            history_rows.append(row)
            metrics.update(
                status="running",
                epoch=epoch,
                latest=row,
                history=history_rows,
                best_epoch=best_epoch,
                best_val_loss=best_val if best_epoch else None,
                best_dice=best_dice if best_epoch else None,
                best_val_dice=best_dice if best_epoch else None,
                epochs_without_improve=stale,
                patience=args.patience,
                device=str(device),
            )
            print(
                f"epoch {epoch:03d}  train={train_loss:.4f}  "
                f"val={val_m['loss']:.4f}  dice={val_m['dice']:.3f}  "
                f"stale={stale}"
            )

            if args.val_loss_target > 0 and best_val <= args.val_loss_target:
                stop_reason = f"val_loss_target<={args.val_loss_target}"
                print(
                    f"Early stop at epoch {epoch}: "
                    f"best val_loss={best_val:.4f} ≤ target {args.val_loss_target}"
                )
                break
            if stale >= args.patience:
                stop_reason = f"patience={args.patience}"
                print(f"Early stop at epoch {epoch}: no val_loss improvement")
                break

    hold_m = None
    if hold_ids and (out_dir / "best.pt").is_file():
        ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        hold_ds = SegmenterRoiDataset(
            args.dataset,
            hold_ids,
            crop_size=crop,
            augment=False,
            jitter_centers=False,
            seed=args.seed + 2,
        )
        hold_loader = DataLoader(
            hold_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        hold_m = evaluate(model, hold_loader, device)
        print(
            f"Holdout: loss={hold_m['loss']:.4f} dice={hold_m['dice']:.3f} "
            f"({len(hold_ds)} slices)"
        )

    elapsed = time.time() - t0
    stats = {
        "arch": args.arch,
        "crop_size": crop,
        "best_epoch": best_epoch,
        "best_val_loss": best_val if best_val < float("inf") else None,
        "best_val_dice": best_dice if best_dice >= 0 else None,
        "holdout": hold_m,
        "n_params": n_params,
        "train_slices": len(train_ds),
        "val_slices": len(val_ds),
        "stop_reason": stop_reason,
        "elapsed_sec": elapsed,
        "val_loss_target": args.val_loss_target,
        "patience": args.patience,
        "epochs_cap": args.epochs,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    (out_dir / "history.json").write_text(json.dumps(history_rows, indent=2) + "\n")
    metrics.update(status="done", stats=stats, history=history_rows)
    print(
        f"Best checkpoint: {out_dir / 'best.pt'}  "
        f"epoch={best_epoch} loss={best_val:.4f} dice={best_dice:.3f}  "
        f"stop={stop_reason}"
    )
    print("Dashboard still running.")
    if not args.no_dashboard:
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
