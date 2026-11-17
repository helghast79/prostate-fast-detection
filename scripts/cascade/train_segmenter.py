#!/usr/bin/env python3
"""Train Stage-4 ROI segmenter (enet | mobilenet_unet) with shared eval."""

from __future__ import annotations

import argparse
import json
import sys
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
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--crop-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--pretrained-encoder", action="store_true")
    p.add_argument("--no-center-jitter", action="store_true")
    p.add_argument("--seed", type=int, default=42)
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

    train_ids = load_split_patients(args.dataset, "train")
    val_ids = load_split_patients(args.dataset, "validation")
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
        f"train={len(train_ds)} val={len(val_ds)} device={device}"
    )

    best_dice = -1.0
    stale = 0
    history = []
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
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_m["loss"],
            "val_dice": val_m["dice"],
            "stale": stale,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}  train={train_loss:.4f}  "
            f"val={val_m['loss']:.4f}  dice={val_m['dice']:.3f}  stale={stale}"
        )
        if val_m["dice"] > best_dice:
            best_dice = val_m["dice"]
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "arch": args.arch,
                    "crop_size": crop,
                    "in_channels": 1,
                    "epoch": epoch,
                    "val": val_m,
                },
                out_dir / "best.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stop at epoch {epoch}")
                break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    stats = {
        "arch": args.arch,
        "crop_size": crop,
        "best_val_dice": best_dice,
        "n_params": n_params,
        "train_slices": len(train_ds),
        "val_slices": len(val_ds),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"Best checkpoint: {out_dir / 'best.pt'}  dice={best_dice:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
