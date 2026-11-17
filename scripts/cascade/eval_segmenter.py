#!/usr/bin/env python3
"""Evaluate a Stage-4 segmenter on a patient split (ROI crops from GT centres)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.bootstrap import sanitize_sys_path

sanitize_sys_path()

import torch
from torch.utils.data import DataLoader

from src.cascade.datasets import SegmenterRoiDataset, load_split_patients
from src.cascade.mobilenet_unet import build_segmenter
from src.cascade.seg_loss import binary_dice, dice_bce_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=ROOT / "data" / "cascade")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    arch = ckpt.get("arch", "enet")
    crop = int(ckpt.get("crop_size", 160))
    in_ch = int(ckpt.get("in_channels", 1))
    model = build_segmenter(arch, in_channels=in_ch).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ids = load_split_patients(args.dataset, args.split)
    ds = SegmenterRoiDataset(
        args.dataset, ids, crop_size=crop, augment=False, jitter_centers=False
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    total_loss = 0.0
    total_dice = 0.0
    n = 0
    with torch.no_grad():
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

    metrics = {
        "split": args.split,
        "arch": arch,
        "crop_size": crop,
        "n_slices": n,
        "loss": total_loss / max(n, 1),
        "dice": total_dice / max(n, 1),
        "checkpoint": str(args.checkpoint),
    }
    print(json.dumps(metrics, indent=2))
    out = args.checkpoint.parent / f"eval_{args.split}.json"
    out.write_text(json.dumps(metrics, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
