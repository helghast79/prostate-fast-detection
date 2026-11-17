#!/usr/bin/env python3
"""Evaluate a Stage-2 localizer checkpoint on a patient split."""

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

from src.cascade.datasets import LocalizerSliceDataset, load_split_patients
from src.cascade.localizer_model import ShallowProstateLocalizer, localizer_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=ROOT / "data" / "cascade")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ids = load_split_patients(args.dataset, args.split)
    ds = LocalizerSliceDataset(args.dataset, ids, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ShallowProstateLocalizer().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    total_loss = 0.0
    n = 0
    correct = 0
    total = 0
    center_err = 0.0
    center_n = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            present = batch["present"].to(device)
            center = batch["center"].to(device)
            logits, center_pred = model(images)
            loss, _ = localizer_loss(logits, center_pred, present, center)
            total_loss += float(loss.item()) * images.size(0)
            n += images.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += int((preds == present).sum().item())
            total += int(present.numel())
            pos = present > 0.5
            if pos.any():
                center_err += float(
                    (center_pred[pos] - center[pos]).abs().mean().item()
                ) * int(pos.sum().item())
                center_n += int(pos.sum().item())

    metrics = {
        "split": args.split,
        "n_slices": n,
        "loss": total_loss / max(n, 1),
        "acc": correct / max(total, 1),
        "center_l1": center_err / max(center_n, 1),
        "checkpoint": str(args.checkpoint),
    }
    print(json.dumps(metrics, indent=2))
    out = args.checkpoint.parent / f"eval_{args.split}.json"
    out.write_text(json.dumps(metrics, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
