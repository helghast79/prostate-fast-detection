#!/usr/bin/env python3
"""Train Stage-2 prostate localizer (presence + centre)."""

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

from src.cascade.datasets import LocalizerSliceDataset, load_split_patients
from src.cascade.decisions import CASCADE_DECISIONS
from src.cascade.localizer_model import ShallowProstateLocalizer, localizer_loss
from src.paths import ensure_task_dirs, task_outputs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=ROOT / "data" / "cascade")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    n = 0
    correct = 0
    total = 0
    center_err = 0.0
    center_n = 0
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
            err = (center_pred[pos] - center[pos]).abs().mean().item()
            center_err += err * int(pos.sum().item())
            center_n += int(pos.sum().item())
    return {
        "loss": total_loss / max(n, 1),
        "acc": correct / max(total, 1),
        "center_l1": center_err / max(center_n, 1),
    }


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    ensure_task_dirs("prostate_detect")
    run_name = args.run_name or datetime.now(timezone.utc).strftime(
        "localizer-%Y%m%d-%H%M%S"
    )
    out_dir = task_outputs("prostate_detect") / "cascade" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ids = load_split_patients(args.dataset, "train")
    val_ids = load_split_patients(args.dataset, "validation")
    if not train_ids:
        print("Empty train split — run split_patients.py first")
        return 1

    train_ds = LocalizerSliceDataset(
        args.dataset, train_ids, augment=True, seed=args.seed
    )
    val_ds = LocalizerSliceDataset(
        args.dataset, val_ids, augment=False, seed=args.seed + 1
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
    model = ShallowProstateLocalizer(in_channels=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    stale = 0
    history = []
    print(
        f"localizer train_slices={len(train_ds)} val_slices={len(val_ds)} "
        f"size={CASCADE_DECISIONS.localizer_input_size} device={device}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            present = batch["present"].to(device)
            center = batch["center"].to(device)
            opt.zero_grad(set_to_none=True)
            logits, center_pred = model(images)
            loss, _ = localizer_loss(logits, center_pred, present, center)
            loss.backward()
            opt.step()
            running += float(loss.item()) * images.size(0)
            n += images.size(0)
        train_loss = running / max(n, 1)
        val_m = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_m.items()}}
        history.append(row)
        print(
            f"epoch {epoch:03d}  train={train_loss:.4f}  "
            f"val={val_m['loss']:.4f}  acc={val_m['acc']:.3f}  "
            f"center_l1={val_m['center_l1']:.4f}  stale={stale}"
        )
        if val_m["loss"] < best_val:
            best_val = val_m["loss"]
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "arch": "shallow_cnn",
                    "image_size": CASCADE_DECISIONS.localizer_input_size,
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
    print(f"Best checkpoint: {out_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
