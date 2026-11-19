#!/usr/bin/env python3
"""Train the prostate lesion model: ENet, 3 channels (T2, ADC, DWI) at 192².

Input box = cascade v2 crop (CROP_MM around the localizer-v2 series centre on
T2), every T2 slice (with and without lesion). Also writes a QC sheet of the
three channels with the lesion contour and the patient split.
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
from PIL import Image, ImageDraw
from scipy import ndimage

from src.cascade.enet import ENetBinary
from src.cascade.lesion import lesion_input, lesion_metrics, load_case, restrict_to_prostate
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
from src.model import TinyUNet

OUT = ROOT / "outputs" / "prostate_detect"
LESION_AUG = {"scale": 0.10, "shift": 0.05, "rot_deg": 10.0}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "lesion_v1")
    p.add_argument("--out", type=Path, default=OUT / "lesion" / "lesion-enet-v1")
    p.add_argument("--loc", type=Path, default=OUT / "cascade_v2" / "localizer-v2" / "best.pt")
    p.add_argument("--seg", type=Path, default=OUT / "cascade_v2" / "seg-enet-v3" / "best.pt")
    p.add_argument("--exclude", type=Path, default=ROOT / "data" / "lesion_v1" / "qc_exclude.txt")
    p.add_argument("--epochs", type=int, default=75)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pos-repeat", type=int, default=4, help="repeat lesion slices N times in training")
    p.add_argument("--qc-only", action="store_true")
    p.add_argument("--eval-only", action="store_true", help="reuse <out>/best.pt, only evaluate")
    p.add_argument("--dashboard-port", type=int, default=8769)
    p.add_argument("--no-dashboard", action="store_true")
    return p.parse_args()


def make_split(cases: dict, seed: int) -> dict[str, list[str]]:
    """Patient split stratified by label source: test 2 manual + 1 automatic, val 1 + 2."""
    rng = random.Random(seed)
    manual = sorted(p for p, c in cases.items() if c.meta["label_source"] == "manual")
    auto = sorted(p for p, c in cases.items() if c.meta["label_source"] != "manual")
    rng.shuffle(manual)
    rng.shuffle(auto)
    return {
        "test": manual[:2] + auto[:1],
        "validation": manual[2:3] + auto[1:3],
        "train": manual[3:] + auto[3:],
    }


@torch.no_grad()
def prostate_crop(case, loc_head, seg_head, crop_mm, device):
    """Localizer-v2 series centre → crop box, plus the predicted prostate mask."""
    t2 = case.t2.pixels
    Z, H, W = t2.shape
    x = np.stack([preprocess(t2[z], full_box(H, W)) for z in range(Z)])[:, None]
    _, mass, center, _ = loc_head(torch.from_numpy(x).to(device))
    c = series_center(mass.cpu().numpy(), center.cpu().numpy(), H, W) or (H / 2, W / 2)
    box = crop_box(c, (float(case.t2.spacing[0]), float(case.t2.spacing[1])), crop_mm)
    xs = np.stack([preprocess(t2[z], box) for z in range(Z)])[:, None]
    prob = seg_head(torch.from_numpy(xs).to(device)).cpu().numpy()[:, 0]
    prostate = keep_largest_slice_run(np.stack([paste_back(prob[z], box, H, W) for z in range(Z)]))
    return box, prostate


def summarize(rows: list[dict]) -> dict:
    return {
        "mean_dice3d": round(float(np.mean([r["dice3d"] for r in rows])), 4),
        "lesions_detected": f"{sum(r['detected'] for r in rows)}/{sum(r['gt_lesions'] for r in rows)}",
        "fp_components_per_patient": round(float(np.mean([r["false_positive_components"] for r in rows])), 2),
    }


def qc_sheet(cases, inputs, path: Path) -> None:
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
            rgb[edge] = (255, 40, 40)
            tiles.append(Image.fromarray(rgb))
        row = Image.new("RGB", (SIZE * 3, SIZE + 14))
        for i, t in enumerate(tiles):
            row.paste(t, (i * SIZE, 14))
        ImageDraw.Draw(row).text(
            (3, 1), f"{pid[:24]} z={z} {c.meta['label_source']} {c.meta['lesion_ml']}mL", fill=(255, 255, 0))
        rows.append(row)
    for s in range(0, len(rows), 8):
        chunk = rows[s : s + 8]
        sheet = Image.new("RGB", (SIZE * 3, sum(r.height for r in chunk)))
        for i, r in enumerate(chunk):
            sheet.paste(r, (0, i * r.height))
        sheet.save(path.parent / f"{path.stem}_{s // 8}.png")


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    excluded = set()
    if args.exclude.exists():
        excluded = {l.split("#")[0].strip() for l in args.exclude.read_text().splitlines() if l.split("#")[0].strip()}
    pids = sorted(f.stem for f in (args.data / "cases").glob("*.npz") if f.stem not in excluded)
    cases = {pid: load_case(args.data, pid) for pid in pids}

    lck = torch.load(args.loc, map_location=device)
    loc_net = TinyUNet(1, 1, base=lck.get("base", 8)).to(device)
    loc_net.load_state_dict(lck["state_dict"])
    sck = torch.load(args.seg, map_location=device)
    seg_net = ENetBinary(1, 1).to(device)
    seg_net.load_state_dict(sck["state_dict"])
    loc_head = LocalizerHead(loc_net).to(device).eval()
    seg_head = SegmenterHead(seg_net).to(device).eval()
    crop_mm = float(sck["crop_mm"])

    inputs, boxes, prostates = {}, {}, {}
    for pid, c in cases.items():
        box, prostate = prostate_crop(c, loc_head, seg_head, crop_mm, device)
        Z = c.t2.pixels.shape[0]
        x = np.stack([lesion_input(c.t2, c.adc, c.dwi, z, box) for z in range(Z)]).astype(np.float32)
        m = np.stack([sample_box(c.mask[z].astype(np.float32), *box, nearest=True) for z in range(Z)])[:, None]
        inputs[pid], boxes[pid], prostates[pid] = (x, m), box, prostate
        y0, x0, h, w = box
        H, W = c.mask.shape[1:]
        inside = ((np.arange(H) + 0.5)[:, None] >= y0) & ((np.arange(H) + 0.5)[:, None] < y0 + h) \
            & ((np.arange(W) + 0.5)[None, :] >= x0) & ((np.arange(W) + 0.5)[None, :] < x0 + w)
        cov = float((c.mask & inside[None]).sum() / max(c.mask.sum(), 1))
        print(f"{pid[:24]} slices={Z} lesion_slices={(m.reshape(Z, -1).sum(1) > 0).sum()} "
              f"lesion_in_crop={cov:.2f} adc_nonzero={(x[:, 1] > 0).mean():.2f} dwi_nonzero={(x[:, 2] > 0).mean():.2f}")
    qc_sheet(cases, inputs, args.out / "qc_channels.png")
    split = make_split(cases, args.seed)
    write_json(args.out / "split.json", split)
    print({k: len(v) for k, v in split.items()})
    if args.qc_only:
        return 0

    def stack(ids):
        return tuple(torch.from_numpy(np.concatenate([inputs[i][k] for i in ids])).to(device) for k in (0, 1))

    train, val = stack(split["train"]), stack(split["validation"])
    if args.pos_repeat > 1:
        pos = train[1].sum((1, 2, 3)) > 0
        train = tuple(torch.cat([t] + [t[pos]] * (args.pos_repeat - 1)) for t in train)
    print(f"slices train {train[0].shape[0]} (lesion {(train[1].sum((1, 2, 3)) > 0).sum().item()}) "
          f"val {val[0].shape[0]}")

    from src.train_monitor import MetricsStore, start_dashboard

    store, history = MetricsStore(), []
    if not args.no_dashboard:
        start_dashboard("0.0.0.0", args.dashboard_port, store)
        print(f"Dashboard: http://0.0.0.0:{args.dashboard_port}/")
    store.update(status="running", run_name=args.out.name, max_epochs=args.epochs, patience=args.patience,
                 description=f"ENet 3ch (T2/ADC/DWI) @{SIZE}, crop {crop_mm:.0f}mm around localizer-v2 centre")

    def on_epoch(row):
        history.append(row)
        store.update(epoch=row["epoch"], latest=row, history=history, best_epoch=row["best_epoch"],
                     best_val_loss=row["best_val_loss"], epochs_without_improve=row["stale"])

    torch.manual_seed(args.seed)
    net = ENetBinary(in_channels=3, out_channels=1).to(device)
    t0 = time.time()
    if args.eval_only:
        ck = torch.load(args.out / "best.pt", map_location=device)
        net.load_state_dict(ck["state_dict"])
        best = ck["best"]
    else:
        net, best = fit(net, train, val, aug=LESION_AUG, epochs=args.epochs, patience=args.patience,
                        batch_size=args.batch_size, on_epoch=on_epoch)
        torch.save({"state_dict": net.state_dict(), "best": best, "crop_mm": crop_mm, "size": SIZE,
                    "in_channels": 3, "channels": ["t2", "adc", "dwi"], "pos_repeat": args.pos_repeat},
                   args.out / "best.pt")

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
                r = {"patient_id": pid, "label_source": c.meta["label_source"], "gt_ml": c.meta["lesion_ml"],
                     **lesion_metrics(pred, c.mask),
                     "in_prostate": lesion_metrics(in_prostate, c.mask)}
                rows.append(r)
                print(split_name, r)
            results[split_name] = {"raw": summarize([r for r in rows]),
                                   "in_prostate": summarize([r["in_prostate"] for r in rows]),
                                   "patients": rows}
    stats = {"best": best, "crop_mm": crop_mm, "split": split, "excluded": sorted(excluded),
             "train_seconds": round(time.time() - t0), "results": results}
    write_json(args.out / "stats.json", stats)
    store.update(status="done", stats={k: v["raw"] for k, v in results.items()})
    print(json.dumps({k: {"raw": v["raw"], "in_prostate": v["in_prostate"]} for k, v in results.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
