#!/usr/bin/env python3
"""Export cascade v2 to ONNX + browser config + parity vectors.

Writes to <out>/:
  prostate_localizer_v2_192.onnx   input [B,1,192,192] → presence[B], mass[B], center_yx[B,2], prob[B,1,192,192]
  prostate_seg_v3_192.onnx         input [B,1,192,192] → prob[B,1,192,192]
  cascade_v2.json                  constants used by frontend/prostate_cascade_v2.js
  parity/<pid>.json + .bin         raw volume and expected intermediate values for one test series
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.bootstrap import sanitize_sys_path

sanitize_sys_path()

import numpy as np
import onnxruntime as ort
import torch

from src.cascade.enet import ENetBinary
from src.cascade.v2 import (
    HI_IDX,
    LO_IDX,
    LOCALIZER_THR,
    SIZE,
    LocalizerHead,
    SegmenterHead,
    dice3d,
    load_patient,
    run_cascade,
    write_json,
)
from src.model import TinyUNet

OUT = ROOT / "outputs" / "prostate_detect" / "cascade_v2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "cascade_v2")
    p.add_argument("--loc-dir", type=Path, default=OUT / "localizer-v2")
    p.add_argument("--seg-dir", type=Path, default=OUT / "seg-enet-v3")
    p.add_argument("--out", type=Path, default=OUT / "export")
    p.add_argument("--parity-patient", type=str, default=None, help="default: first test patient")
    return p.parse_args()


def export(model: torch.nn.Module, path: Path, outputs: list[str]) -> None:
    model.eval()
    dummy = torch.zeros(1, 1, SIZE, SIZE)
    torch.onnx.export(
        model, dummy, str(path), input_names=["input"], output_names=outputs,
        dynamic_axes={"input": {0: "batch"}, **{o: {0: "batch"} for o in outputs}},
        opset_version=17, dynamo=False,
    )


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cpu = torch.device("cpu")

    lck = torch.load(args.loc_dir / "best.pt", map_location=cpu)
    loc_net = TinyUNet(1, 1, base=lck.get("base", 8))
    loc_net.load_state_dict(lck["state_dict"])
    loc_head = LocalizerHead(loc_net).eval()
    sck = torch.load(args.seg_dir / "best.pt", map_location=cpu)
    seg_net = ENetBinary(1, 1)
    seg_net.load_state_dict(sck["state_dict"])
    seg_head = SegmenterHead(seg_net).eval()
    crop_mm = float(sck["crop_mm"])

    loc_path = args.out / "prostate_localizer_v2_192.onnx"
    seg_path = args.out / "prostate_seg_v3_192.onnx"
    export(loc_head, loc_path, ["presence", "mass", "center_yx", "prob"])
    export(seg_head, seg_path, ["prob"])

    config = {
        "version": "cascade-v2",
        "input_size": SIZE,
        "percentile_lo_index": LO_IDX,
        "percentile_hi_index": HI_IDX,
        "localizer_threshold": LOCALIZER_THR,
        "segmenter_threshold": 0.5,
        "crop_mm": crop_mm,
        "localizer_model": loc_path.name,
        "segmenter_model": seg_path.name,
        "localizer_outputs": ["presence", "mass", "center_yx", "prob"],
        "segmenter_outputs": ["prob"],
    }
    write_json(args.out / "cascade_v2.json", config)

    pid = args.parity_patient or next(
        s.strip() for s in (args.data / "splits" / "test.txt").read_text().splitlines() if s.strip()
    )
    p = load_patient(args.data, pid)
    ref = run_cascade(loc_head, seg_head, p, crop_mm, cpu)

    # ONNX Runtime must match PyTorch on the same preprocessed inputs.
    from src.cascade.v2 import localizer_tensors, segmenter_tensors

    xl, _ = localizer_tensors(p)
    lo = ort.InferenceSession(str(loc_path)).run(None, {"input": xl})
    xs, _, _ = segmenter_tensors(p, ref["center"], crop_mm)
    so = ort.InferenceSession(str(seg_path)).run(None, {"input": xs})[0]
    with torch.no_grad():
        so_t = seg_head(torch.from_numpy(xs)).numpy()
    print("ORT vs torch max |Δ|: presence", float(np.abs(lo[0] - ref["loc"]["presence"]).max()),
          "center", float(np.abs(lo[2] - ref["loc"]["center_frac"]).max()),
          "seg prob", float(np.abs(so - so_t).max()))

    par_dir = args.out / "parity"
    par_dir.mkdir(exist_ok=True)
    Z, H, W = p.image.shape
    p.image.astype("<f4").tofile(par_dir / f"{pid}.raw_f32.bin")
    mid = int(np.argmax(ref["pred"].reshape(Z, -1).sum(1)))
    write_json(par_dir / f"{pid}.json", {
        "patient_id": pid,
        "raw_file": f"{pid}.raw_f32.bin",
        "shape_zyx": [Z, H, W],
        "pixel_spacing_yx": list(p.spacing_yx),
        "crop_mm": crop_mm,
        "localizer": {
            "presence": [float(v) for v in ref["loc"]["presence"]],
            "mass": [float(v) for v in ref["loc"]["mass"]],
            "center_frac": [[float(a), float(b)] for a, b in ref["loc"]["center_frac"]],
            "input_check_slice": mid,
            "input_check_sum": float(xl[mid].sum()),
        },
        "series_center_yx": list(ref["center"]),
        "crop_box_y0_x0_h_w": list(ref["box"]),
        "segmenter": {"input_check_sum": float(xs[mid].sum())},
        "mask_pixels_per_slice": [int(v) for v in ref["pred"].reshape(Z, -1).sum(1)],
        "dice3d_vs_gt": dice3d(ref["pred"], p.mask),
    })
    print(f"exported → {args.out} (parity patient {pid})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
