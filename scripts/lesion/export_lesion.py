#!/usr/bin/env python3
"""Export the lesion model to ONNX + parity vectors for frontend/prostate_cascade_v2.js.

Writes to <out>/:
  prostate_lesion_v1_192.onnx   input [B,3,192,192] (T2, ADC, DWI) → prob [B,1,192,192]
  lesion_v1.json                constants
  parity/<pid>.json + *.bin     raw T2/ADC/DWI + geometry and expected values
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
from src.cascade.lesion import lesion_input, load_case, restrict_to_prostate
from src.cascade.v2 import (
    SIZE,
    LocalizerHead,
    SegmenterHead,
    paste_back,
    run_cascade,
    write_json,
)
from src.model import TinyUNet

OUT = ROOT / "outputs" / "prostate_detect"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "lesion_v1")
    ap.add_argument("--model", type=Path, default=OUT / "lesion" / "lesion-enet-v1" / "best.pt")
    ap.add_argument("--loc", type=Path, default=OUT / "cascade_v2" / "localizer-v2" / "best.pt")
    ap.add_argument("--seg", type=Path, default=OUT / "cascade_v2" / "seg-enet-v3" / "best.pt")
    ap.add_argument("--out", type=Path, default=OUT / "lesion" / "lesion-enet-v1" / "export")
    ap.add_argument("--parity-patient", default="PCa-19760396733364480959673991650144739651")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cpu = torch.device("cpu")

    ck = torch.load(args.model, map_location=cpu)
    net = ENetBinary(in_channels=3, out_channels=1)
    net.load_state_dict(ck["state_dict"])
    head = SegmenterHead(net).eval()
    onnx_path = args.out / "prostate_lesion_v1_192.onnx"
    torch.onnx.export(head, torch.zeros(1, 3, SIZE, SIZE), str(onnx_path), input_names=["input"],
                      output_names=["prob"], dynamic_axes={"input": {0: "batch"}, "prob": {0: "batch"}},
                      opset_version=17, dynamo=False)
    write_json(args.out / "lesion_v1.json", {
        "version": "lesion-v1",
        "model": onnx_path.name,
        "input_size": SIZE,
        "channels": ["t2", "adc", "dwi"],
        "crop_mm": ck["crop_mm"],
        "threshold": 0.5,
        "prostate_restrict_radius_px": 3,
        "requires": "cascade v2 prostate result (box + masks) for the same T2 series",
    })

    lck = torch.load(args.loc, map_location=cpu)
    loc_net = TinyUNet(1, 1, base=lck.get("base", 8))
    loc_net.load_state_dict(lck["state_dict"])
    sck = torch.load(args.seg, map_location=cpu)
    seg_net = ENetBinary(1, 1)
    seg_net.load_state_dict(sck["state_dict"])
    loc_head, seg_head = LocalizerHead(loc_net).eval(), SegmenterHead(seg_net).eval()

    case = load_case(args.data, args.parity_patient)
    from src.cascade.v2 import Patient

    p = Patient(case.pid, case.t2.pixels, np.zeros(case.t2.pixels.shape, np.uint8),
                (float(case.t2.spacing[0]), float(case.t2.spacing[1])), 3.0)
    ref = run_cascade(loc_head, seg_head, p, float(sck["crop_mm"]), cpu)
    box = ref["box"]
    Z, H, W = case.t2.pixels.shape
    x = np.stack([lesion_input(case.t2, case.adc, case.dwi, z, box) for z in range(Z)]).astype(np.float32)
    with torch.no_grad():
        prob_t = head(torch.from_numpy(x)).numpy()
    prob = ort.InferenceSession(str(onnx_path)).run(None, {"input": x})[0]
    print("ORT vs torch max |Δ| prob", float(np.abs(prob - prob_t).max()))
    lesion = np.stack([paste_back(prob[z, 0], box, H, W) for z in range(Z)])
    lesion = restrict_to_prostate(lesion, ref["pred"])

    par = args.out / "parity"
    par.mkdir(exist_ok=True)
    series_meta = {}
    for name in ("t2", "adc", "dwi"):
        s = getattr(case, name)
        s.pixels.astype("<f4").tofile(par / f"{name}.f32.bin")
        series_meta[name] = {"file": f"{name}.f32.bin", "shape_zyx": list(s.pixels.shape),
                             "ipp": s.ipp.tolist(), "iop": s.iop.tolist(), "spacing": s.spacing.tolist()}
    zl = int(np.argmax(case.mask.reshape(Z, -1).sum(1)))
    write_json(par / "lesion_parity.json", {
        "patient_id": case.pid,
        "series": series_meta,
        "prostate_crop_box": list(box),
        "check_slice": zl,
        "lesion_input_channel_sums": [float(x[zl, c].sum()) for c in range(3)],
        "prostate_pixels_per_slice": [int(v) for v in ref["pred"].reshape(Z, -1).sum(1)],
        "lesion_pixels_per_slice": [int(v) for v in lesion.reshape(Z, -1).sum(1)],
    })
    print(f"exported → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
