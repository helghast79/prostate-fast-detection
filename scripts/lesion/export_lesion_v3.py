#!/usr/bin/env python3
"""Export lesion_detection-v3 to ONNX + parity vectors for the browser cascade.

Writes to <out>/:
  prostate_lesion_v3_192.onnx   input [B,3,192,192] (T2,ADC,DWI) → prob [B,1,192,192]
  lesion_v3.json                constants
  parity/<pid>.json + *.bin     raw volumes + expected crop / lesion pixels

Parity uses cascade v3 (localizer → 110 mm crop → prostate mask) so it matches
frontend/prostate_cascade_v2.js ``segmentSeries`` + ``detectLesions``.
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
from src.cascade.v2 import SIZE, LocalizerHead, SegmenterHead, paste_back, write_json
from src.cascade.v3 import DetectionHead, Patient, run_cascade
from src.model import TinyUNet

OUT = ROOT / "outputs" / "prostate_detect"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "lesion_v3")
    p.add_argument("--model", type=Path, default=OUT / "lesion" / "lesion_detection-v3" / "best.pt")
    p.add_argument("--loc", type=Path, default=OUT / "cascade_v3" / "localizer-v3" / "best.pt")
    p.add_argument("--det", type=Path, default=OUT / "cascade_v3" / "prostate_detection-v3" / "best.pt")
    p.add_argument("--out", type=Path, default=OUT / "lesion" / "lesion_detection-v3" / "export")
    p.add_argument(
        "--parity-patient",
        default="PCa-101750428546695148273213887696174053612",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cpu = torch.device("cpu")

    ck = torch.load(args.model, map_location=cpu, weights_only=False)
    net = ENetBinary(in_channels=3, out_channels=1)
    net.load_state_dict(ck["state_dict"])
    head = SegmenterHead(net).eval()
    onnx_path = args.out / "prostate_lesion_v3_192.onnx"
    torch.onnx.export(
        head,
        torch.zeros(1, 3, SIZE, SIZE),
        str(onnx_path),
        input_names=["input"],
        output_names=["prob"],
        dynamic_axes={"input": {0: "batch"}, "prob": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    crop_mm = float(ck.get("crop_mm", 110.0))
    write_json(
        args.out / "lesion_v3.json",
        {
            "version": "lesion-v3",
            "model": onnx_path.name,
            "input_size": SIZE,
            "channels": ["t2", "adc", "dwi"],
            "crop_mm": crop_mm,
            "threshold": 0.5,
            "prostate_restrict_radius_px": 3,
            "requires": "cascade v3 prostate result (box + binary masks) for the same T2 series",
            "cascade_config": "outputs/prostate_detect/cascade_v3/export/cascade_v3.json",
            "frontend": "frontend/prostate_cascade_v2.js → segmentSeries + detectLesions",
        },
    )

    lck = torch.load(args.loc, map_location=cpu, weights_only=False)
    loc_net = TinyUNet(1, 1, base=lck.get("base", 8))
    loc_net.load_state_dict(lck["state_dict"])
    loc_head = LocalizerHead(loc_net).eval()

    dck = torch.load(args.det, map_location=cpu, weights_only=False)
    det_net = ENetBinary(1, int(dck.get("num_classes", 3)))
    det_net.load_state_dict(dck["state_dict"])
    det_head = DetectionHead(det_net).eval()

    case = load_case(args.data, args.parity_patient)
    p = Patient(
        case.pid,
        case.t2.pixels,
        np.zeros(case.t2.pixels.shape, np.uint8),
        (float(case.t2.spacing[0]), float(case.t2.spacing[1])),
        3.0,
    )
    ref = run_cascade(loc_head, det_head, p, crop_mm, cpu)
    box = ref["box"]
    if box is None:
        raise SystemExit("cascade returned no prostate box for parity patient")
    prostate = ref["pred_binary"]
    Z, H, W = case.t2.pixels.shape
    x = np.stack([lesion_input(case.t2, case.adc, case.dwi, z, box) for z in range(Z)]).astype(
        np.float32
    )
    with torch.no_grad():
        prob_t = head(torch.from_numpy(x)).numpy()
    prob = ort.InferenceSession(str(onnx_path)).run(None, {"input": x})[0]
    print("ORT vs torch max |Δ| prob", float(np.abs(prob - prob_t).max()))
    lesion = np.stack([paste_back(prob[z, 0], box, H, W) for z in range(Z)])
    lesion = restrict_to_prostate(lesion, prostate)

    par = args.out / "parity"
    par.mkdir(exist_ok=True)
    series_meta = {}
    for name in ("t2", "adc", "dwi"):
        s = getattr(case, name)
        s.pixels.astype("<f4").tofile(par / f"{name}.f32.bin")
        series_meta[name] = {
            "file": f"{name}.f32.bin",
            "shape_zyx": list(s.pixels.shape),
            "ipp": s.ipp.tolist(),
            "iop": s.iop.tolist(),
            "spacing": s.spacing.tolist(),
        }
    zl = int(np.argmax(case.mask.reshape(Z, -1).sum(1)))
    write_json(
        par / "lesion_parity.json",
        {
            "patient_id": case.pid,
            "series": series_meta,
            "prostate_crop_box": list(box),
            "check_slice": zl,
            "lesion_input_channel_sums": [float(x[zl, c].sum()) for c in range(3)],
            "prostate_pixels_per_slice": [int(v) for v in prostate.reshape(Z, -1).sum(1)],
            "lesion_pixels_per_slice": [int(v) for v in lesion.reshape(Z, -1).sum(1)],
        },
    )
    print(f"exported → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
