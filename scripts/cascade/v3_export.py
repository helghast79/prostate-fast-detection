#!/usr/bin/env python3
"""Export cascade v3 ONNX models with clear version names.

  prostate_localizer_v3_192.onnx
  prostate_detection_v3_192.onnx   (3-class softmax: bg / gland / PZ)
  cascade_v3.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.bootstrap import sanitize_sys_path

sanitize_sys_path()

import torch

from src.cascade.enet import ENetBinary
from src.cascade.v2 import HI_IDX, LO_IDX, LOCALIZER_THR
from src.cascade.v3 import (
    NUM_CLASSES,
    SIZE,
    DetectionHead,
    LocalizerHead,
    write_json,
)
from src.model import TinyUNet
from src.zone_labels import CLASS_NAMES

OUT = ROOT / "outputs" / "prostate_detect" / "cascade_v3"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--loc-dir", type=Path, default=OUT / "localizer-v3")
    p.add_argument("--det-dir", type=Path, default=OUT / "prostate_detection-v3")
    p.add_argument("--out", type=Path, default=OUT / "export")
    return p.parse_args()


def export(model: torch.nn.Module, path: Path, outputs: list[str]) -> None:
    model.eval()
    dummy = torch.zeros(1, 1, SIZE, SIZE)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["input"],
        output_names=outputs,
        dynamic_axes={"input": {0: "batch"}, **{o: {0: "batch"} for o in outputs}},
        opset_version=17,
        dynamo=False,
    )


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cpu = torch.device("cpu")

    lck = torch.load(args.loc_dir / "best.pt", map_location=cpu, weights_only=False)
    loc_net = TinyUNet(1, 1, base=lck.get("base", 8))
    loc_net.load_state_dict(lck["state_dict"])
    loc_head = LocalizerHead(loc_net).eval()

    dck = torch.load(args.det_dir / "best.pt", map_location=cpu, weights_only=False)
    det_net = ENetBinary(1, int(dck.get("num_classes", NUM_CLASSES)))
    det_net.load_state_dict(dck["state_dict"])
    det_head = DetectionHead(det_net).eval()
    crop_mm = float(dck["crop_mm"])

    loc_path = args.out / "prostate_localizer_v3_192.onnx"
    det_path = args.out / "prostate_detection_v3_192.onnx"
    export(loc_head, loc_path, ["presence", "mass", "center_yx", "prob"])
    export(det_head, det_path, ["prob"])

    config = {
        "version": "cascade-v3",
        "localizer_version": "localizer-v3",
        "detection_version": "prostate_detection-v3",
        "input_size": SIZE,
        "percentile_lo_index": LO_IDX,
        "percentile_hi_index": HI_IDX,
        "localizer_threshold": LOCALIZER_THR,
        "crop_mm": crop_mm,
        "num_classes": NUM_CLASSES,
        "class_names": list(CLASS_NAMES),
        "localizer_model": loc_path.name,
        "detection_model": det_path.name,
        "localizer_outputs": ["presence", "mass", "center_yx", "prob"],
        "detection_outputs": ["prob"],
        "detection_prob_layout": "softmax over [background, gland_non_pz, peripheral_zone]",
    }
    write_json(args.out / "cascade_v3.json", config)
    print(f"wrote {loc_path.name}, {det_path.name}, cascade_v3.json → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
