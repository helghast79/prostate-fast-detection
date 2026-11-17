#!/usr/bin/env python3
"""Export cascade stage checkpoints to ONNX (+ optional dynamic quantize)."""

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

from src.cascade.decisions import CASCADE_DECISIONS
from src.cascade.localizer_model import ShallowProstateLocalizer
from src.cascade.mobilenet_unet import build_segmenter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--stage", type=str, required=True, choices=["localizer", "segmenter"])
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--quantize", action="store_true")
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args()


class _LocalizerOnnxWrapper(torch.nn.Module):
    def __init__(self, model: ShallowProstateLocalizer) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        presence, center = self.model(x)
        return presence, center


def main() -> int:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    if args.stage == "localizer":
        size = int(ckpt.get("image_size", CASCADE_DECISIONS.localizer_input_size))
        model = ShallowProstateLocalizer()
        model.load_state_dict(ckpt["model"])
        model.eval()
        wrapped = _LocalizerOnnxWrapper(model)
        dummy = torch.zeros(1, 1, size, size)
        out = args.output or args.checkpoint.parent / f"prostate_localizer_{size}.onnx"
        torch.onnx.export(
            wrapped,
            dummy,
            str(out),
            input_names=["input"],
            output_names=["presence_logit", "center_yx"],
            dynamic_axes={"input": {0: "batch"}, "presence_logit": {0: "batch"}, "center_yx": {0: "batch"}},
            opset_version=args.opset,
        )
        meta = {"stage": "localizer", "image_size": size, "onnx": str(out)}
    else:
        arch = ckpt.get("arch", "enet")
        size = int(ckpt.get("crop_size", CASCADE_DECISIONS.roi_crop_size))
        in_ch = int(ckpt.get("in_channels", 1))
        model = build_segmenter(arch, in_channels=in_ch)
        model.load_state_dict(ckpt["model"])
        model.eval()
        dummy = torch.zeros(1, in_ch, size, size)
        out = args.output or args.checkpoint.parent / f"prostate_seg_{arch}_{size}.onnx"
        torch.onnx.export(
            model,
            dummy,
            str(out),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=args.opset,
        )
        meta = {"stage": "segmenter", "arch": arch, "crop_size": size, "onnx": str(out)}

    print(f"Exported {out} ({out.stat().st_size} bytes)")
    if args.quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        qpath = out.with_name(out.stem + ".quant.onnx")
        quantize_dynamic(str(out), str(qpath), weight_type=QuantType.QUInt8)
        print(f"Quantized {qpath} ({qpath.stat().st_size} bytes)")
        meta["quant_onnx"] = str(qpath)

    meta_path = out.with_suffix(".frontend.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
