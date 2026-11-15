#!/usr/bin/env python3
"""Export a TinyUNet prostate checkpoint to ONNX (+ optional INT8 dynamic quantization)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.bootstrap import sanitize_sys_path
from src.paths import TASK_PROSTATE_DETECT, task_outputs

TASK = TASK_PROSTATE_DETECT

sanitize_sys_path()

import numpy as np
import torch

from src.model import TinyUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=task_outputs(TASK) / "runs" / "t2-prostate-tiny-v1" / "best.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Float ONNX path (default: alongside checkpoint as prostate_unet_192.onnx)",
    )
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument(
        "--in-channels",
        type=int,
        default=1,
        help="Input channels (T2 single-slice model uses 1)",
    )
    parser.add_argument(
        "--out-channels",
        type=int,
        default=None,
        help="Output classes (default: from checkpoint, else 1)",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--quantize", action="store_true", default=True)
    parser.add_argument("--no-quantize", action="store_false", dest="quantize")
    parser.add_argument("--smoke-test", action="store_true", default=True)
    parser.add_argument("--no-smoke-test", action="store_false", dest="smoke_test")
    return parser.parse_args()


def load_checkpoint(
    checkpoint: Path,
    base_channels: int,
    image_size: int,
    in_channels: int,
    out_channels: int | None,
):
    device = torch.device("cpu")
    state = torch.load(checkpoint, map_location=device, weights_only=False)

    weights = None
    ckpt_base = base_channels
    ckpt_size = image_size
    ckpt_in = in_channels
    ckpt_out = out_channels if out_channels is not None else 1
    class_names = None

    if isinstance(state, dict):
        if "model" in state:
            weights = state["model"]
        elif "model_state_dict" in state:
            weights = state["model_state_dict"]
        elif "state_dict" in state:
            weights = state["state_dict"]

        args = state.get("args") or {}
        if isinstance(args, dict):
            if args.get("base_channels") is not None:
                ckpt_base = int(args["base_channels"])
            if args.get("image_size") is not None:
                ckpt_size = int(args["image_size"])
            if args.get("num_classes") is not None and out_channels is None:
                ckpt_out = int(args["num_classes"])
        if state.get("base_channels") is not None:
            ckpt_base = int(state["base_channels"])
        if state.get("image_size") is not None:
            ckpt_size = int(state["image_size"])
        if state.get("in_channels") is not None:
            ckpt_in = int(state["in_channels"])
        if state.get("num_classes") is not None and out_channels is None:
            ckpt_out = int(state["num_classes"])
        if state.get("class_names") is not None:
            class_names = list(state["class_names"])
    else:
        weights = state

    if weights is None:
        raise ValueError(f"Could not find model weights in {checkpoint}")

    # Strip optional module. prefix
    if any(k.startswith("module.") for k in weights):
        weights = {k.replace("module.", "", 1): v for k, v in weights.items()}

    # Infer out channels from head weight if still ambiguous.
    head_w = weights.get("head.weight")
    if head_w is not None and out_channels is None:
        ckpt_out = int(head_w.shape[0])

    model = TinyUNet(in_channels=ckpt_in, out_channels=ckpt_out, base=ckpt_base).to(device)
    model.load_state_dict(weights)
    model.eval()
    return model, ckpt_in, ckpt_base, ckpt_size, ckpt_out, class_names


def export_onnx(
    model: torch.nn.Module,
    output: Path,
    image_size: int,
    in_channels: int,
    opset: int,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, in_channels, image_size, image_size)
    export_kwargs = dict(
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    try:
        # Prefer legacy exporter for broad onnxruntime-web compatibility.
        torch.onnx.export(
            model,
            dummy,
            str(output),
            dynamo=False,
            **export_kwargs,
        )
    except TypeError:
        torch.onnx.export(model, dummy, str(output), **export_kwargs)
    return output


def quantize_onnx(onnx_path: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quant_path = onnx_path.with_name(onnx_path.stem + ".quant.onnx")
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(quant_path),
        weight_type=QuantType.QUInt8,
    )
    return quant_path


def smoke_test(onnx_path: Path, image_size: int, in_channels: int = 1, runs: int = 20) -> dict:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.random.randn(1, in_channels, image_size, image_size).astype(np.float32)
    for _ in range(3):
        session.run(None, {"input": x})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = session.run(None, {"input": x})
        times.append((time.perf_counter() - t0) * 1000)
    logits = out[0]
    return {
        "onnx_path": str(onnx_path),
        "size_bytes": onnx_path.stat().st_size,
        "output_shape": list(logits.shape),
        "mean_ms": float(np.mean(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
    }


def main() -> int:
    args = parse_args()
    if not args.checkpoint.is_file():
        print(f"Missing checkpoint: {args.checkpoint}")
        return 1

    model, in_ch, base, size, out_ch, class_names = load_checkpoint(
        args.checkpoint,
        base_channels=args.base_channels,
        image_size=args.image_size,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
    )
    # CLI image-size wins when checkpoint didn't store one.
    image_size = args.image_size if args.image_size else size
    tag = "zones" if out_ch > 1 else "gland"

    output = args.output
    if output is None:
        output = args.checkpoint.parent / f"prostate_unet_{tag}_{image_size}.onnx"

    onnx_path = export_onnx(model, output, image_size, in_ch, args.opset)
    print(
        f"Exported ONNX: {onnx_path} "
        f"({onnx_path.stat().st_size} bytes, in={in_ch}, out={out_ch}, "
        f"base={base}, size={image_size})"
    )

    results = []
    if args.smoke_test:
        results.append(smoke_test(onnx_path, image_size, in_channels=in_ch))

    quant_path = None
    if args.quantize:
        quant_path = quantize_onnx(onnx_path)
        print(f"Quantized ONNX: {quant_path} ({quant_path.stat().st_size} bytes)")
        if args.smoke_test:
            results.append(smoke_test(quant_path, image_size, in_channels=in_ch))

    for r in results:
        print(
            f"smoke {Path(r['onnx_path']).name}: "
            f"{r['size_bytes']} bytes, shape={r['output_shape']}, "
            f"mean={r['mean_ms']:.2f}ms p50={r['p50_ms']:.2f}ms p95={r['p95_ms']:.2f}ms"
        )

    if out_ch > 1:
        postprocess = (
            "softmax over channel dim → argmax; "
            "0=background, 1=gland_non_pz, 2=peripheral_zone; "
            "whole_gland = (pred==1)|(pred==2)"
        )
    else:
        postprocess = "sigmoid(logits) > 0.5"
    meta = {
        "checkpoint": str(args.checkpoint),
        "float_onnx": str(onnx_path),
        "quant_onnx": str(quant_path) if quant_path else None,
        "input_name": "input",
        "output_name": "logits",
        "input_layout": "NCHW",
        "in_channels": in_ch,
        "out_channels": out_ch,
        "class_names": class_names,
        "image_size": image_size,
        "base_channels": base,
        "preprocess": "resize 192x192 grayscale; percentile 1-99 stretch to [0,1] float32",
        "postprocess": postprocess,
        "smoke": results,
    }
    meta_path = onnx_path.with_suffix(".frontend.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Frontend meta: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
