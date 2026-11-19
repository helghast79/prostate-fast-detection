#!/usr/bin/env python3
"""Build the lesion dataset: T2 + ADC + DWI + lesion mask (on the T2 grid).

Uses the review catalog (scripts/serve_manifest_review.py): saved tags, or the
catalog's suggestions for series not reviewed yet. Discarded series are ignored.
  lesion SEG : every SEG labelled "binary" or "prob60", merged (union)
               binary → any marked pixel of the lesion-like segments
               prob60 → FRACTIONAL value ≥ 0.6·max, or "(x-y%)" bins with x ≥ 60
  T2         : the t2 series referenced by a lesion SEG, else the first axial t2
  ADC, DWI   : first adc / dwi tagged series; multi-b DWI keeps the darkest
               image per slice position (highest b-value)

Output: <out>/cases/<pid>.npz (+ .json) and <out>/index.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.bootstrap import sanitize_sys_path

sanitize_sys_path()

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from src.cascade.lesion import Series
from src.cascade.orthanc_volumes import (
    keep_lowest_signal_per_position,
    load_mr_slices,
    seg_to_mask_volume,
    segmentation_type,
)
from src.orthanc_client import OrthancClient
from src.seg_masks import probability_bin_at_least
from src.series_manifest import LESION_TOKENS, enrich_patients

PROB_THRESHOLD = 0.6


def lesion_like(label: str) -> bool:
    return any(t in label.lower() for t in LESION_TOKENS)


def label_filter_for(choice: str):
    if choice == "prob60":
        return lambda l: probability_bin_at_least(l, PROB_THRESHOLD * 100) or ("%" not in l and lesion_like(l))
    return lesion_like


def label_source(segs: list[dict]) -> str:
    if any("_lesions" in s["series_description"].lower() for s in segs):
        return "manual"
    return "probabilistic" if all(s["lesion_seg"] == "prob60" for s in segs) else "automatic"


def select_cases(patients: list[dict]) -> list[dict]:
    cases = []
    for p in patients:
        series = [s for s in p["series"] if not s["discarded"]]
        tag = lambda t: [s for s in series if s["modality"] == "MR" and s["mr_tag"] == t]
        segs = [s for s in series if s["modality"] == "SEG" and s.get("lesion_seg") in ("binary", "prob60")]
        t2s, adcs, dwis = tag("t2"), tag("adc"), tag("dwi")
        if not (segs and t2s and adcs and dwis):
            continue
        refs = {u for s in segs for u in s["referenced_series_uids"]}
        axial = [t for t in t2s if any(k in t["series_description"].lower() for k in ("tra", "ax"))]
        t2 = next((t for t in t2s if t["series_instance_uid"] in refs), (axial or t2s)[0])
        cases.append({"patient_id": p["patient_id"], "segs": segs, "t2": t2, "adc": adcs[0], "dwi": dwis[0]})
    return cases


def to_series(slices) -> Series:
    g = slices[0].geometry
    return Series(
        pixels=np.stack([s.pixels for s in slices]).astype(np.float32),
        ipp=np.stack([s.geometry.ipp for s in slices]).astype(np.float64),
        iop=np.asarray(g.iop, dtype=np.float64),
        spacing=np.asarray(g.spacing, dtype=np.float64),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "lesion_v2")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "series_manifest.json")
    ap.add_argument("--annotations", type=Path, default=ROOT / "data" / "series_annotations.json")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")

    manifest = json.loads(args.manifest.read_text())
    annotations = json.loads(args.annotations.read_text())
    client = OrthancClient()
    patients = enrich_patients(manifest["patients"], annotations, lambda s: segmentation_type(client, s))
    cases = select_cases(patients)
    print(f"{len(cases)} patients with lesion SEG + T2 + ADC + DWI")
    (args.out / "cases").mkdir(parents=True, exist_ok=True)
    index = []
    for c in tqdm(cases, desc="Patients"):
        pid = c["patient_id"]
        t2_sl = load_mr_slices(client, c["t2"]["series_instance_uid"])
        adc_sl = keep_lowest_signal_per_position(load_mr_slices(client, c["adc"]["series_instance_uid"]))
        dwi_sl = keep_lowest_signal_per_position(load_mr_slices(client, c["dwi"]["series_instance_uid"]))
        mask = np.zeros((len(t2_sl), *t2_sl[0].pixels.shape), dtype=np.uint8)
        info = {"matched_planes": 0, "unmatched_planes": 0}
        for seg in c["segs"]:
            m, i = seg_to_mask_volume(
                client, seg["orthanc_series_id"], t2_sl,
                label_filter=label_filter_for(seg["lesion_seg"]),
                fractional_threshold=PROB_THRESHOLD if seg["lesion_seg"] == "prob60" else None,
            )
            mask |= m
            info["matched_planes"] += i["matched_planes"]
            info["unmatched_planes"] += i["unmatched_planes"]
        t2, adc, dwi = to_series(t2_sl), to_series(adc_sl), to_series(dwi_sl)
        np.savez_compressed(args.out / "cases" / f"{pid}.npz", mask=mask,
                            **t2.to_npz("t2"), **adc.to_npz("adc"), **dwi.to_npz("dwi"))
        meta = {
            "patient_id": pid,
            "lesion_segs": [
                {"description": s["series_description"], "series_instance_uid": s["series_instance_uid"],
                 "labels": s["segment_labels"], "segmentation_type": s.get("segmentation_type"),
                 "lesion_seg": s["lesion_seg"]}
                for s in c["segs"]
            ],
            "label_source": label_source(c["segs"]),
            "t2": c["t2"]["series_description"], "adc": c["adc"]["series_description"],
            "dwi": c["dwi"]["series_description"],
            "shape_t2": list(t2.pixels.shape), "shape_adc": list(adc.pixels.shape),
            "shape_dwi": list(dwi.pixels.shape),
            "lesion_voxels": int(mask.sum()),
            "lesion_ml": round(float(mask.sum()) * t2.spacing[0] * t2.spacing[1]
                               * float(np.linalg.norm(t2.ipp[-1] - t2.ipp[0]) / max(len(t2.ipp) - 1, 1)) / 1000, 2),
            "lesion_slices": int((mask.reshape(len(mask), -1).sum(1) > 0).sum()),
            **info,
        }
        (args.out / "cases" / f"{pid}.json").write_text(json.dumps(meta, indent=2))
        index.append(meta)
        print(pid[:20], meta["label_source"], meta["lesion_ml"], "mL", meta["lesion_slices"], "slices",
              "unmatched", info["unmatched_planes"])
    (args.out / "index.json").write_text(json.dumps(index, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
