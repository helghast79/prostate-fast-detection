"""Explicit architecture decisions for the cascade pipeline.

These are locked defaults. Scripts may override via CLI flags, but changing
a default here should be intentional and called out in the run description.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CascadeDecisions:
    # Stage 1 — dataset
    canonical_format: Literal["nifti"] = "nifti"
    dicom_adapter: bool = True  # Orthanc / folder DICOM → NIfTI
    modality: Literal["t2"] = "t2"
    normalization: Literal["percentile_01"] = "percentile_01"
    percentile_low: float = 1.0
    percentile_high: float = 99.0
    spacing_policy: Literal["native_with_affine"] = "native_with_affine"
    split_unit: Literal["patient"] = "patient"

    # Stage 2 — localization
    localizer_arch: Literal["shallow_cnn"] = "shallow_cnn"
    localizer_input_size: int = 128
    localizer_outputs: tuple[str, ...] = ("presence", "center_xy")
    # center_xy is normalized to [0, 1] in full-slice coordinates

    # Stage 3 — ROI
    roi_crop_size: int = 160  # recommended starting point (128 / 160 / 192)
    roi_margin_frac: float = 0.15  # extra pad around estimated gland extent
    roi_train_with_noisy_centers: bool = True
    roi_center_jitter_frac: float = 0.08  # relative to crop size

    # Stage 4 — segmentation
    segmenter_candidates: tuple[str, ...] = ("enet", "mobilenet_unet")
    segmenter_input_mode: Literal["2d"] = "2d"  # 2.5D deferred as experiment
    segmenter_in_channels_2d: int = 1
    segmenter_in_channels_25d: int = 3
    segmenter_loss: Literal["dice_bce"] = "dice_bce"


CASCADE_DECISIONS = CascadeDecisions()
