# Experiment log

## cascade-v3 (2026-11-18)

ProCAncer-I DICOM catalog (`studies_with_masks` → `data/procancer_zones_dicom`), 637 kept cases (8 discarded). Cascade: full-slice localizer → 110 mm crop → zone segmenter. 192², percentile norm. Train/val/test = 509/64/64.

### localizer-v3
- Arch: TinyUNet base8, binary prostate on full T2
- Val center err median 0.66 mm (p90 1.57); slice recall@0.5 0.962, FP 0.096
- Test center err median 0.57 mm (p90 1.02); slice recall@0.5 0.961, FP 0.099
- Best val dice 0.916 @ epoch 34
- Artifacts: `outputs/prostate_detect/cascade_v3/localizer-v3/`, ONNX `prostate_localizer_v3_192.onnx`

### prostate_detection-v3
- Arch: ENet, 3-class softmax (bg / gland_non_pz / PZ) on 110 mm crop around localizer-v3 centre
- Val 3D Dice prostate 0.9104, gland 0.8849, PZ 0.8089
- Test 3D Dice prostate 0.9229, gland 0.8945, PZ 0.8165
- Best val multiclass dice 0.809 @ epoch 52
- Artifacts: `outputs/prostate_detect/cascade_v3/prostate_detection-v3/`, ONNX `prostate_detection_v3_192.onnx`, config `cascade_v3.json`

## lesion_detection-v3 (2026-11-19)

ProCAncer-I DICOM catalog (`studies_with_lesion` → `data/procancer_lesion_dicom`), 377 kept ready cases (9 discarded + 2 prep skips). ENet binary lesion on 110 mm crop around zone-SEG prostate centroid (localizer-v3 fallback unused — all cases had prostate SEG). Inputs: T2+ADC+DWI @ 192². Train/val/test = 301/38/38.

### lesion_detection-v3
- Arch: ENetBinary 3ch → 1, pos-repeat 4; best val dice 0.374 @ epoch 35 (~21 min)
- Val 3D Dice 0.272, lesions detected 36/46, FP components/patient 3.71 (in-prostate: Dice 0.271, FP 3.03)
- Test 3D Dice 0.328, lesions detected 35/47, FP components/patient 3.63 (in-prostate: Dice 0.301, FP 2.89)
- Artifacts: `outputs/prostate_detect/lesion/lesion_detection-v3/` (`best.pt`, `stats.json`)
- Export: `export/prostate_lesion_v3_192.onnx`, `lesion_v3.json` (ORT Δprob ~1e-5)

