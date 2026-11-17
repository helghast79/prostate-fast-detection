// Node parity test: JS pipeline vs Python reference on the exported parity series.
// Usage: node frontend/test_parity.mjs <export_dir> [<onnxruntime-node path>]
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { createRequire } from 'node:module';
import { preprocess, segmentSeries } from './prostate_cascade_v2.js';

const exportDir = process.argv[2];
const require = createRequire(import.meta.url);
const ort = require(process.argv[3] || 'onnxruntime-node');

const cfg = JSON.parse(readFileSync(join(exportDir, 'cascade_v2.json'), 'utf8'));
const parDir = join(exportDir, 'parity');
const ref = JSON.parse(readFileSync(join(parDir, readdirSync(parDir).find((f) => f.endsWith('.json'))), 'utf8'));
const [Z, H, W] = ref.shape_zyx;
const buf = readFileSync(join(parDir, ref.raw_file));
const vol = new Float32Array(buf.buffer, buf.byteOffset, Z * H * W);
const slices = Array.from({ length: Z }, (_, z) => vol.subarray(z * H * W, (z + 1) * H * W));

const loc = await ort.InferenceSession.create(join(exportDir, cfg.localizer_model));
const seg = await ort.InferenceSession.create(join(exportDir, cfg.segmenter_model));
const res = await segmentSeries(ort, loc, seg, slices, H, W, ref.pixel_spacing_yx, cfg.crop_mm);

const mid = ref.localizer.input_check_slice;
const sum = (a) => a.reduce((s, v) => s + v, 0);
const locSum = sum(preprocess(slices[mid], H, W, [0, 0, H, W]));
const segSum = sum(preprocess(slices[mid], H, W, res.box));
const maxDiff = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));

const report = {
  localizer_input_sum: [locSum, ref.localizer.input_check_sum],
  presence_max_abs_diff: maxDiff(res.localizer.map((s) => s.presence), ref.localizer.presence),
  center_yx: [res.center, ref.series_center_yx],
  segmenter_input_sum: [segSum, ref.segmenter.input_check_sum],
  mask_pixels_per_slice_js: res.masks.map((m) => sum(m)),
  mask_pixels_per_slice_py: ref.mask_pixels_per_slice,
};
console.log(JSON.stringify(report, null, 1));
const pixDiff = report.mask_pixels_per_slice_js.reduce((s, v, i) => s + Math.abs(v - ref.mask_pixels_per_slice[i]), 0);
const total = ref.mask_pixels_per_slice.reduce((s, v) => s + v, 0);
const ok = report.presence_max_abs_diff < 1e-3 && pixDiff / Math.max(total, 1) < 0.005;
console.log(ok ? `PARITY OK (mask pixel diff ${pixDiff}/${total})` : `PARITY FAIL (mask pixel diff ${pixDiff}/${total})`);
process.exit(ok ? 0 : 1);
