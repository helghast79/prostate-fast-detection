// Node parity test for the lesion path: prostate cascade → lesion model, vs Python.
// Usage: node frontend/test_lesion_parity.mjs <cascade export dir> <lesion export dir> [<onnxruntime-node path>]
import { readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { createRequire } from 'node:module';
import { detectLesions, lesionInput, segmentSeries } from './prostate_cascade_v2.js';

const [cascadeDir, lesionDir, ortPath] = process.argv.slice(2);
const ort = createRequire(import.meta.url)(ortPath || 'onnxruntime-node');
const casPath = existsSync(join(cascadeDir, 'cascade_v3.json'))
  ? join(cascadeDir, 'cascade_v3.json')
  : join(cascadeDir, 'cascade_v2.json');
const lesPath = existsSync(join(lesionDir, 'lesion_v3.json'))
  ? join(lesionDir, 'lesion_v3.json')
  : join(lesionDir, 'lesion_v1.json');
const cas = JSON.parse(readFileSync(casPath, 'utf8'));
const les = JSON.parse(readFileSync(lesPath, 'utf8'));
const parDir = join(lesionDir, 'parity');
const ref = JSON.parse(readFileSync(join(parDir, 'lesion_parity.json'), 'utf8'));

function loadSeries(meta) {
  const [Z, H, W] = meta.shape_zyx;
  const buf = readFileSync(join(parDir, meta.file));
  const vol = new Float32Array(buf.buffer, buf.byteOffset, Z * H * W);
  return {
    slices: Array.from({ length: Z }, (_, z) => vol.subarray(z * H * W, (z + 1) * H * W)),
    ipp: meta.ipp, iop: meta.iop, spacing: meta.spacing, rows: H, cols: W,
  };
}
const t2 = loadSeries(ref.series.t2);
const adc = loadSeries(ref.series.adc);
const dwi = loadSeries(ref.series.dwi);

const detModel = cas.detection_model || cas.segmenter_model;
const loc = await ort.InferenceSession.create(join(cascadeDir, cas.localizer_model));
const seg = await ort.InferenceSession.create(join(cascadeDir, detModel));
const lesion = await ort.InferenceSession.create(join(lesionDir, les.model));

const prostate = await segmentSeries(ort, loc, seg, t2.slices, t2.rows, t2.cols, t2.spacing, cas.crop_mm);
const masks = await detectLesions(ort, lesion, t2, adc, dwi, prostate);

const sum = (a) => a.reduce((s, v) => s + v, 0);
const x = lesionInput(t2, adc, dwi, ref.check_slice, prostate.box);
const n = 192 * 192;
const chSums = [0, 1, 2].map((c) => sum(x.subarray(c * n, (c + 1) * n)));
const jsLesion = masks.map(sum);
const jsProstate = prostate.masks.map(sum);
console.log(JSON.stringify({
  box: [prostate.box, ref.prostate_crop_box],
  channel_sums: [chSums, ref.lesion_input_channel_sums],
  prostate_pixels: [sum(jsProstate), sum(ref.prostate_pixels_per_slice)],
  lesion_pixels_js: jsLesion,
  lesion_pixels_py: ref.lesion_pixels_per_slice,
}));
const diff = jsLesion.reduce((s, v, i) => s + Math.abs(v - ref.lesion_pixels_per_slice[i]), 0);
const total = Math.max(sum(ref.lesion_pixels_per_slice), 1);
const chOk = chSums.every((v, c) => Math.abs(v - ref.lesion_input_channel_sums[c]) < 1e-2 * Math.max(1, Math.abs(v)));
const ok = chOk && diff / total < 0.005;
console.log(ok ? `LESION PARITY OK (pixel diff ${diff}/${total})` : `LESION PARITY FAIL (pixel diff ${diff}/${total}, channels ok=${chOk})`);
process.exit(ok ? 0 : 1);
