// Prostate cascade v2 — browser reference implementation.
// Line-for-line mirror of src/cascade/v2.py. Works with onnxruntime-web or
// onnxruntime-node: pass the `ort` namespace and two InferenceSessions.
//
// Input per series: every slice of ONE axial T2 series, each slice a
// Float32Array of rows*cols values in DICOM storage order (row-major, no flips),
// plus PixelSpacing [rowSpacingMm, colSpacingMm]. Slices must be sorted by
// ImagePositionPatient (only the final slice-run filter depends on order).

export const SIZE = 192;
export const LO_IDX = 369; // round(0.01 * (192*192 - 1))
export const HI_IDX = 36494; // round(0.99 * (192*192 - 1))

// Resample box [y0, y0+h) x [x0, x0+w) (edge pixel coordinates) to SIZE x SIZE.
// Bilinear, or nearest for masks; samples outside the image are 0.
export function sampleBox(src, H, W, y0, x0, h, w, nearest = false) {
  const out = new Float32Array(SIZE * SIZE);
  const sy = h / SIZE;
  const sx = w / SIZE;
  const at = (r, c) => (r >= 0 && r < H && c >= 0 && c < W ? src[r * W + c] : 0);
  for (let i = 0; i < SIZE; i++) {
    const cy = y0 + (i + 0.5) * sy;
    for (let j = 0; j < SIZE; j++) {
      const cx = x0 + (j + 0.5) * sx;
      let v;
      if (nearest) {
        v = at(Math.floor(cy), Math.floor(cx));
      } else {
        const py = cy - 0.5;
        const px = cx - 0.5;
        const fy = Math.floor(py);
        const fx = Math.floor(px);
        const ty = py - fy;
        const tx = px - fx;
        v =
          at(fy, fx) * (1 - ty) * (1 - tx) +
          at(fy, fx + 1) * (1 - ty) * tx +
          at(fy + 1, fx) * ty * (1 - tx) +
          at(fy + 1, fx + 1) * ty * tx;
      }
      out[i * SIZE + j] = v;
    }
  }
  return out;
}

// Percentile normalisation of the 192x192 input itself → [0, 1].
export function normalize(x) {
  const s = Float32Array.from(x).sort();
  const lo = s[LO_IDX];
  const hi = s[HI_IDX];
  const out = new Float32Array(x.length);
  if (hi - lo < 1e-6) return out;
  const scale = hi - lo;
  for (let k = 0; k < x.length; k++) {
    const v = (x[k] - lo) / scale;
    out[k] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
  return out;
}

export function preprocess(src, H, W, box) {
  return normalize(sampleBox(src, H, W, box[0], box[1], box[2], box[3]));
}

export function cropBox(centerYX, spacingYX, cropMm) {
  const h = cropMm / spacingYX[0];
  const w = cropMm / spacingYX[1];
  return [centerYX[0] - h / 2, centerYX[1] - w / 2, h, w];
}

// Map a SIZE x SIZE probability map back to native H x W (Uint8 0/1).
export function pasteBack(prob, box, H, W, thr = 0.5) {
  const [y0, x0, h, w] = box;
  const out = new Uint8Array(H * W);
  for (let r = 0; r < H; r++) {
    const iy = Math.floor((r + 0.5 - y0) * SIZE / h);
    if (iy < 0 || iy >= SIZE) continue;
    for (let c = 0; c < W; c++) {
      const ix = Math.floor((c + 0.5 - x0) * SIZE / w);
      if (ix < 0 || ix >= SIZE) continue;
      out[r * W + c] = prob[iy * SIZE + ix] > thr ? 1 : 0;
    }
  }
  return out;
}

// Keep the run of consecutive non-empty slices with the most mask pixels.
// Requires masks in spatial (ImagePositionPatient) order.
export function keepLargestSliceRun(masks) {
  const counts = masks.map((m) => m.reduce((s, v) => s + v, 0));
  let best = [0, 0];
  let bestMass = 0;
  let z = 0;
  while (z < counts.length) {
    if (counts[z] > 0) {
      let end = z;
      let mass = 0;
      while (end < counts.length && counts[end] > 0) mass += counts[end++];
      if (mass > bestMass) { best = [z, end]; bestMass = mass; }
      z = end;
    } else {
      z++;
    }
  }
  return masks.map((m, k) => (k >= best[0] && k < best[1] ? m : new Uint8Array(m.length)));
}

async function runBatched(ort, session, inputs, channels = 1, batch = 8) {
  const results = [];
  const n = channels * SIZE * SIZE;
  for (let s = 0; s < inputs.length; s += batch) {
    const chunk = inputs.slice(s, s + batch);
    const data = new Float32Array(chunk.length * n);
    chunk.forEach((x, k) => data.set(x, k * n));
    const feeds = { input: new ort.Tensor('float32', data, [chunk.length, channels, SIZE, SIZE]) };
    const out = await session.run(feeds);
    for (let k = 0; k < chunk.length; k++) results.push({ out, k });
  }
  return results;
}

// Binary mask from a detection head output for one batch item.
// v2: [1,H,W] sigmoid → threshold. v3: [3,H,W] softmax → argmax, class>0 = prostate.
function detectionMask(probTensor, batchIndex, box, rows, cols, thr = 0.5) {
  const dims = probTensor.dims; // [B,C,S,S]
  const C = dims[1];
  const plane = SIZE * SIZE;
  const base = batchIndex * C * plane;
  if (C === 1) {
    const prob = probTensor.data.subarray(base, base + plane);
    return pasteBack(prob, box, rows, cols, thr);
  }
  // Softmax classes: 0=bg, 1=gland, 2=PZ → binary prostate = argmax > 0.
  const classes = new Float32Array(plane);
  for (let i = 0; i < plane; i++) {
    let best = 0;
    let bestV = -Infinity;
    for (let c = 0; c < C; c++) {
      const v = probTensor.data[base + c * plane + i];
      if (v > bestV) { bestV = v; best = c; }
    }
    classes[i] = best > 0 ? 1 : 0;
  }
  return pasteBack(classes, box, rows, cols, 0.5);
}

// Full pipeline for one series. Returns per-slice Uint8Array masks (rows*cols).
// Works with cascade v2 (binary seg) and v3 (3-class softmax detection).
export async function segmentSeries(ort, locSession, segSession, slices, rows, cols, spacingYX,
                                    cropMm, thr = 0.5) {
  const full = [0, 0, rows, cols];
  const locIn = slices.map((s) => preprocess(s, rows, cols, full));
  const loc = await runBatched(ort, locSession, locIn);

  let m = 0, my = 0, mx = 0;
  const perSlice = loc.map(({ out, k }) => {
    const presence = out.presence.data[k];
    const mass = out.mass.data[k];
    const cy = out.center_yx.data[2 * k];
    const cx = out.center_yx.data[2 * k + 1];
    m += mass; my += mass * cy; mx += mass * cx;
    return { presence, mass, cy, cx };
  });
  if (m <= 0) return { center: null, box: null, localizer: perSlice, masks: slices.map(() => new Uint8Array(rows * cols)) };
  const center = [(my / m) * rows, (mx / m) * cols];
  const box = cropBox(center, spacingYX, cropMm);

  const segIn = slices.map((s) => preprocess(s, rows, cols, box));
  const seg = await runBatched(ort, segSession, segIn);
  const masks = seg.map(({ out, k }) => detectionMask(out.prob, k, box, rows, cols, thr));
  return { center, box, localizer: perSlice, masks: keepLargestSliceRun(masks) };
}

// ---------------------------------------------------------------------------
// Lesion model (T2 + ADC + DWI). Mirrors src/cascade/lesion.py.
//
// A "series" here is { slices: Float32Array[] (IPP-sorted), ipp: [[x,y,z],...],
// iop: [6 numbers], spacing: [rowSpacing, colSpacing], rows, cols }.
// ---------------------------------------------------------------------------

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

function seriesNormal(iop) {
  const r = iop.slice(0, 3);
  const c = iop.slice(3, 6);
  const n = [r[1] * c[2] - r[2] * c[1], r[2] * c[0] - r[0] * c[2], r[0] * c[1] - r[1] * c[0]];
  const len = Math.hypot(n[0], n[1], n[2]);
  return [n[0] / len, n[1] / len, n[2] / len];
}

// Multi-b DWI: one image per slice position, the darkest (= highest b-value).
// `images`: [{ pixels: Float32Array, ipp: [x,y,z] }], iop shared. Returns IPP-sorted.
export function keepLowestSignalPerPosition(images, iop) {
  const n = seriesNormal(iop);
  const mean = (p) => p.reduce((s, v) => s + v, 0) / p.length;
  const groups = new Map();
  for (const im of images) {
    const key = (Math.round(dot(im.ipp, n) * 100) / 100).toFixed(2);
    const prev = groups.get(key);
    if (!prev || mean(im.pixels) < prev.m) groups.set(key, { im, m: mean(im.pixels) });
  }
  return [...groups.values()].map((g) => g.im).sort((a, b) => dot(a.ipp, n) - dot(b.ipp, n));
}

// Patient-space positions of the SIZE x SIZE box grid on T2 slice z.
export function gridPoints(t2, z, box) {
  const [y0, x0, h, w] = box;
  const rd = t2.iop.slice(0, 3);
  const cd = t2.iop.slice(3, 6);
  const o = t2.ipp[z];
  const pts = new Float64Array(SIZE * SIZE * 3);
  for (let i = 0; i < SIZE; i++) {
    const r = y0 + (i + 0.5) * (h / SIZE) - 0.5;
    for (let j = 0; j < SIZE; j++) {
      const c = x0 + (j + 0.5) * (w / SIZE) - 0.5;
      const k = (i * SIZE + j) * 3;
      for (let a = 0; a < 3; a++) {
        pts[k + a] = o[a] + c * t2.spacing[1] * rd[a] + r * t2.spacing[0] * cd[a];
      }
    }
  }
  return pts;
}

// Sample a series at patient points: bilinear in-plane, linear between slices,
// 0 outside the image or more than half a slice gap beyond the first/last slice.
export function sampleSeries(s, pts) {
  const Z = s.slices.length;
  const H = s.rows;
  const W = s.cols;
  const n = seriesNormal(s.iop);
  const rd = s.iop.slice(0, 3);
  const cd = s.iop.slice(3, 6);
  const pos = s.ipp.map((p) => dot(p, n));
  const out = new Float32Array(pts.length / 3);

  const bilinear = (k, p) => {
    const o = s.ipp[k];
    const d = [p[0] - o[0], p[1] - o[1], p[2] - o[2]];
    const col = dot(d, rd) / s.spacing[1];
    const row = dot(d, cd) / s.spacing[0];
    const fr = Math.floor(row);
    const fc = Math.floor(col);
    const tr = row - fr;
    const tc = col - fc;
    const px = s.slices[k];
    const at = (r, c) => (r >= 0 && r < H && c >= 0 && c < W ? px[r * W + c] : 0);
    return at(fr, fc) * (1 - tr) * (1 - tc) + at(fr, fc + 1) * (1 - tr) * tc +
      at(fr + 1, fc) * tr * (1 - tc) + at(fr + 1, fc + 1) * tr * tc;
  };

  for (let q = 0; q < out.length; q++) {
    const p = [pts[3 * q], pts[3 * q + 1], pts[3 * q + 2]];
    const wp = dot(p, n);
    let k;
    let k1;
    let t;
    let valid;
    if (Z === 1) {
      k = 0; k1 = 0; t = 0;
      valid = Math.abs(wp - pos[0]) <= 2.0;
    } else {
      let cnt = 0;
      while (cnt < Z && pos[cnt] <= wp) cnt++;
      k = Math.min(Math.max(cnt - 1, 0), Z - 2);
      k1 = k + 1;
      t = (wp - pos[k]) / (pos[k1] - pos[k]);
      valid = wp >= pos[0] - 0.5 * (pos[1] - pos[0]) && wp <= pos[Z - 1] + 0.5 * (pos[Z - 1] - pos[Z - 2]);
      t = t < 0 ? 0 : t > 1 ? 1 : t;
    }
    out[q] = valid ? bilinear(k, p) * (1 - t) + bilinear(k1, p) * t : 0;
  }
  return out;
}

// 3-channel input [T2, ADC, DWI] for T2 slice z, each channel normalised on its own.
export function lesionInput(t2, adc, dwi, z, box) {
  const pts = gridPoints(t2, z, box);
  const x = new Float32Array(3 * SIZE * SIZE);
  x.set(normalize(sampleBox(t2.slices[z], t2.rows, t2.cols, box[0], box[1], box[2], box[3])), 0);
  x.set(normalize(sampleSeries(adc, pts)), SIZE * SIZE);
  x.set(normalize(sampleSeries(dwi, pts)), 2 * SIZE * SIZE);
  return x;
}

// Keep lesion pixels within a (2r+1)x(2r+1) square of the prostate mask (same slice).
export function restrictToProstate(lesion, prostate, rows, cols, r = 3) {
  const tmp = new Uint8Array(rows * cols);
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      let v = 0;
      for (let dx = Math.max(0, x - r); dx <= Math.min(cols - 1, x + r) && !v; dx++) v = prostate[y * cols + dx];
      tmp[y * cols + x] = v;
    }
  }
  const out = new Uint8Array(rows * cols);
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      if (!lesion[y * cols + x]) continue;
      let v = 0;
      for (let dy = Math.max(0, y - r); dy <= Math.min(rows - 1, y + r) && !v; dy++) v = tmp[dy * cols + x];
      out[y * cols + x] = v;
    }
  }
  return out;
}

// Lesion masks for every T2 slice. `prostate` is the segmentSeries() result
// for the same T2 series (its crop box and masks are reused).
export async function detectLesions(ort, lesionSession, t2, adc, dwi, prostate, thr = 0.5) {
  if (!prostate.box) return t2.slices.map(() => new Uint8Array(t2.rows * t2.cols));
  const inputs = t2.slices.map((_, z) => lesionInput(t2, adc, dwi, z, prostate.box));
  const res = await runBatched(ort, lesionSession, inputs, 3);
  return res.map(({ out, k }, z) => {
    const prob = out.prob.data.subarray(k * SIZE * SIZE, (k + 1) * SIZE * SIZE);
    const m = pasteBack(prob, prostate.box, t2.rows, t2.cols, thr);
    return restrictToProstate(m, prostate.masks[z], t2.rows, t2.cols);
  });
}
