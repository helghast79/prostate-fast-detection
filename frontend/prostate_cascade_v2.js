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

async function runBatched(ort, session, inputs, batch = 8) {
  const results = [];
  for (let s = 0; s < inputs.length; s += batch) {
    const chunk = inputs.slice(s, s + batch);
    const data = new Float32Array(chunk.length * SIZE * SIZE);
    chunk.forEach((x, k) => data.set(x, k * SIZE * SIZE));
    const feeds = { input: new ort.Tensor('float32', data, [chunk.length, 1, SIZE, SIZE]) };
    const out = await session.run(feeds);
    for (let k = 0; k < chunk.length; k++) results.push({ out, k });
  }
  return results;
}

// Full pipeline for one series. Returns per-slice Uint8Array masks (rows*cols).
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
  const masks = seg.map(({ out, k }) => {
    const prob = out.prob.data.subarray(k * SIZE * SIZE, (k + 1) * SIZE * SIZE);
    return pasteBack(prob, box, rows, cols, thr);
  });
  return { center, box, localizer: perSlice, masks: keepLargestSliceRun(masks) };
}
