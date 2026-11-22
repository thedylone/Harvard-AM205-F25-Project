/**
 * Multilevel 2D discrete wavelet transform matching PyWavelets'
 * `wavedec2` / `waverec2` with the default "symmetric" signal extension,
 * as used by `dwt_compression` in code/compression.py.
 */

/** bior4.4 filters, exactly as reported by pywt.Wavelet("bior4.4"). */
export const BIOR44 = {
  decLo: [
    0.0, 0.03782845550726404, -0.023849465019556843, -0.11062440441843718,
    0.37740285561283066, 0.8526986790088938, 0.37740285561283066,
    -0.11062440441843718, -0.023849465019556843, 0.03782845550726404,
  ],
  decHi: [
    -0.0, -0.06453888262869706, 0.04068941760916406, 0.41809227322161724,
    -0.7884856164055829, 0.41809227322161724, 0.04068941760916406,
    -0.06453888262869706, -0.0, 0.0,
  ],
  recLo: [
    0.0, -0.06453888262869706, -0.04068941760916406, 0.41809227322161724,
    0.7884856164055829, 0.41809227322161724, -0.04068941760916406,
    -0.06453888262869706, 0.0, 0.0,
  ],
  recHi: [
    0.0, -0.03782845550726404, -0.023849465019556843, 0.11062440441843718,
    0.37740285561283066, -0.8526986790088938, 0.37740285561283066,
    0.11062440441843718, -0.023849465019556843, -0.03782845550726404,
  ],
};

export type Filters = typeof BIOR44;

const F = BIOR44.decLo.length;

/** Symmetric (half-sample) extension index, as in pywt mode "symmetric". */
function sym(idx: number, n: number): number {
  const p = 2 * n;
  let k = idx % p;
  if (k < 0) k += p;
  return k >= n ? p - 1 - k : k;
}

export const dwtLen = (n: number) => Math.floor((n + F - 1) / 2);
export const idwtLen = (l: number) => 2 * l - F + 2;

/** pywt.dwt_max_level(min(shape), filter_len) */
export function maxLevel(minDim: number): number {
  if (minDim < F - 1) return 0;
  return Math.floor(Math.log2(minDim / (F - 1)));
}

/** One 1D analysis step on a strided signal. */
function analyze(
  x: Float64Array, xOff: number, xStride: number, n: number,
  lo: Float64Array, hi: Float64Array, oOff: number, oStride: number,
  f: Filters,
): void {
  const out = dwtLen(n);
  const dl = f.decLo;
  const dh = f.decHi;
  for (let o = 0; o < out; o++) {
    const i = 2 * o + 1;
    let sl = 0;
    let sh = 0;
    for (let j = 0; j < F; j++) {
      const idx = i - j;
      const v = x[xOff + xStride * (idx >= 0 && idx < n ? idx : sym(idx, n))];
      sl += dl[j] * v;
      sh += dh[j] * v;
    }
    lo[oOff + oStride * o] = sl;
    hi[oOff + oStride * o] = sh;
  }
}

/** One 1D synthesis step: y = idwt(a, d), output length 2L - F + 2. */
function synthesize(
  a: Float64Array, d: Float64Array, iOff: number, iStride: number, l: number,
  y: Float64Array, yOff: number, yStride: number,
  f: Filters,
): void {
  const out = idwtLen(l);
  const gl = f.recLo;
  const gh = f.recHi;
  for (let n = 0; n < out; n++) {
    // y[n] = sum_k a[k] g[n + F - 2 - 2k]
    const t = n + F - 2;
    let kMin = Math.ceil((t - (F - 1)) / 2);
    if (kMin < 0) kMin = 0;
    let kMax = Math.floor(t / 2);
    if (kMax > l - 1) kMax = l - 1;
    let s = 0;
    for (let k = kMin; k <= kMax; k++) {
      const g = t - 2 * k;
      const p = iOff + iStride * k;
      s += a[p] * gl[g] + d[p] * gh[g];
    }
    y[yOff + yStride * n] = s;
  }
}

export interface Band {
  rows: number;
  cols: number;
  data: Float64Array;
}

export interface Level {
  /** detail bands: [detail along rows-axis, along cols-axis, diagonal] */
  h: Band;
  v: Band;
  d: Band;
}

export interface WaveDec {
  rows: number;
  cols: number;
  approx: Band;
  /** coarsest first, like pywt (coeffs[1:]) */
  levels: Level[];
}

function dwt2(x: Band, f: Filters): { a: Band; h: Band; v: Band; d: Band } {
  const { rows, cols, data } = x;
  const cw = dwtLen(cols);
  const rh = dwtLen(rows);
  // along columns axis (each row)
  const L = new Float64Array(rows * cw);
  const H = new Float64Array(rows * cw);
  for (let r = 0; r < rows; r++) analyze(data, r * cols, 1, cols, L, H, r * cw, 1, f);
  const a = new Float64Array(rh * cw);
  const h = new Float64Array(rh * cw);
  const v = new Float64Array(rh * cw);
  const d = new Float64Array(rh * cw);
  for (let c = 0; c < cw; c++) {
    analyze(L, c, cw, rows, a, h, c, cw, f);
    analyze(H, c, cw, rows, v, d, c, cw, f);
  }
  const b = (data: Float64Array): Band => ({ rows: rh, cols: cw, data });
  return { a: b(a), h: b(h), v: b(v), d: b(d) };
}

function idwt2(a: Band, lv: Level, f: Filters): Band {
  const rh = lv.h.rows;
  const cw = lv.h.cols;
  // pywt trims the approximation to the detail shape if it is one larger
  let ad = a.data;
  if (a.rows !== rh || a.cols !== cw) {
    ad = new Float64Array(rh * cw);
    for (let r = 0; r < rh; r++) ad.set(a.data.subarray(r * a.cols, r * a.cols + cw), r * cw);
  }
  const rows = idwtLen(rh);
  const cols = idwtLen(cw);
  const L = new Float64Array(rows * cw);
  const H = new Float64Array(rows * cw);
  for (let c = 0; c < cw; c++) {
    synthesize(ad, lv.h.data, c, cw, rh, L, c, cw, f);
    synthesize(lv.v.data, lv.d.data, c, cw, rh, H, c, cw, f);
  }
  const out = new Float64Array(rows * cols);
  for (let r = 0; r < rows; r++) synthesize(L, H, r * cw, 1, cw, out, r * cols, 1, f);
  return { rows, cols, data: out };
}

export function wavedec2(img: Float64Array, rows: number, cols: number, f: Filters = BIOR44): WaveDec {
  const level = maxLevel(Math.min(rows, cols));
  let cur: Band = { rows, cols, data: img };
  const levels: Level[] = [];
  for (let i = 0; i < level; i++) {
    const { a, h, v, d } = dwt2(cur, f);
    levels.unshift({ h, v, d });
    cur = a;
  }
  return { rows, cols, approx: cur, levels };
}

export function waverec2(dec: WaveDec, f: Filters = BIOR44): Float64Array {
  let a = dec.approx;
  for (const lv of dec.levels) a = idwt2(a, lv, f);
  if (a.rows === dec.rows && a.cols === dec.cols) return a.data;
  const out = new Float64Array(dec.rows * dec.cols);
  for (let r = 0; r < dec.rows; r++) {
    out.set(a.data.subarray(r * a.cols, r * a.cols + dec.cols), r * dec.cols);
  }
  return out;
}

/** All bands in pywt order: approx, then (h, v, d) per level coarse→fine. */
export function bands(dec: WaveDec): Band[] {
  const out: Band[] = [dec.approx];
  for (const lv of dec.levels) out.push(lv.h, lv.v, lv.d);
  return out;
}

/** Shape of pywt.coeffs_to_array(coeffs) — includes zero padding. */
export function coeffArrayShape(dec: WaveDec): [number, number] {
  let r = dec.approx.rows;
  let c = dec.approx.cols;
  for (const lv of dec.levels) {
    r += lv.h.rows;
    c += lv.h.cols;
  }
  return [r, c];
}

/** Map a decomposition onto the coeffs_to_array layout (for display). */
export function toArray(dec: WaveDec, value: (band: number, i: number) => number): {
  rows: number;
  cols: number;
  data: Float32Array;
  mask: Uint8Array;
} {
  const [rows, cols] = coeffArrayShape(dec);
  const data = new Float32Array(rows * cols);
  const mask = new Uint8Array(rows * cols); // 1 where a real coefficient lives
  const put = (b: Band, bi: number, r0: number, c0: number) => {
    for (let r = 0; r < b.rows; r++) {
      for (let c = 0; c < b.cols; c++) {
        const p = (r0 + r) * cols + c0 + c;
        data[p] = value(bi, r * b.cols + c);
        mask[p] = 1;
      }
    }
  };
  put(dec.approx, 0, 0, 0);
  let r0 = dec.approx.rows;
  let c0 = dec.approx.cols;
  let bi = 1;
  for (const lv of dec.levels) {
    put(lv.h, bi, r0, 0); // pywt: 'da' bottom-left
    put(lv.v, bi + 1, 0, c0); // 'ad' top-right
    put(lv.d, bi + 2, r0, c0);
    r0 += lv.h.rows;
    c0 += lv.h.cols;
    bi += 3;
  }
  return { rows, cols, data, mask };
}
