/**
 * The three compression schemes from code/compression.py, ported to run in
 * the browser on a grayscale image (values 0–255, row-major).
 *
 *  DWT  pywt.wavedec2(img, "bior4.4"), keep the kf largest |coefficients|
 *       (hard threshold), pywt.waverec2.  CR = mn / kf.
 *  DFT  fft2, keep the kf largest |coefficients|, real(ifft2).  CR = mn / kf.
 *  PCA  sklearn PCA(n_components=k).fit_transform(img) → inverse_transform,
 *       i.e. column-centred rank-k approximation.  CR = mn / (k (m + n)).
 *
 *  kf = int(#coefficients / ratio), k = max(1, int(min(m, n, mn/(m+n)) / ratio)).
 */
import { fft2 } from "./fft.ts";
import { bands, coeffArrayShape, toArray, wavedec2, waverec2, type WaveDec } from "./dwt.ts";
import { symEig } from "./eig.ts";
import { energy, psnrFromMse, sse } from "./metrics.ts";

export type Method = "DWT" | "DFT" | "PCA";
export const METHODS: Method[] = ["DWT", "DFT", "PCA"];

export interface Reconstruction {
  method: Method;
  ratio: number;
  data: Float64Array;
  /** compression ratio as reported by the repo */
  cr: number;
  /** coefficients (or components) kept / available */
  kept: number;
  total: number;
  mse: number;
  psnr: number;
  relErr: number;
}

export interface CurvePoint {
  cr: number;
  psnr: number;
  relErr: number;
  kept: number;
}

const sortDesc = (a: Float64Array) => a.sort().reverse();

export class Compressor {
  readonly rows: number;
  readonly cols: number;
  readonly img: Float64Array;
  readonly norm2: number;

  // DWT
  dec!: WaveDec;
  dwtTotal = 0; // len(coeffs_to_array(...).flatten())
  dwtSorted!: Float64Array;

  // DFT
  specRe!: Float64Array;
  specIm!: Float64Array;
  dftMag!: Float64Array;
  dftSorted!: Float64Array;

  // PCA
  mean!: Float64Array; // per column
  eigVals!: Float64Array;
  /** basis vectors (columns) of the space the eigendecomposition lives in */
  eigVecs!: Float64Array;
  /** true → eigendecomposition of XcᵀXc (cols×cols), false → XcXcᵀ (rows×rows) */
  pcaOnCols = true;
  pcaMaxK = 1;

  constructor(img: Float64Array, rows: number, cols: number) {
    this.img = img;
    this.rows = rows;
    this.cols = cols;
    this.norm2 = energy(img);
  }

  get size() {
    return this.rows * this.cols;
  }

  prepareDWT() {
    this.dec = wavedec2(this.img, this.rows, this.cols);
    const [r, c] = coeffArrayShape(this.dec);
    this.dwtTotal = r * c;
    const bs = bands(this.dec);
    const n = bs.reduce((s, b) => s + b.data.length, 0);
    const mags = new Float64Array(n);
    let o = 0;
    for (const b of bs) for (let i = 0; i < b.data.length; i++) mags[o++] = Math.abs(b.data[i]);
    this.dwtSorted = sortDesc(mags);
  }

  prepareDFT() {
    const re = Float64Array.from(this.img);
    const im = new Float64Array(this.size);
    fft2(re, im, this.rows, this.cols);
    // The image is real, so F(-k) = conj(F(k)). Enforce it exactly so that
    // conjugate pairs have bit-identical magnitudes and are kept or dropped
    // together by the threshold, as they are with NumPy.
    const { rows, cols } = this;
    for (let r = 0; r < rows; r++) {
      const r2 = (rows - r) % rows;
      for (let c = 0; c < cols; c++) {
        const c2 = (cols - c) % cols;
        const i = r * cols + c;
        const j = r2 * cols + c2;
        if (j < i) continue;
        const ar = (re[i] + re[j]) / 2;
        const ai = (im[i] - im[j]) / 2;
        re[i] = ar;
        im[i] = ai;
        re[j] = ar;
        im[j] = i === j ? 0 : -ai;
      }
    }
    this.specRe = re;
    this.specIm = im;
    const mag = new Float64Array(this.size);
    for (let i = 0; i < mag.length; i++) mag[i] = Math.hypot(re[i], im[i]);
    this.dftMag = mag;
    this.dftSorted = sortDesc(Float64Array.from(mag));
  }

  preparePCA() {
    const { rows: m, cols: n, img } = this;
    const mean = new Float64Array(n);
    for (let r = 0; r < m; r++) for (let c = 0; c < n; c++) mean[c] += img[r * n + c];
    for (let c = 0; c < n; c++) mean[c] /= m;
    const Xc = new Float64Array(m * n);
    for (let r = 0; r < m; r++) for (let c = 0; c < n; c++) Xc[r * n + c] = img[r * n + c] - mean[c];
    this.mean = mean;
    this.pcaOnCols = n <= m;
    const d = Math.min(m, n);
    const G = new Float64Array(d * d);
    if (this.pcaOnCols) {
      // XcᵀXc
      for (let r = 0; r < m; r++) {
        const row = Xc.subarray(r * n, r * n + n);
        for (let i = 0; i < n; i++) {
          const v = row[i];
          if (v === 0) continue;
          const gi = i * n;
          for (let j = i; j < n; j++) G[gi + j] += v * row[j];
        }
      }
    } else {
      for (let i = 0; i < m; i++) {
        const ri = Xc.subarray(i * n, i * n + n);
        for (let j = i; j < m; j++) {
          const rj = Xc.subarray(j * n, j * n + n);
          let s = 0;
          for (let c = 0; c < n; c++) s += ri[c] * rj[c];
          G[i * m + j] = s;
        }
      }
    }
    for (let i = 0; i < d; i++) for (let j = 0; j < i; j++) G[i * d + j] = G[j * d + i];
    const { values, vectors } = symEig(G, d);
    for (let i = 0; i < d; i++) if (values[i] < 0) values[i] = 0;
    this.eigVals = values;
    this.eigVecs = vectors;
    this.pcaMaxK = d;
    this.xc = Xc;
  }

  private xc!: Float64Array;

  prepare() {
    this.prepareDWT();
    this.prepareDFT();
    this.preparePCA();
  }

  // ---------- coefficient counts ----------

  dwtKf(ratio: number) {
    return Math.max(1, Math.floor(this.dwtTotal / ratio));
  }

  dftKf(ratio: number) {
    return Math.max(1, Math.floor(this.size / ratio));
  }

  pcaK(ratio: number) {
    const { rows: h, cols: w } = this;
    return Math.max(1, Math.floor(Math.min(h, w, (h * w) / (h + w)) / ratio));
  }

  /** np.partition(|flat|, -kf)[-kf], i.e. the kf-th largest magnitude */
  private static thresh(sorted: Float64Array, kf: number) {
    return kf - 1 < sorted.length ? sorted[kf - 1] : 0;
  }

  dwtThreshold(ratio: number) {
    return Compressor.thresh(this.dwtSorted, this.dwtKf(ratio));
  }

  dftThreshold(ratio: number) {
    return Compressor.thresh(this.dftSorted, this.dftKf(ratio));
  }

  // ---------- reconstructions ----------

  private finish(method: Method, ratio: number, data: Float64Array, cr: number, kept: number, total: number): Reconstruction {
    const e = sse(this.img, data);
    const mse = e / this.size;
    return {
      method,
      ratio,
      data,
      cr,
      kept,
      total,
      mse,
      psnr: psnrFromMse(mse),
      relErr: Math.sqrt(e / this.norm2),
    };
  }

  reconstructDWT(ratio: number): Reconstruction {
    const kf = this.dwtKf(ratio);
    const t = Compressor.thresh(this.dwtSorted, kf);
    const thr = (b: { rows: number; cols: number; data: Float64Array }) => {
      const out = new Float64Array(b.data.length);
      for (let i = 0; i < out.length; i++) {
        const v = b.data[i];
        out[i] = Math.abs(v) >= t ? v : 0;
      }
      return { rows: b.rows, cols: b.cols, data: out };
    };
    const dec: WaveDec = {
      rows: this.dec.rows,
      cols: this.dec.cols,
      approx: thr(this.dec.approx),
      levels: this.dec.levels.map((lv) => ({ h: thr(lv.h), v: thr(lv.v), d: thr(lv.d) })),
    };
    const data = waverec2(dec);
    return this.finish("DWT", ratio, data, this.size / kf, kf, this.dwtTotal);
  }

  reconstructDFT(ratio: number): Reconstruction {
    const kf = this.dftKf(ratio);
    const t = Compressor.thresh(this.dftSorted, kf);
    const re = new Float64Array(this.size);
    const im = new Float64Array(this.size);
    for (let i = 0; i < re.length; i++) {
      if (this.dftMag[i] >= t) {
        re[i] = this.specRe[i];
        im[i] = this.specIm[i];
      }
    }
    fft2(re, im, this.rows, this.cols, true);
    return this.finish("DFT", ratio, re, this.size / kf, kf, this.size);
  }

  reconstructPCA(ratio: number): Reconstruction {
    const k = this.pcaK(ratio);
    const { rows: m, cols: n, mean, xc } = this;
    const d = this.pcaMaxK;
    const E = this.eigVecs;
    const out = new Float64Array(m * n);
    if (this.pcaOnCols) {
      // out = mean + Xc V_k V_kᵀ
      const proj = new Float64Array(k);
      for (let r = 0; r < m; r++) {
        const row = xc.subarray(r * n, r * n + n);
        proj.fill(0);
        for (let c = 0; c < n; c++) {
          const v = row[c];
          const base = c * d;
          for (let j = 0; j < k; j++) proj[j] += v * E[base + j];
        }
        const o = r * n;
        for (let c = 0; c < n; c++) {
          const base = c * d;
          let s = mean[c];
          for (let j = 0; j < k; j++) s += proj[j] * E[base + j];
          out[o + c] = s;
        }
      }
    } else {
      // out = mean + U_k U_kᵀ Xc
      const P = new Float64Array(k * n); // U_kᵀ Xc
      for (let r = 0; r < m; r++) {
        const row = xc.subarray(r * n, r * n + n);
        for (let j = 0; j < k; j++) {
          const u = E[r * d + j];
          if (u === 0) continue;
          const pj = j * n;
          for (let c = 0; c < n; c++) P[pj + c] += u * row[c];
        }
      }
      for (let r = 0; r < m; r++) {
        const o = r * n;
        for (let c = 0; c < n; c++) out[o + c] = mean[c];
        for (let j = 0; j < k; j++) {
          const u = E[r * d + j];
          const pj = j * n;
          for (let c = 0; c < n; c++) out[o + c] += u * P[pj + c];
        }
      }
    }
    return this.finish("PCA", ratio, out, (m * n) / (k * (m + n)), k, d);
  }

  reconstruct(method: Method, ratio: number): Reconstruction {
    if (method === "DWT") return this.reconstructDWT(ratio);
    if (method === "DFT") return this.reconstructDFT(ratio);
    return this.reconstructPCA(ratio);
  }

  // ---------- error curves ----------

  private point(cr: number, sseVal: number, kept: number): CurvePoint {
    return {
      cr,
      psnr: psnrFromMse(sseVal / this.size),
      relErr: Math.sqrt(sseVal / this.norm2),
      kept,
    };
  }

  /**
   * DFT error via Parseval: dropping coefficient F_k adds |F_k|²/(mn) to the
   * squared error. Thresholding by magnitude keeps conjugate pairs together,
   * so the reconstruction is real and this is exact up to rounding.
   */
  dftCurve(ratios: number[]): CurvePoint[] {
    const s = this.dftSorted;
    const tail = new Float64Array(s.length + 1);
    for (let i = s.length - 1; i >= 0; i--) tail[i] = tail[i + 1] + s[i] * s[i];
    return ratios.map((r) => {
      const kf = this.dftKf(r);
      // coefficients tied with the threshold (conjugate partners) are kept too
      const t = s[kf - 1];
      let j = kf;
      while (j < s.length && s[j] >= t) j++;
      return this.point(this.size / kf, tail[j] / this.size, kf);
    });
  }

  /** PCA error: squared residual of a rank-k centred approximation = Σ_{i>k} λ_i. */
  pcaCurve(ratios: number[]): CurvePoint[] {
    const v = this.eigVals;
    const tail = new Float64Array(v.length + 1);
    for (let i = v.length - 1; i >= 0; i--) tail[i] = tail[i + 1] + v[i];
    const seen = new Set<number>();
    const out: CurvePoint[] = [];
    const { rows: m, cols: n } = this;
    for (const r of ratios) {
      const k = this.pcaK(r);
      if (seen.has(k)) continue;
      seen.add(k);
      out.push(this.point((m * n) / (k * (m + n)), tail[k], k));
    }
    return out;
  }

  dwtPoint(ratio: number): CurvePoint {
    const rec = this.reconstructDWT(ratio);
    return { cr: rec.cr, psnr: rec.psnr, relErr: rec.relErr, kept: rec.kept };
  }

  // ---------- coefficient maps for display ----------

  /** log-magnitude of DWT coefficients in coeffs_to_array layout */
  dwtMap() {
    const bs = bands(this.dec);
    return toArray(this.dec, (b, i) => Math.log1p(Math.abs(bs[b].data[i])));
  }

  /** fftshifted log-magnitude spectrum */
  dftMap() {
    const { rows, cols } = this;
    const data = new Float32Array(rows * cols);
    const hr = Math.floor(rows / 2);
    const hc = Math.floor(cols / 2);
    for (let r = 0; r < rows; r++) {
      const sr = (r + hr) % rows;
      for (let c = 0; c < cols; c++) {
        const sc = (c + hc) % cols;
        data[sr * cols + sc] = Math.log1p(this.dftMag[r * cols + c]);
      }
    }
    return { rows, cols, data };
  }

  /** singular values σ_i = sqrt(λ_i) of the centred image */
  singularValues() {
    return Float64Array.from(this.eigVals, (v) => Math.sqrt(v));
  }
}

export function logspace(a: number, b: number, n: number): number[] {
  return Array.from({ length: n }, (_, i) => Math.pow(10, a + ((b - a) * i) / (n - 1)));
}
