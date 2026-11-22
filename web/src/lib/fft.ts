/**
 * Complex FFT for arbitrary lengths: iterative radix-2 for powers of two and
 * Bluestein's chirp-z algorithm for everything else. Unnormalised forward
 * transform, 1/N on the inverse (NumPy convention).
 */

interface Radix2Plan {
  n: number;
  levels: number;
  cos: Float64Array;
  sin: Float64Array;
  rev: Uint32Array;
}

function makeRadix2(n: number): Radix2Plan {
  const levels = Math.round(Math.log2(n));
  const cos = new Float64Array(n / 2);
  const sin = new Float64Array(n / 2);
  for (let i = 0; i < n / 2; i++) {
    cos[i] = Math.cos((2 * Math.PI * i) / n);
    sin[i] = Math.sin((2 * Math.PI * i) / n);
  }
  const rev = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    let r = 0;
    let x = i;
    for (let b = 0; b < levels; b++) {
      r = (r << 1) | (x & 1);
      x >>>= 1;
    }
    rev[i] = r;
  }
  return { n, levels, cos, sin, rev };
}

/** In-place forward radix-2 FFT (e^{-i...}). */
function radix2(p: Radix2Plan, re: Float64Array, im: Float64Array): void {
  const { n, cos, sin, rev } = p;
  for (let i = 0; i < n; i++) {
    const j = rev[i];
    if (j > i) {
      let t = re[i];
      re[i] = re[j];
      re[j] = t;
      t = im[i];
      im[i] = im[j];
      im[j] = t;
    }
  }
  for (let size = 2; size <= n; size *= 2) {
    const half = size / 2;
    const step = n / size;
    for (let i = 0; i < n; i += size) {
      for (let j = i, k = 0; j < i + half; j++, k += step) {
        const l = j + half;
        const tre = re[l] * cos[k] + im[l] * sin[k];
        const tim = -re[l] * sin[k] + im[l] * cos[k];
        re[l] = re[j] - tre;
        im[l] = im[j] - tim;
        re[j] += tre;
        im[j] += tim;
      }
    }
  }
}

export interface FFTPlan {
  n: number;
  /** forward transform of (re, im) in place */
  forward(re: Float64Array, im: Float64Array): void;
  /** inverse transform (with 1/n) in place */
  inverse(re: Float64Array, im: Float64Array): void;
}

const cache = new Map<number, FFTPlan>();

export function fftPlan(n: number): FFTPlan {
  const hit = cache.get(n);
  if (hit) return hit;
  let plan: FFTPlan;
  if ((n & (n - 1)) === 0) {
    const p = makeRadix2(n);
    plan = {
      n,
      forward: (re, im) => radix2(p, re, im),
      inverse: (re, im) => {
        for (let i = 0; i < n; i++) im[i] = -im[i];
        radix2(p, re, im);
        for (let i = 0; i < n; i++) {
          re[i] /= n;
          im[i] = -im[i] / n;
        }
      },
    };
  } else {
    // Bluestein: X_k = conj(w_k) * sum_j (x_j conj(w_j)) w_{k-j}, w_j = e^{i pi j^2 / n}
    let m = 1;
    while (m < 2 * n - 1) m *= 2;
    const p = makeRadix2(m);
    const wr = new Float64Array(n);
    const wi = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      const a = (Math.PI * ((j * j) % (2 * n))) / n;
      wr[j] = Math.cos(a);
      wi[j] = Math.sin(a);
    }
    const br = new Float64Array(m);
    const bi = new Float64Array(m);
    br[0] = wr[0];
    bi[0] = wi[0];
    for (let j = 1; j < n; j++) {
      br[j] = br[m - j] = wr[j];
      bi[j] = bi[m - j] = wi[j];
    }
    radix2(p, br, bi);
    const ar = new Float64Array(m);
    const ai = new Float64Array(m);
    const fwd = (re: Float64Array, im: Float64Array) => {
      ar.fill(0);
      ai.fill(0);
      for (let j = 0; j < n; j++) {
        // x_j * conj(w_j)
        ar[j] = re[j] * wr[j] + im[j] * wi[j];
        ai[j] = im[j] * wr[j] - re[j] * wi[j];
      }
      radix2(p, ar, ai);
      for (let k = 0; k < m; k++) {
        const r = ar[k] * br[k] - ai[k] * bi[k];
        const i = ar[k] * bi[k] + ai[k] * br[k];
        // conj for inverse via forward trick
        ar[k] = r;
        ai[k] = -i;
      }
      radix2(p, ar, ai);
      for (let k = 0; k < n; k++) {
        const r = ar[k] / m;
        const i = -ai[k] / m;
        re[k] = r * wr[k] + i * wi[k];
        im[k] = i * wr[k] - r * wi[k];
      }
    };
    plan = {
      n,
      forward: fwd,
      inverse: (re, im) => {
        for (let i = 0; i < n; i++) im[i] = -im[i];
        fwd(re, im);
        for (let i = 0; i < n; i++) {
          re[i] /= n;
          im[i] = -im[i] / n;
        }
      },
    };
  }
  cache.set(n, plan);
  return plan;
}

/** 2D FFT of a rows×cols complex array (row-major), in place. */
export function fft2(
  re: Float64Array,
  im: Float64Array,
  rows: number,
  cols: number,
  inverse = false,
): void {
  const pr = fftPlan(cols);
  const pc = fftPlan(rows);
  const tr = new Float64Array(Math.max(rows, cols));
  const ti = new Float64Array(Math.max(rows, cols));
  const rr = tr.subarray(0, cols);
  const ri = ti.subarray(0, cols);
  for (let r = 0; r < rows; r++) {
    const o = r * cols;
    rr.set(re.subarray(o, o + cols));
    ri.set(im.subarray(o, o + cols));
    if (inverse) pr.inverse(rr, ri);
    else pr.forward(rr, ri);
    re.set(rr, o);
    im.set(ri, o);
  }
  const cr = tr.subarray(0, rows);
  const ci = ti.subarray(0, rows);
  for (let c = 0; c < cols; c++) {
    for (let r = 0; r < rows; r++) {
      cr[r] = re[r * cols + c];
      ci[r] = im[r * cols + c];
    }
    if (inverse) pc.inverse(cr, ci);
    else pc.forward(cr, ci);
    for (let r = 0; r < rows; r++) {
      re[r * cols + c] = cr[r];
      im[r * cols + c] = ci[r];
    }
  }
}
