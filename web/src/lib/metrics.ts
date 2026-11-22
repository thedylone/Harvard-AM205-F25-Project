/** Image quality metrics, mirroring skimage.metrics with data_range=255. */

export function sse(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

export function energy(a: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  return s;
}

/** PSNR in dB for a given mean squared error (data_range = 255). */
export const psnrFromMse = (mse: number) => (mse <= 0 ? Infinity : 10 * Math.log10((255 * 255) / mse));

/**
 * SSIM as computed by skimage.metrics.structural_similarity with its defaults:
 * 7×7 uniform window, K1 = 0.01, K2 = 0.03, sample covariance, mean over the
 * image with a 3-pixel border cropped.
 */
export function ssim(x: ArrayLike<number>, y: ArrayLike<number>, rows: number, cols: number): number {
  const W = 7;
  const pad = 3;
  if (rows < W || cols < W) return NaN;
  const NP = W * W;
  const covNorm = NP / (NP - 1);
  const C1 = (0.01 * 255) ** 2;
  const C2 = (0.03 * 255) ** 2;
  const c1 = cols + 1;
  const Sx = new Float64Array((rows + 1) * c1);
  const Sy = new Float64Array((rows + 1) * c1);
  const Sxx = new Float64Array((rows + 1) * c1);
  const Syy = new Float64Array((rows + 1) * c1);
  const Sxy = new Float64Array((rows + 1) * c1);
  for (let r = 0; r < rows; r++) {
    let ax = 0, ay = 0, axx = 0, ayy = 0, axy = 0;
    for (let c = 0; c < cols; c++) {
      const xv = x[r * cols + c];
      const yv = y[r * cols + c];
      ax += xv;
      ay += yv;
      axx += xv * xv;
      ayy += yv * yv;
      axy += xv * yv;
      const p = (r + 1) * c1 + c + 1;
      const q = r * c1 + c + 1;
      Sx[p] = Sx[q] + ax;
      Sy[p] = Sy[q] + ay;
      Sxx[p] = Sxx[q] + axx;
      Syy[p] = Syy[q] + ayy;
      Sxy[p] = Sxy[q] + axy;
    }
  }
  const box = (S: Float64Array, r0: number, c0: number) =>
    S[(r0 + W) * c1 + c0 + W] - S[r0 * c1 + c0 + W] - S[(r0 + W) * c1 + c0] + S[r0 * c1 + c0];
  let total = 0;
  let count = 0;
  for (let r = pad; r < rows - pad; r++) {
    for (let c = pad; c < cols - pad; c++) {
      const r0 = r - pad;
      const c0 = c - pad;
      const ux = box(Sx, r0, c0) / NP;
      const uy = box(Sy, r0, c0) / NP;
      const vx = covNorm * (box(Sxx, r0, c0) / NP - ux * ux);
      const vy = covNorm * (box(Syy, r0, c0) / NP - uy * uy);
      const vxy = covNorm * (box(Sxy, r0, c0) / NP - ux * uy);
      const s =
        ((2 * ux * uy + C1) * (2 * vxy + C2)) / ((ux * ux + uy * uy + C1) * (vx + vy + C2));
      total += s;
      count++;
    }
  }
  return total / count;
}
