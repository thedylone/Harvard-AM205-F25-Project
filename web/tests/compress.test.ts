import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { Compressor, METHODS, logspace } from "../src/lib/compress.ts";
import { wavedec2, waverec2, coeffArrayShape } from "../src/lib/dwt.ts";
import { fft2 } from "../src/lib/fft.ts";
import { ssim } from "../src/lib/metrics.ts";

const ref = JSON.parse(readFileSync(new URL("./fixtures/reference.json", import.meta.url), "utf8"));

const close = (a: number, b: number, tol: number, msg: string) =>
  assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), `${msg}: ${a} vs ${b}`);

for (const [name, c] of Object.entries<any>(ref)) {
  const img = Float64Array.from(c.img as number[]);
  const { rows, cols } = c;

  test(`${name}: wavedec2 matches pywt`, () => {
    const dec = wavedec2(img, rows, cols);
    assert.deepEqual(coeffArrayShape(dec), c.dwtShape);
    c.dwtApprox.forEach((v: number, i: number) => close(dec.approx.data[i], v, 1e-9, `approx ${i}`));
    c.dwtDetail0.forEach((v: number, i: number) => close(dec.levels[0].h.data[i], v, 1e-9, `detail ${i}`));
    const rec = waverec2(dec);
    c.recon.forEach((v: number, i: number) => close(rec[i], v, 1e-9, `recon ${i}`));
    for (let i = 0; i < img.length; i++) close(rec[i], img[i], 1e-8, `perfect recon ${i}`);
  });

  test(`${name}: compression matches code/compression.py`, () => {
    const comp = new Compressor(img, rows, cols);
    comp.prepare();
    for (const m of METHODS) {
      const exp = c[m];
      const ratios: number[] = c.ratios;
      // PCA skips duplicate ranks, so match by CR
      const got = ratios.map((r) => comp.reconstruct(m, r));
      exp.cr.forEach((cr: number, i: number) => {
        const g = got.find((x) => Math.abs(x.cr - cr) < 1e-9 * cr);
        assert.ok(g, `${m} cr ${cr} present in ${got.map((x) => x.cr)}`);
        // NumPy's FFT rounding splits some conjugate pairs at the threshold; we keep
        // pairs together, so DFT agrees to within a fraction of a percent.
        const tol = m === "DFT" ? 5e-3 : 1e-6;
        if (Number.isFinite(exp.psnr[i]) && exp.psnr[i] < 200) close(g!.psnr, exp.psnr[i], tol, `${m} psnr @${cr}`);
        close(ssim(img, g!.data, rows, cols), exp.ssim[i], m === "DFT" ? 1e-2 : 1e-6, `${m} ssim @${cr}`);
      });
    }
  });

  test(`${name}: analytic curves agree with reconstructions`, () => {
    const comp = new Compressor(img, rows, cols);
    comp.prepare();
    const rs = logspace(0, 3, 7);
    comp.dftCurve(rs).forEach((p, i) => close(p.relErr, comp.reconstructDFT(rs[i]).relErr, 1e-6, "dft curve"));
    for (const p of comp.pcaCurve(rs)) {
      const r = rs.find((x) => comp.pcaK(x) === p.kept)!;
      close(p.relErr, comp.reconstructPCA(r).relErr, 1e-6, "pca curve");
    }
  });

  test(`${name}: ssim matches skimage`, () => {
    close(ssim(img, Float64Array.from(c.noisy), rows, cols), c.ssimNoisy, 1e-9, "ssim");
  });
}

test("fft2 round-trips odd sizes", () => {
  for (const [r, cc] of [[5, 7], [12, 9], [16, 8]]) {
    const re = Float64Array.from({ length: r * cc }, (_, i) => Math.sin(i * 1.3) + i % 3);
    const orig = Float64Array.from(re);
    const im = new Float64Array(r * cc);
    fft2(re, im, r, cc);
    // DC term = sum
    close(re[0], orig.reduce((a, b) => a + b, 0), 1e-9, "dc");
    fft2(re, im, r, cc, true);
    for (let i = 0; i < re.length; i++) close(re[i], orig[i], 1e-9, "roundtrip");
  }
});
