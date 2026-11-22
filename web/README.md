# Project page

Source for <https://projects.thedylone.com/image-compression/>: an in-browser
compression demo (DWT / DFT / PCA) plus a short writeup of the report. Vite + TypeScript,
no framework. Styled with the shared project-page kit: `public/project-kit.css` and
`public/project-kit.js` are verbatim copies of v1.2 from the `projects-gallery` branch of
thedylone.github.io (don't edit them here); page-only styles are in `src/style.css`. The report itself (`main.tex`, `code/`, `images/`, `results/`) is untouched.

```sh
cd web
npm install
npm run dev      # http://localhost:5173/image-compression/
npm test         # checks the TS port against the repo's Python (tests/fixtures)
npm run build    # → web/dist
```

Deployed by `.github/workflows/pages.yml` on pushes to `main` that touch `web/`
(Settings → Pages → Source must be **GitHub Actions**).

## How the demo works

- `src/lib/` ports `code/compression.py`: `dwt.ts` reproduces `pywt.wavedec2`/`waverec2`
  with `bior4.4` and symmetric padding; `fft.ts` is a radix-2 + Bluestein FFT; PCA is the
  column-centred rank-k approximation that `sklearn.decomposition.PCA` gives, via a
  symmetric eigensolver (`eig.ts`). Coefficient counts and compression ratios follow the
  repo exactly (`kf = int(N / ratio)`, `CR = mn / kf`, `CR_PCA = mn / (k(m+n))`).
- Images are converted to grayscale with the repo's `0.299 R + 0.587 G + 0.114 B`,
  downscaled to at most 512 px and cropped to even dimensions, then processed in a Web Worker.
- The error curves use real reconstructions for the DWT (biorthogonal), Parseval for the
  DFT and the tail of the eigenvalues for PCA.
- The only intentional difference: conjugate DFT pairs are always kept together (NumPy's
  rounding sometimes splits one pair at the threshold), so DFT numbers can differ from
  NumPy's by a few hundredths of a dB.

## Regenerating assets

`python3 scripts/export_assets.py` writes `public/samples/*.png`, `public/hero-strip.webp`
and `src/data/kodak-results.json` (from `results/med_res_*.pkl`).
`python3 scripts/make_fixtures.py` regenerates the test fixtures by calling
`code/compression.py`. Both need numpy, scipy, pywavelets, scikit-learn, scikit-image,
matplotlib, tqdm and Pillow. `public/gallery-cover.webp` is a 960×600 screenshot of the
top of the page, for the projects gallery.
