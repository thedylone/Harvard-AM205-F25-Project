import "katex/dist/katex.min.css";
import "./style.css";
import katex from "katex";
import { LineChart, type Series } from "./chart.ts";
import type { CurvePoint, Method } from "./lib/compress.ts";
import type { FromWorker, MapData, ToWorker } from "./worker.ts";
import kodak from "./data/kodak-results.json";

// ---------------------------------------------------------------- maths
for (const node of document.querySelectorAll<HTMLElement>("[data-tex]")) {
  katex.render(node.dataset.tex!, node, { throwOnError: false });
}
for (const node of document.querySelectorAll<HTMLElement>("[data-tex-display]")) {
  katex.render(node.dataset.texDisplay!, node, { throwOnError: false, displayMode: true });
}

// ---------------------------------------------------------------- constants
const MAX_SIDE = 512;
const COLORS: Record<Method, string> = { DWT: "#3a5a9a", DFT: "#b8860b", PCA: "#9a3d63" };
const NAMES: Record<Method, string> = { DWT: "DWT", DFT: "DFT", PCA: "PCA" };
const BASE = import.meta.env.BASE_URL;

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const ui = {
  samples: $("samples"),
  upload: $<HTMLInputElement>("upload"),
  methods: $("methods"),
  keep: $<HTMLInputElement>("keep"),
  keepOut: $<HTMLOutputElement>("keep-out"),
  original: $<HTMLCanvasElement>("original"),
  recon: $<HTMLCanvasElement>("recon"),
  figures: $("figures"),
  dims: $("dims"),
  busy: $("busy"),
  reconLabel: $("recon-label"),
  reconMeta: $("recon-meta"),
  mCr: $("m-cr"),
  mKept: $("m-kept"),
  mKeptLabel: $("m-kept-label"),
  mPsnr: $("m-psnr"),
  mSsim: $("m-ssim"),
  mRel: $("m-rel"),
  pcaNote: $("pca-note"),
  metricToggle: $("metric-toggle"),
  map: $<HTMLCanvasElement>("map"),
  mapTitle: $("map-title"),
  mapNote: $("map-note"),
  scree: $("scree"),
  kodakToggle: $("kodak-toggle"),
};

// ---------------------------------------------------------------- state
type Metric = "psnr" | "relErr";
const state = {
  id: 0,
  rows: 0,
  cols: 0,
  method: "DWT" as Method,
  metric: "psnr" as Metric,
  curves: { DWT: [], DFT: [], PCA: [] } as Record<Method, CurvePoint[]>,
  curveDone: { DWT: false, DFT: false, PCA: false } as Record<Method, boolean>,
  maps: null as { dwt: MapData; dft: MapData } | null,
  singular: null as Float64Array | null,
  last: null as Extract<FromWorker, { type: "recon" }> | null,
  inflight: false,
  queued: false,
};

const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
const send = (msg: ToWorker, transfer: Transferable[] = []) => worker.postMessage(msg, transfer);

// ---------------------------------------------------------------- slider mapping
const sliderToFraction = (v: number) => Math.pow(10, -4 + (4 * v) / 1000);
const fractionToSlider = (f: number) => Math.round(((Math.log10(f) + 4) / 4) * 1000);
const currentRatio = () => 1 / sliderToFraction(+ui.keep.value);

function fmtPct(f: number) {
  const p = f * 100;
  if (p >= 10) return `${p.toFixed(0)}%`;
  if (p >= 1) return `${p.toFixed(1)}%`;
  if (p >= 0.1) return `${p.toFixed(2)}%`;
  return `${p.toFixed(3)}%`;
}

const fmtInt = (n: number) => Math.round(n).toLocaleString("en-US");
const fmtCr = (cr: number) => (cr < 10 ? cr.toFixed(2) : cr < 100 ? cr.toFixed(1) : fmtInt(cr));

function updateSliderLabel() {
  const f = sliderToFraction(+ui.keep.value);
  const ratio = 1 / f;
  ui.keepOut.innerHTML = `${fmtPct(f)}<small>1 in ${ratio < 10 ? ratio.toFixed(1) : fmtInt(ratio)}</small>`;
  ui.keep.style.setProperty("--fill", `${(+ui.keep.value / 1000) * 100}%`);
  ui.keep.setAttribute("aria-valuetext", `${fmtPct(f)} of coefficients kept, target compression ratio ${fmtCr(ratio)}`);
}

// ---------------------------------------------------------------- requests
function requestRecon() {
  if (!state.rows) return;
  if (state.inflight) {
    state.queued = true;
    return;
  }
  state.inflight = true;
  state.queued = false;
  ui.busy.hidden = false;
  send({ type: "recon", id: state.id, method: state.method, ratio: currentRatio() });
}

function loadPixels(pixels: Float64Array, rows: number, cols: number, label: string) {
  state.id += 1;
  state.rows = rows;
  state.cols = cols;
  state.curves = { DWT: [], DFT: [], PCA: [] };
  state.curveDone = { DWT: false, DFT: false, PCA: false };
  state.maps = null;
  state.singular = null;
  state.last = null;
  state.inflight = false;
  state.queued = false;

  ui.figures.style.setProperty("--aspect", `${cols} / ${rows}`);
  drawGray(ui.original, Uint8ClampedArray.from(pixels), rows, cols);
  const blank = ui.recon.getContext("2d")!;
  ui.recon.width = cols;
  ui.recon.height = rows;
  blank.clearRect(0, 0, cols, rows);
  ui.dims.textContent = `${label} · ${cols}×${rows}`;
  renderCurves();
  send({ type: "load", id: state.id, rows, cols, pixels }, [pixels.buffer]);
  requestRecon();
}

worker.onmessage = (ev: MessageEvent<FromWorker>) => {
  const m = ev.data;
  if (m.id !== state.id) {
    if (m.type === "recon") {
      state.inflight = false;
      requestRecon();
    }
    return;
  }
  switch (m.type) {
    case "maps":
      state.maps = { dwt: m.dwt, dft: m.dft };
      renderMap();
      break;
    case "pcaReady":
      state.singular = m.singular;
      renderMap();
      break;
    case "curve":
      state.curves[m.method] = m.points;
      state.curveDone[m.method] = m.done;
      renderCurves();
      break;
    case "recon":
      state.inflight = false;
      state.last = m;
      showRecon(m);
      if (state.queued || m.method !== state.method || Math.abs(m.ratio - currentRatio()) > 1e-9) {
        requestRecon();
      } else {
        ui.busy.hidden = true;
      }
      break;
    case "error":
      state.inflight = false;
      ui.busy.hidden = true;
      console.error(m.message);
      ui.pcaNote.hidden = false;
      ui.pcaNote.textContent = `Something went wrong: ${m.message}`;
      break;
  }
};

// ---------------------------------------------------------------- drawing
function drawGray(canvas: HTMLCanvasElement, gray: Uint8ClampedArray, rows: number, cols: number) {
  canvas.width = cols;
  canvas.height = rows;
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(cols, rows);
  for (let i = 0; i < gray.length; i++) {
    const v = gray[i];
    img.data[4 * i] = v;
    img.data[4 * i + 1] = v;
    img.data[4 * i + 2] = v;
    img.data[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function showRecon(m: Extract<FromWorker, { type: "recon" }>) {
  drawGray(ui.recon, m.gray, state.rows, state.cols);
  ui.reconLabel.textContent = `${NAMES[m.method]} reconstruction`;
  ui.reconMeta.textContent = `CR ${fmtCr(m.cr)}`;
  ui.mCr.textContent = fmtCr(m.cr);
  if (m.method === "PCA") {
    ui.mKeptLabel.textContent = "Components k";
    ui.mKept.innerHTML = `${fmtInt(m.kept)}<small>of ${fmtInt(m.total)}</small>`;
  } else {
    ui.mKeptLabel.textContent = "Nonzero coefficients";
    ui.mKept.innerHTML = `${fmtInt(m.kept)}<small>of ${fmtInt(m.total)}</small>`;
  }
  ui.mPsnr.innerHTML = Number.isFinite(m.psnr) && m.psnr < 150 ? `${m.psnr.toFixed(1)}<small>dB</small>` : "∞<small>lossless</small>";
  ui.mSsim.textContent = m.ssim.toFixed(3);
  ui.mRel.textContent = m.relErr < 1e-6 ? "≈ 0" : `${(m.relErr * 100).toFixed(m.relErr < 0.01 ? 2 : 1)}%`;

  const notes: string[] = [];
  const maxCr = (state.rows * state.cols) / (state.rows + state.cols);
  if (m.method === "PCA" && m.ratio > maxCr) {
    notes.push(
      `PCA can't compress this image beyond CR = mn/(m+n) ≈ ${fmtCr(maxCr)}: even k = 1 must store a left and right singular vector. The other two methods keep going.`,
    );
  }
  if (m.method === "DWT" && m.cr < 1) {
    notes.push(
      `With symmetric boundary padding the bior4.4 transform produces ${fmtInt(m.total)} coefficients for ${fmtInt(state.rows * state.cols)} pixels, so keeping them all gives a CR just under 1 (CR is counted against the pixels, as in the repo).`,
    );
  }
  ui.pcaNote.hidden = !notes.length;
  ui.pcaNote.textContent = notes.join(" ");
  renderCurves();
  renderMap();
}

// ---------------------------------------------------------------- coefficient map
function renderMap() {
  const m = state.method;
  const last = state.last && state.last.method === m ? state.last : null;
  if (m === "PCA") {
    ui.mapTitle.textContent = "Singular values kept";
    ui.map.hidden = true;
    ui.scree.hidden = false;
    renderScree(last ? last.kept : null);
    ui.mapNote.textContent =
      "Singular values σᵢ of the column-centred image, on a log scale. Rank-k PCA keeps the first k; each one costs m + n stored numbers.";
    return;
  }
  ui.map.hidden = false;
  ui.scree.hidden = true;
  ui.mapTitle.textContent = m === "DWT" ? "Wavelet coefficients kept" : "Fourier coefficients kept";
  ui.mapNote.textContent =
    m === "DWT"
      ? "log |coefficient| in the pywt coeffs_to_array layout, coarsest approximation top-left. Gold coefficients are above the threshold and kept."
      : "log |coefficient| of the centred (fftshift) spectrum, low frequencies in the middle. Gold coefficients are above the threshold and kept.";
  if (!state.maps) return;
  const map = m === "DWT" ? state.maps.dwt : state.maps.dft;
  const mask = last?.mask ?? null;
  const ctx = ui.map.getContext("2d")!;
  ui.map.width = map.cols;
  ui.map.height = map.rows;
  const img = ctx.createImageData(map.cols, map.rows);
  const d = img.data;
  // dropped: navy-strong → slate; kept: gold → warm white
  for (let i = 0; i < map.data.length; i++) {
    const v = map.data[i];
    let r: number, g: number, b: number;
    if (Number.isNaN(v)) {
      r = 0x0c; g = 0x15; b = 0x26;
    } else if (mask && mask[i]) {
      const t = Math.min(1, Math.max(0, (v - 0.35) / 0.65));
      r = 184 + (255 - 184) * t;
      g = 134 + (244 - 134) * t;
      b = 11 + (214 - 11) * t;
    } else {
      const t = Math.pow(Math.max(0, v), 1.6) * 0.85;
      r = 0x14 + (0x8d - 0x14) * t;
      g = 0x21 + (0x9a - 0x21) * t;
      b = 0x3a + (0xb3 - 0x3a) * t;
    }
    d[4 * i] = r;
    d[4 * i + 1] = g;
    d[4 * i + 2] = b;
    d[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function renderScree(k: number | null) {
  const sv = state.singular;
  const host = ui.scree;
  if (!sv) {
    host.innerHTML = `<p style="padding:1rem;font-family:var(--font-mono);font-size:.75rem;color:var(--color-text-faint)">Computing the SVD…</p>`;
    return;
  }
  const W = 400;
  const H = 300;
  const m = { l: 42, r: 10, t: 14, b: 30 };
  const n = sv.length;
  const vals = Array.from(sv).filter((v) => v > 0);
  const hi = Math.log10(Math.max(...vals));
  const lo = Math.max(hi - 5, Math.log10(Math.min(...vals)));
  const X = (i: number) => m.l + (i / (n - 1 || 1)) * (W - m.l - m.r);
  const Y = (v: number) => m.t + (1 - (Math.log10(Math.max(v, 10 ** lo)) - lo) / (hi - lo)) * (H - m.t - m.b);
  const base = H - m.b;
  const pts = Array.from(sv, (v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`);
  const kk = k ?? 0;
  const keptPts = pts.slice(0, Math.max(1, kk));
  const area = (p: string[], i0: number, i1: number) => `M${X(i0)},${base} L${p.join(" L")} L${X(i1)},${base} Z`;
  let ticks = "";
  for (let e = Math.ceil(lo); e <= Math.floor(hi); e++) {
    ticks += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(10 ** e)}" y2="${Y(10 ** e)}" stroke="rgba(15,23,42,.1)"/><text x="${m.l - 6}" y="${Y(10 ** e) + 3.5}" text-anchor="end">${e === 0 ? "1" : e === 1 ? "10" : `10<tspan dy="-4" font-size="8">${e}</tspan>`}</text>`;
  }
  host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Singular values of the image; ${kk} of ${n} kept">
    <style>text{font-family:var(--font-mono);font-size:10px;fill:rgba(15,23,42,.5)}</style>
    ${ticks}
    <path d="${area(pts, 0, n - 1)}" fill="rgba(15,23,42,.07)"/>
    <polyline points="${pts.join(" ")}" fill="none" stroke="rgba(15,23,42,.35)" stroke-width="1.5"/>
    ${kk ? `<path d="${area(keptPts, 0, Math.max(0, kk - 1))}" fill="${COLORS.PCA}" fill-opacity=".22"/><polyline points="${keptPts.join(" ")}" fill="none" stroke="${COLORS.PCA}" stroke-width="2"/><line x1="${X(kk - 1)}" x2="${X(kk - 1)}" y1="${m.t}" y2="${base}" stroke="${COLORS.PCA}" stroke-dasharray="3 3"/><text x="${Math.min(X(kk - 1) + 6, W - 90)}" y="${m.t + 12}" style="fill:${COLORS.PCA}">k = ${kk}</text>` : ""}
    <line x1="${m.l}" x2="${W - m.r}" y1="${base}" y2="${base}" stroke="rgba(15,23,42,.2)"/>
    <text x="${m.l}" y="${H - 10}">1</text><text x="${W - m.r}" y="${H - 10}" text-anchor="end">${n}</text>
    <text x="${(m.l + W - m.r) / 2}" y="${H - 10}" text-anchor="middle" style="text-transform:uppercase;letter-spacing:.06em">component index</text>
  </svg>`;
}

// ---------------------------------------------------------------- curves
const curveChart = new LineChart($("curve"));

function renderCurves() {
  const metric = state.metric;
  const methods: Method[] = ["DWT", "DFT", "PCA"];
  const series: Series[] = methods.map((m) => ({
    key: m,
    label: m,
    color: COLORS[m],
    points: state.curves[m].map((p) => ({ x: p.cr, y: metric === "psnr" ? p.psnr : p.relErr * 100 })),
    pending: !state.curveDone[m],
  }));
  let yDomain: [number, number] | undefined;
  const ys = series.flatMap((s) => s.points.filter((p) => p.x >= 1 && Number.isFinite(p.y)).map((p) => p.y));
  if (ys.length) {
    const lo = Math.min(...ys);
    const hi = Math.max(...ys);
    yDomain = metric === "psnr" ? [Math.max(0, lo - 1.5), Math.min(hi, 60) + 1.5] : [0, hi * 1.06];
  }
  const last = state.last;
  curveChart.update(series, {
    xLabel: "Compression ratio (log)",
    yLabel: metric === "psnr" ? "PSNR (dB)" : "Relative error (%)",
    xDomain: [1, 1e4],
    yDomain,
    yFormat: metric === "psnr" ? (v) => `${v.toFixed(0)}` : (v) => `${v.toFixed(v < 10 ? 1 : 0)}`,
    marker: last
      ? { key: last.method, x: last.cr, y: metric === "psnr" ? last.psnr : last.relErr * 100 }
      : null,
    ariaLabel: `${metric === "psnr" ? "PSNR" : "Relative error"} against compression ratio for DWT, DFT and PCA on the current image.`,
  });
  // tooltip formatting uses the same formatter; units for clarity
}

// ---------------------------------------------------------------- Kodak chart
type KodakPoint = { cr: number; mean: number; lo: number; hi: number };
const kodakData = kodak as unknown as { methods: Record<Method, Record<"psnr" | "ssim", KodakPoint[]>> };
const kodakChart = new LineChart($("kodak"));
let kodakMetric: "psnr" | "ssim" = "psnr";
function renderKodak() {
  const series: Series[] = (["DWT", "DFT", "PCA"] as Method[]).map((m) => ({
    key: m,
    label: m,
    color: COLORS[m],
    points: kodakData.methods[m][kodakMetric].map((p) => ({ x: p.cr, y: p.mean, lo: p.lo, hi: p.hi })),
  }));
  kodakChart.update(series, {
    xLabel: "Compression ratio (log)",
    yLabel: kodakMetric === "psnr" ? "PSNR (dB)" : "SSIM",
    xDomain: [1, 1e4],
    yFormat: kodakMetric === "psnr" ? (v) => v.toFixed(0) : (v) => v.toFixed(2),
    ariaLabel: `Mean ${kodakMetric.toUpperCase()} against compression ratio on the Kodak dataset for DWT, DFT and PCA.`,
  });
}
renderKodak();

// ---------------------------------------------------------------- radio groups
function radioGroup(root: HTMLElement, attr: string, onPick: (v: string) => void) {
  const buttons = [...root.querySelectorAll<HTMLButtonElement>(`button[${attr}]`)];
  const pick = (b: HTMLButtonElement, focus = false) => {
    buttons.forEach((x) => {
      x.setAttribute("aria-checked", String(x === b));
      x.tabIndex = x === b ? 0 : -1;
    });
    if (focus) b.focus();
    onPick(b.getAttribute(attr)!);
  };
  buttons.forEach((b, i) => {
    b.tabIndex = b.getAttribute("aria-checked") === "true" ? 0 : -1;
    b.addEventListener("click", () => pick(b));
    b.addEventListener("keydown", (e) => {
      const d = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0;
      if (!d) return;
      e.preventDefault();
      pick(buttons[(i + d + buttons.length) % buttons.length], true);
    });
  });
  return (value: string | null) =>
    buttons.forEach((x) => {
      const on = x.getAttribute(attr) === value;
      x.setAttribute("aria-checked", String(on));
      x.tabIndex = on ? 0 : -1;
    });
}

radioGroup(ui.methods, "data-method", (v) => {
  state.method = v as Method;
  renderMap();
  requestRecon();
});
radioGroup(ui.metricToggle, "data-metric", (v) => {
  state.metric = v as Metric;
  renderCurves();
});
radioGroup(ui.kodakToggle, "data-metric", (v) => {
  kodakMetric = v as "psnr" | "ssim";
  renderKodak();
});
const setSampleChecked = radioGroup(ui.samples, "data-sample", (v) => loadSample(v));

ui.keep.addEventListener("input", () => {
  updateSliderLabel();
  requestRecon();
});
for (const b of document.querySelectorAll<HTMLButtonElement>("button[data-cr]")) {
  b.addEventListener("click", () => {
    ui.keep.value = String(fractionToSlider(1 / +b.dataset.cr!));
    updateSliderLabel();
    requestRecon();
  });
}

// ---------------------------------------------------------------- image loading
function mulberry32(a: number) {
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const SAMPLE_LABEL: Record<string, string> = {
  raccoon: "Raccoon",
  cameraman: "Cameraman",
  astronaut: "Astronaut",
  noise: "Uniform noise",
};

async function loadSample(name: string) {
  if (name === "noise") {
    // like code/generate_random_images.py: i.i.d. uniform pixels
    const n = 512;
    const rnd = mulberry32(205);
    const px = new Float64Array(n * n);
    for (let i = 0; i < px.length; i++) px[i] = Math.floor(rnd() * 256);
    loadPixels(px, n, n, SAMPLE_LABEL[name]);
    return;
  }
  const img = new Image();
  img.decoding = "async";
  img.src = `${BASE}samples/${name}.png`;
  await img.decode();
  const { data, rows, cols } = rasterize(img, img.naturalWidth, img.naturalHeight);
  const px = new Float64Array(rows * cols);
  for (let i = 0; i < px.length; i++) px[i] = data[4 * i]; // already grayscale
  loadPixels(px, rows, cols, SAMPLE_LABEL[name]);
}

/** Draw a source into a canvas no larger than MAX_SIDE, cropped to even dimensions. */
function rasterize(src: CanvasImageSource, w: number, h: number) {
  let scale = Math.min(1, MAX_SIDE / Math.max(w, h));
  let cur: CanvasImageSource = src;
  let cw = w;
  let ch = h;
  // halve repeatedly for a cleaner downscale
  while (scale < 0.5) {
    const c = document.createElement("canvas");
    c.width = Math.max(1, Math.round(cw / 2));
    c.height = Math.max(1, Math.round(ch / 2));
    const x = c.getContext("2d")!;
    x.imageSmoothingQuality = "high";
    x.drawImage(cur, 0, 0, c.width, c.height);
    cur = c;
    cw = c.width;
    ch = c.height;
    scale *= 2;
  }
  let cols = Math.max(2, Math.round(cw * scale));
  let rows = Math.max(2, Math.round(ch * scale));
  const fullC = cols;
  const fullR = rows;
  cols -= cols % 2; // crop_to_even in code/compression.py
  rows -= rows % 2;
  const c = document.createElement("canvas");
  c.width = cols;
  c.height = rows;
  const x = c.getContext("2d", { willReadFrequently: true })!;
  x.fillStyle = "#fff";
  x.fillRect(0, 0, cols, rows);
  x.imageSmoothingQuality = "high";
  x.drawImage(cur, 0, 0, fullC, fullR);
  return { data: x.getImageData(0, 0, cols, rows).data, rows, cols };
}

async function loadFile(file: File) {
  if (!file.type.startsWith("image/")) return;
  try {
    const bmp = await createImageBitmap(file);
    const { data, rows, cols } = rasterize(bmp, bmp.width, bmp.height);
    bmp.close();
    if (rows < 16 || cols < 16) throw new Error("Image is too small");
    const px = new Float64Array(rows * cols);
    // rgb_to_grayscale from code/main.py
    for (let i = 0; i < px.length; i++) {
      px[i] = Math.floor(0.299 * data[4 * i] + 0.587 * data[4 * i + 1] + 0.114 * data[4 * i + 2]);
    }
    setSampleChecked(null);
    const label = file.name.length > 24 ? `${file.name.slice(0, 21)}…` : file.name;
    loadPixels(px, rows, cols, label);
  } catch (err) {
    ui.pcaNote.hidden = false;
    ui.pcaNote.textContent = `Couldn't read that image (${(err as Error).message}).`;
  }
}

ui.upload.addEventListener("change", () => {
  const f = ui.upload.files?.[0];
  if (f) loadFile(f);
  ui.upload.value = "";
});
ui.figures.addEventListener("dragover", (e) => {
  e.preventDefault();
  ui.figures.classList.add("drag");
});
ui.figures.addEventListener("dragleave", () => ui.figures.classList.remove("drag"));
ui.figures.addEventListener("drop", (e) => {
  e.preventDefault();
  ui.figures.classList.remove("drag");
  const f = e.dataTransfer?.files?.[0];
  if (f) loadFile(f);
});

// ---------------------------------------------------------------- go
updateSliderLabel();
renderMap();
loadSample("raccoon");
