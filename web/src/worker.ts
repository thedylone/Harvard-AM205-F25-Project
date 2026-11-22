/// <reference lib="webworker" />
import { Compressor, logspace, type CurvePoint, type Method } from "./lib/compress.ts";
import { ssim } from "./lib/metrics.ts";
import { bands, toArray } from "./lib/dwt.ts";

export type ToWorker =
  | { type: "load"; id: number; rows: number; cols: number; pixels: Float64Array }
  | { type: "recon"; id: number; method: Method; ratio: number };

export type FromWorker =
  | { type: "maps"; id: number; dwt: MapData; dft: MapData }
  | { type: "pcaReady"; id: number; singular: Float64Array; maxK: number }
  | { type: "curve"; id: number; method: Method; points: CurvePoint[]; done: boolean }
  | {
      type: "recon";
      id: number;
      method: Method;
      ratio: number;
      gray: Uint8ClampedArray;
      cr: number;
      kept: number;
      total: number;
      psnr: number;
      ssim: number;
      relErr: number;
      mask: Uint8Array | null;
    }
  | { type: "error"; id: number; message: string };

export interface MapData {
  rows: number;
  cols: number;
  /** log-magnitude, normalised to 0–1; NaN where there is no coefficient (padding) */
  data: Float32Array;
}

const ctx = self as unknown as DedicatedWorkerGlobalScope;
const post = (msg: FromWorker, transfer: Transferable[] = []) => ctx.postMessage(msg, transfer);
const tick = () => new Promise<void>((r) => setTimeout(r, 0));

let current: {
  id: number;
  comp: Compressor;
  basics: boolean;
  pca: Promise<void>;
} | null = null;

function normalise(data: Float32Array, valid?: Uint8Array): Float32Array {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < data.length; i++) {
    if (valid && !valid[i]) continue;
    const v = data[i];
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const out = new Float32Array(data.length);
  const span = hi - lo || 1;
  for (let i = 0; i < data.length; i++) out[i] = valid && !valid[i] ? NaN : (data[i] - lo) / span;
  return out;
}

async function load(id: number, rows: number, cols: number, pixels: Float64Array) {
  const comp = new Compressor(pixels, rows, cols);
  let resolvePca!: () => void;
  const pca = new Promise<void>((r) => (resolvePca = r));
  current = { id, comp, basics: false, pca };

  comp.prepareDWT();
  comp.prepareDFT();
  current.basics = true;
  const dwtMap = comp.dwtMap();
  const dftMap = comp.dftMap();
  const dwt: MapData = { rows: dwtMap.rows, cols: dwtMap.cols, data: normalise(dwtMap.data, dwtMap.mask) };
  const dft: MapData = { rows: dftMap.rows, cols: dftMap.cols, data: normalise(dftMap.data) };
  post({ type: "maps", id, dwt, dft }, [dwt.data.buffer, dft.data.buffer]);

  const dense = logspace(0, 4, 100);
  post({ type: "curve", id, method: "DFT", points: comp.dftCurve(dense), done: true });
  await tick(); // let queued reconstruction requests through
  if (current?.id !== id) return;

  comp.preparePCA();
  resolvePca();
  const singular = comp.singularValues();
  post({ type: "pcaReady", id, singular, maxK: comp.pcaMaxK }, [singular.buffer]);
  post({ type: "curve", id, method: "PCA", points: comp.pcaCurve(dense), done: true });

  // DWT is biorthogonal, so its error needs real reconstructions.
  const ratios = logspace(0, 4, 41);
  const pts: CurvePoint[] = [];
  for (let i = 0; i < ratios.length; i++) {
    await tick();
    if (current?.id !== id) return;
    pts.push(comp.dwtPoint(ratios[i]));
    if (i % 8 === 7 || i === ratios.length - 1) {
      post({ type: "curve", id, method: "DWT", points: pts.slice(), done: i === ratios.length - 1 });
    }
  }
}

async function recon(id: number, method: Method, ratio: number) {
  if (!current || current.id !== id) return;
  if (method === "PCA") await current.pca;
  if (!current || current.id !== id) return;
  const { comp } = current;
  const rec = comp.reconstruct(method, ratio);
  const gray = new Uint8ClampedArray(rec.data.length);
  for (let i = 0; i < gray.length; i++) gray[i] = Math.round(rec.data[i]);
  let mask: Uint8Array | null = null;
  if (method === "DWT") {
    const t = comp.dwtThreshold(ratio);
    const bs = bands(comp.dec);
    const kept = toArray(comp.dec, (b, i) => (Math.abs(bs[b].data[i]) >= t ? 1 : 0)).data;
    mask = new Uint8Array(kept.length);
    for (let i = 0; i < mask.length; i++) mask[i] = kept[i];
  } else if (method === "DFT") {
    const t = comp.dftThreshold(ratio);
    const { rows, cols } = comp;
    mask = new Uint8Array(rows * cols);
    const hr = Math.floor(rows / 2);
    const hc = Math.floor(cols / 2);
    for (let r = 0; r < rows; r++) {
      const sr = (r + hr) % rows;
      for (let c = 0; c < cols; c++) {
        if (comp.dftMag[r * cols + c] >= t) mask[sr * cols + ((c + hc) % cols)] = 1;
      }
    }
  }
  const msg: FromWorker = {
    type: "recon",
    id,
    method,
    ratio,
    gray,
    cr: rec.cr,
    kept: rec.kept,
    total: rec.total,
    psnr: rec.psnr,
    ssim: ssim(comp.img, rec.data, comp.rows, comp.cols),
    relErr: rec.relErr,
    mask,
  };
  post(msg, mask ? [gray.buffer, mask.buffer] : [gray.buffer]);
}

ctx.onmessage = (ev: MessageEvent<ToWorker>) => {
  const m = ev.data;
  const run = m.type === "load" ? load(m.id, m.rows, m.cols, m.pixels) : recon(m.id, m.method, m.ratio);
  run.catch((err) => post({ type: "error", id: m.id, message: String(err?.message ?? err) }));
};
