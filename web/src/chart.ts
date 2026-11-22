/** Small dependency-free SVG line chart with a log-scaled x axis. */

export interface Series {
  key: string;
  label: string;
  color: string;
  points: { x: number; y: number; lo?: number; hi?: number }[];
  pending?: boolean;
}

export interface ChartOptions {
  xLabel: string;
  yLabel: string;
  xDomain: [number, number];
  yDomain?: [number, number];
  yFormat: (v: number) => string;
  xFormat?: (v: number) => string;
  /** highlighted point (current slider setting) */
  marker?: { key: string; x: number; y: number } | null;
  ariaLabel: string;
}

const SVGNS = "http://www.w3.org/2000/svg";
const el = <K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number> = {}) => {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  return e;
};

const sup = (n: number) =>
  String(n)
    .split("")
    .map((c) => "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"["0123456789-".indexOf(c)])
    .join("");

export const fmtPow10 = (v: number) => {
  const e = Math.round(Math.log10(v));
  return e === 0 ? "1" : e === 1 ? "10" : `10${sup(e)}`;
};

function niceTicks(lo: number, hi: number, count: number): number[] {
  const span = hi - lo;
  const step0 = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const err = step0 / mag;
  const step = (err >= 5 ? 10 : err >= 2 ? 5 : err >= 1.5 ? 2 : 1) * mag;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

/** Linear interpolation in log-x of a series at x. */
function interp(points: Series["points"], x: number): number | null {
  if (!points.length) return null;
  const sorted = points;
  if (x < sorted[0].x || x > sorted[sorted.length - 1].x) return null;
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i].x >= x) {
      const a = sorted[i - 1];
      const b = sorted[i];
      const t = b.x === a.x ? 0 : (Math.log(x) - Math.log(a.x)) / (Math.log(b.x) - Math.log(a.x));
      return a.y + t * (b.y - a.y);
    }
  }
  return sorted[sorted.length - 1].y;
}

export class LineChart {
  private root: HTMLElement;
  private svg: SVGSVGElement;
  private tip: HTMLDivElement;
  private series: Series[] = [];
  private opts!: ChartOptions;
  private ro: ResizeObserver;

  constructor(root: HTMLElement) {
    this.root = root;
    root.classList.add("chart");
    this.svg = el("svg", { role: "img" });
    this.tip = document.createElement("div");
    this.tip.className = "chart-tip";
    this.tip.hidden = true;
    root.append(this.svg, this.tip);
    this.ro = new ResizeObserver(() => this.render());
    this.ro.observe(root);
  }

  update(series: Series[], opts: ChartOptions) {
    this.series = series.map((s) => ({ ...s, points: [...s.points].sort((a, b) => a.x - b.x) }));
    this.opts = opts;
    this.render();
  }

  private render() {
    if (!this.opts) return;
    const { opts } = this;
    const W = Math.max(280, this.root.clientWidth);
    const H = Math.round(Math.min(360, Math.max(240, W * 0.58)));
    const narrow = W < 480;
    const m = { t: 14, r: narrow ? 14 : 58, b: 44, l: 50 };
    const iw = W - m.l - m.r;
    const ih = H - m.t - m.b;
    const svg = this.svg;
    svg.replaceChildren();
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("width", String(W));
    svg.setAttribute("height", String(H));
    svg.setAttribute("aria-label", opts.ariaLabel);

    const [x0, x1] = opts.xDomain;
    const lx0 = Math.log10(x0);
    const lx1 = Math.log10(x1);
    const X = (x: number) => m.l + ((Math.log10(x) - lx0) / (lx1 - lx0)) * iw;
    let [y0, y1] = opts.yDomain ?? [Infinity, -Infinity];
    if (!opts.yDomain) {
      for (const s of this.series)
        for (const p of s.points) {
          if (p.x < x0 || p.x > x1 || !Number.isFinite(p.y)) continue;
          y0 = Math.min(y0, p.lo ?? p.y);
          y1 = Math.max(y1, p.hi ?? p.y);
        }
      if (!Number.isFinite(y0)) [y0, y1] = [0, 1];
      const pad = (y1 - y0) * 0.06 || 1;
      y0 -= pad;
      y1 += pad;
    }
    const yt = niceTicks(y0, y1, narrow ? 4 : 5);
    y0 = Math.min(y0, yt[0]);
    y1 = Math.max(y1, yt[yt.length - 1]);
    const Y = (y: number) => m.t + ih - ((y - y0) / (y1 - y0)) * ih;

    // grid + axes
    const g = el("g", { class: "axis" });
    for (const t of yt) {
      g.append(el("line", { x1: m.l, x2: m.l + iw, y1: Y(t), y2: Y(t), class: "grid" }));
      const tx = el("text", { x: m.l - 8, y: Y(t) + 4, "text-anchor": "end" });
      tx.textContent = opts.yFormat(t);
      g.append(tx);
    }
    for (let e = Math.ceil(lx0); e <= Math.floor(lx1); e++) {
      const v = Math.pow(10, e);
      g.append(el("line", { x1: X(v), x2: X(v), y1: m.t, y2: m.t + ih, class: "grid grid-v" }));
      const tx = el("text", { x: X(v), y: m.t + ih + 18, "text-anchor": "middle" });
      tx.textContent = (opts.xFormat ?? fmtPow10)(v);
      g.append(tx);
    }
    g.append(el("line", { x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih, class: "baseline" }));
    const xl = el("text", { x: m.l + iw / 2, y: H - 6, "text-anchor": "middle", class: "axis-label" });
    xl.textContent = opts.xLabel;
    const yl = el("text", {
      x: -(m.t + ih / 2),
      y: 13,
      transform: "rotate(-90)",
      "text-anchor": "middle",
      class: "axis-label",
    });
    yl.textContent = opts.yLabel;
    g.append(xl, yl);
    svg.append(g);

    const clipId = `clip-${Math.random().toString(36).slice(2)}`;
    const clip = el("clipPath", { id: clipId });
    clip.append(el("rect", { x: m.l, y: m.t - 4, width: iw, height: ih + 8 }));
    svg.append(clip);
    const plot = el("g", { "clip-path": `url(#${clipId})` });
    svg.append(plot);

    const visible = (p: { x: number; y: number }) => p.x >= x0 * 0.999 && p.x <= x1 * 1.001 && Number.isFinite(p.y);
    const labels: { x: number; y: number; text: string; color: string }[] = [];
    for (const s of this.series) {
      const pts = s.points.filter(visible);
      if (!pts.length) continue;
      if (pts.some((p) => p.lo !== undefined)) {
        const top = pts.map((p) => `${X(p.x)},${Y(p.hi ?? p.y)}`);
        const bot = pts.map((p) => `${X(p.x)},${Y(p.lo ?? p.y)}`).reverse();
        plot.append(el("polygon", { points: [...top, ...bot].join(" "), fill: s.color, "fill-opacity": 0.14 }));
      }
      const d = pts.map((p, i) => `${i ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join("");
      plot.append(
        el("path", {
          d,
          fill: "none",
          stroke: s.color,
          "stroke-width": 2,
          "stroke-linejoin": "round",
          "stroke-linecap": "round",
          "stroke-dasharray": s.pending ? "3 4" : "none",
        }),
      );
      const last = pts[pts.length - 1];
      labels.push({ x: X(last.x), y: Math.min(m.t + ih, Math.max(m.t, Y(last.y))), text: s.label, color: s.color });
    }

    // direct labels at line ends (wide layouts), nudged apart
    if (!narrow) {
      // lines reaching the right edge are labelled in the margin; others just past their last point
      const edge = labels.filter((l) => l.x > m.l + iw - 4).sort((a, b) => a.y - b.y);
      for (let i = 1; i < edge.length; i++) edge[i].y = Math.max(edge[i].y, edge[i - 1].y + 14);
      for (const l of labels) {
        const atEdge = edge.includes(l);
        const t = el("text", {
          x: atEdge ? m.l + iw + 8 : l.x + 7,
          y: atEdge ? l.y + 4 : l.y + 4,
          class: "direct-label",
        });
        t.textContent = l.text;
        svg.append(t);
      }
    }

    if (opts.marker && opts.marker.x >= x0 && opts.marker.x <= x1 && Number.isFinite(opts.marker.y)) {
      const s = this.series.find((q) => q.key === opts.marker!.key);
      const mx = X(opts.marker.x);
      const my = Math.min(m.t + ih, Math.max(m.t, Y(opts.marker.y)));
      svg.append(el("line", { x1: mx, x2: mx, y1: m.t, y2: m.t + ih, class: "marker-rule" }));
      svg.append(
        el("circle", { cx: mx, cy: my, r: 5.5, fill: s?.color ?? "#000", stroke: "#f9f9f6", "stroke-width": 2 }),
      );
    }

    // hover layer
    const cross = el("line", { y1: m.t, y2: m.t + ih, class: "crosshair", visibility: "hidden" });
    const dots = this.series.map((s) =>
      el("circle", { r: 4, fill: s.color, stroke: "#f9f9f6", "stroke-width": 2, visibility: "hidden" }),
    );
    svg.append(cross, ...dots);
    const hit = el("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent", class: "hit" });
    svg.append(hit);
    const move = (clientX: number) => {
      const box = svg.getBoundingClientRect();
      const px = ((clientX - box.left) / box.width) * W;
      const x = Math.pow(10, lx0 + ((px - m.l) / iw) * (lx1 - lx0));
      if (x < x0 || x > x1) return hide();
      cross.setAttribute("x1", String(px));
      cross.setAttribute("x2", String(px));
      cross.setAttribute("visibility", "visible");
      const rows: string[] = [];
      this.series.forEach((s, i) => {
        const y = interp(s.points.filter(visible), x);
        if (y === null) {
          dots[i].setAttribute("visibility", "hidden");
          return;
        }
        dots[i].setAttribute("cx", String(px));
        dots[i].setAttribute("cy", String(Y(y)));
        dots[i].setAttribute("visibility", "visible");
        rows.push(
          `<div class="tip-row"><span class="swatch" style="background:${s.color}"></span><span>${s.label}</span><b>${opts.yFormat(y)}</b></div>`,
        );
      });
      this.tip.innerHTML = `<div class="tip-head">CR ${x < 10 ? x.toFixed(1) : Math.round(x).toLocaleString()}</div>${rows.join("") || '<div class="tip-row">no data</div>'}`;
      this.tip.hidden = false;
      const tw = this.tip.offsetWidth;
      const left = (px / W) * box.width;
      this.tip.style.left = `${Math.min(box.width - tw - 4, Math.max(4, left + 12 + tw > box.width ? left - tw - 12 : left + 12))}px`;
      this.tip.style.top = `${(m.t / H) * box.height + 4}px`;
    };
    const hide = () => {
      cross.setAttribute("visibility", "hidden");
      dots.forEach((d) => d.setAttribute("visibility", "hidden"));
      this.tip.hidden = true;
    };
    hit.addEventListener("pointermove", (e) => move(e.clientX));
    hit.addEventListener("pointerdown", (e) => move(e.clientX));
    hit.addEventListener("pointerleave", hide);
  }
}
