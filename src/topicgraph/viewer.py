"""Self-contained HTML viewer. No CDN and no network calls: chat content stays on disk."""

from __future__ import annotations

import json
from typing import Any

_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Topic Graph</title>
<style>
:root {
  --bg: #fafaf9; --panel: #ffffff; --ink: #1c1917; --muted: #78716c; --line: #e7e5e4;
  --edge: #a8a29e; --accent: #2563eb; --hidden: #d97706;
  --chatgpt: #10a37f; --claude: #c2410c; --gemini: #4f46e5;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0c0a09; --panel: #1c1917; --ink: #f5f5f4; --muted: #a8a29e; --line: #292524;
    --edge: #57534e; --accent: #60a5fa; --hidden: #fbbf24;
    --chatgpt: #34d399; --claude: #fb923c; --gemini: #a5b4fc;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; }
header { display: flex; flex-wrap: wrap; gap: 12px 24px; align-items: baseline; padding: 12px 16px; border-bottom: 1px solid var(--line); }
header h1 { font-size: 16px; margin: 0; }
.stat { color: var(--muted); }
.stat b { color: var(--ink); font-variant-numeric: tabular-nums; }
.controls { display: flex; gap: 12px; align-items: center; padding: 8px 16px; flex-wrap: wrap; }
input[type=search] { padding: 6px 10px; border: 1px solid var(--line); border-radius: 6px; background: var(--panel); color: var(--ink); min-width: 220px; }
main { display: grid; grid-template-columns: 1fr 380px; height: calc(100vh - 100px); }
@media (max-width: 800px) { main { grid-template-columns: 1fr; height: auto; } #graph { height: 60vh; } }
#graph { width: 100%; height: 100%; display: block; cursor: grab; }
aside { border-left: 1px solid var(--line); background: var(--panel); overflow-y: auto; padding: 16px; }
.node circle { stroke: var(--panel); stroke-width: 1.5; cursor: pointer; }
.node.hiddenTopic circle { stroke: var(--hidden); stroke-width: 3; }
.node.sel circle { stroke: var(--accent); stroke-width: 3.5; }
.node.dim { opacity: .15; }
.node text { font-size: 11px; fill: var(--ink); pointer-events: none; paint-order: stroke; stroke: var(--bg); stroke-width: 3px; }
line { stroke: var(--edge); }
line.follows { stroke-dasharray: 3 3; }
.seg { border-top: 1px solid var(--line); padding: 10px 0; }
.meta { font-size: 12px; color: var(--muted); display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.badge { font-size: 11px; padding: 1px 6px; border-radius: 9px; color: #fff; }
.badge.chatgpt { background: var(--chatgpt); } .badge.claude { background: var(--claude); } .badge.gemini { background: var(--gemini); }
.tag { font-size: 11px; color: var(--hidden); border: 1px solid var(--hidden); border-radius: 9px; padding: 0 6px; }
details summary { cursor: pointer; color: var(--muted); font-size: 12px; }
pre { white-space: pre-wrap; font: 12px/1.4 ui-monospace, monospace; max-height: 240px; overflow: auto; }
.bars { display: flex; align-items: flex-end; gap: 2px; height: 36px; margin: 8px 0; }
.bars div { background: var(--accent); width: 8px; min-height: 2px; }
a { color: var(--accent); }
.legend { display: flex; gap: 12px; font-size: 12px; color: var(--muted); }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; vertical-align: -1px; }
</style>
</head>
<body>
<header>
  <h1>Topic Graph</h1>
  <span class="stat"><b id="sChats"></b> chats</span>
  <span class="stat"><b id="sSegs"></b> segments</span>
  <span class="stat"><b id="sTopics"></b> topics</span>
  <span class="stat"><b id="sHidden"></b> hidden topics</span>
</header>
<div class="controls">
  <input type="search" id="q" placeholder="Filter topics…">
  <label><input type="checkbox" id="eCo" checked> same chat</label>
  <label><input type="checkbox" id="eFo" checked> follows</label>
  <label>min size <input type="range" id="minSize" min="1" max="10" value="1"></label>
  <label>edges per topic <input type="range" id="topK" min="0" max="8" value="3"></label>
  <span class="legend">
    <span><span class="dot" style="background:var(--chatgpt)"></span> ChatGPT</span>
    <span><span class="dot" style="background:var(--claude)"></span> Claude</span>
    <span><span class="dot" style="background:var(--gemini)"></span> Gemini</span>
    <span><span class="dot" style="border:2px solid var(--hidden)"></span> hidden topic</span>
  </span>
</div>
<main>
  <svg id="graph"></svg>
  <aside id="panel"><p class="stat">Click a topic to list every segment about it, from every tool.</p></aside>
</main>
<script>
const DATA = __DATA__;
const $ = id => document.getElementById(id);
const S = DATA.summary;
$("sChats").textContent = S.chats; $("sSegs").textContent = S.segments; $("sTopics").textContent = S.topics;
$("sHidden").textContent = Math.round(S.hidden_topic_rate * 100) + "%";
const segById = Object.fromEntries(DATA.segments.map(s => [s.id, s]));
const svg = $("graph"), NS = "http://www.w3.org/2000/svg";
const dominant = n => Object.entries(n.sources).sort((a, b) => b[1] - a[1])[0][0];
let nodes = [], links = [], sel = null, view = {x: 0, y: 0, k: 1};

function build() {
  const minSize = +$("minSize").value;
  const keep = new Set(DATA.nodes.filter(n => n.segment_ids.length >= minSize).map(n => n.id));
  const old = Object.fromEntries(nodes.map(n => [n.id, n]));
  const W = svg.clientWidth || 800, H = svg.clientHeight || 600;
  nodes = DATA.nodes.filter(n => keep.has(n.id)).map(n => old[n.id] || Object.assign({}, n, {
    x: W / 2 + (Math.random() - .5) * W * .6, y: H / 2 + (Math.random() - .5) * H * .6, vx: 0, vy: 0,
    r: Math.min(22, 4 + 2 * Math.sqrt(n.segment_ids.length))}));
  const idx = Object.fromEntries(nodes.map(n => [n.id, n]));
  const kinds = new Set([...($("eCo").checked ? ["cooccur"] : []), ...($("eFo").checked ? ["follows"] : [])]);
  // Keep each node's strongest edges so the layout shows structure, not a hairball.
  const perNode = +$("topK").value, count = {};
  links = [];
  for (const e of DATA.edges) {  // edges arrive sorted by weight, heaviest first
    if (!kinds.has(e.kind) || !idx[e.source] || !idx[e.target]) continue;
    if ((count[e.source] || 0) >= perNode && (count[e.target] || 0) >= perNode) continue;
    count[e.source] = (count[e.source] || 0) + 1; count[e.target] = (count[e.target] || 0) + 1;
    links.push({s: idx[e.source], t: idx[e.target], kind: e.kind, w: e.weight});
  }
  const maxW = Math.max(1, ...links.map(l => l.w));
  for (const l of links) l.k = l.w / maxW;
  simulate(400);
  fit();
  draw();
}

function simulate(steps) {
  for (let it = 0; it < steps; it++) {
    const alpha = 1 - it / steps;
    for (let i = 0; i < nodes.length; i++) for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i], b = nodes[j];
      const dx = b.x - a.x, dy = b.y - a.y, d = Math.sqrt(dx * dx + dy * dy) || .1;
      let f = 9000 * alpha / (d * d);
      const overlap = a.r + b.r + 14 - d;
      if (overlap > 0) f += overlap * .5;  // collision
      a.vx -= f * dx / d; a.vy -= f * dy / d; b.vx += f * dx / d; b.vy += f * dy / d;
    }
    for (const l of links) {
      const dx = l.t.x - l.s.x, dy = l.t.y - l.s.y, d = Math.sqrt(dx * dx + dy * dy) || 1;
      const f = (d - 120) * .03 * l.k * alpha;
      l.s.vx += f * dx / d; l.s.vy += f * dy / d; l.t.vx -= f * dx / d; l.t.vy -= f * dy / d;
    }
    let cx = 0, cy = 0;
    for (const n of nodes) { cx += n.x; cy += n.y; }
    cx /= nodes.length || 1; cy /= nodes.length || 1;
    for (const n of nodes) {
      n.vx += (cx - n.x) * .002; n.vy += (cy - n.y) * .002;
      n.x += n.vx *= .55; n.y += n.vy *= .55;
    }
  }
}

function fit() {
  if (!nodes.length) return;
  const W = svg.clientWidth || 800, H = svg.clientHeight || 600, pad = 40;
  const xs = nodes.map(n => n.x), ys = nodes.map(n => n.y);
  const x0 = Math.min(...xs) - pad, x1 = Math.max(...xs) + pad * 3, y0 = Math.min(...ys) - pad, y1 = Math.max(...ys) + pad;
  const k = Math.min(1, W / (x1 - x0), H / (y1 - y0));
  view = {k, x: (W - (x1 - x0) * k) / 2 - x0 * k, y: (H - (y1 - y0) * k) / 2 - y0 * k};
}

function draw() {
  svg.innerHTML = "";
  const g = document.createElementNS(NS, "g");
  g.setAttribute("transform", `translate(${view.x},${view.y}) scale(${view.k})`);
  svg.appendChild(g);
  const q = $("q").value.trim().toLowerCase();
  for (const l of links) {
    const e = document.createElementNS(NS, "line");
    e.setAttribute("x1", l.s.x); e.setAttribute("y1", l.s.y); e.setAttribute("x2", l.t.x); e.setAttribute("y2", l.t.y);
    e.setAttribute("stroke-width", Math.min(1 + Math.log2(l.w), 5)); e.setAttribute("class", l.kind);
    g.appendChild(e);
  }
  for (const n of nodes) {
    const ng = document.createElementNS(NS, "g");
    const cls = ["node"];
    if (n.hidden_share === 1) cls.push("hiddenTopic");
    if (sel === n.id) cls.push("sel");
    if (q && !n.label.toLowerCase().includes(q)) cls.push("dim");
    ng.setAttribute("class", cls.join(" "));
    const c = document.createElementNS(NS, "circle");
    c.setAttribute("cx", n.x); c.setAttribute("cy", n.y); c.setAttribute("r", n.r);
    c.setAttribute("fill", `var(--${dominant(n)})`);
    const t = document.createElementNS(NS, "text");
    t.setAttribute("x", n.x + n.r + 3); t.setAttribute("y", n.y + 4); t.textContent = n.label;
    const title = document.createElementNS(NS, "title");
    title.textContent = `${n.label} — ${n.segment_ids.length} segments`;
    ng.append(c, t, title);
    ng.addEventListener("click", ev => { ev.stopPropagation(); select(n); });
    g.appendChild(ng);
  }
}

function esc(s) { return String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c])); }

function select(n) {
  sel = n.id; draw();
  const segs = n.segment_ids.map(id => segById[id]).sort((a, b) => (b.timestamp || "").localeCompare(a.timestamp || ""));
  const months = Object.entries(n.by_month), max = Math.max(1, ...months.map(m => m[1]));
  const hiddenCount = segs.filter(s => s.hidden).length;
  $("panel").innerHTML = `
    <h2 style="margin:0 0 4px;font-size:18px">${esc(n.label)}</h2>
    <div class="meta">${segs.length} segments · ${Object.entries(n.sources).map(([k, v]) => `${esc(k)} ${v}`).join(" · ")}
      · ${hiddenCount} hidden under other titles</div>
    <div class="bars" title="segments per month">${months.map(([m, c]) => `<div title="${esc(m)}: ${c}" style="height:${100 * c / max}%"></div>`).join("")}</div>
    ${segs.map(s => `<div class="seg">
      <div class="meta"><span class="badge ${esc(s.source)}">${esc(s.source)}</span>
        <span>${esc((s.timestamp || "").slice(0, 10))}</span>
        ${s.hidden ? '<span class="tag">hidden</span>' : ""}</div>
      <div><b>${esc(s.label)}</b></div>
      <div class="meta">in “${esc(s.chat_title)}”, ${s.end - s.start > 1 ? `turns ${s.start + 1}–${s.end}` : `turn ${s.end}`}
        ${s.url ? `· <a href="${esc(s.url)}" target="_blank" rel="noopener">open chat</a>` : ""}</div>
      <details><summary>show text</summary><pre>${esc(s.text)}</pre></details>
    </div>`).join("")}`;
}

let drag = null;
svg.addEventListener("mousedown", e => { drag = {x: e.clientX - view.x, y: e.clientY - view.y}; });
window.addEventListener("mouseup", () => drag = null);
window.addEventListener("mousemove", e => { if (drag) { view.x = e.clientX - drag.x; view.y = e.clientY - drag.y; draw(); } });
svg.addEventListener("wheel", e => {
  e.preventDefault();
  const k = Math.min(4, Math.max(.2, view.k * (e.deltaY < 0 ? 1.1 : 0.9)));
  const r = svg.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
  view.x = mx - (mx - view.x) * k / view.k; view.y = my - (my - view.y) * k / view.k; view.k = k; draw();
}, {passive: false});
$("q").addEventListener("input", draw);
for (const id of ["eCo", "eFo", "minSize", "topK"]) $(id).addEventListener("input", build);
build();
</script>
</body>
</html>
"""


def render_html(graph: dict[str, Any]) -> str:
    # Escape "</" so chat text cannot close the script tag.
    payload = json.dumps(graph, ensure_ascii=False).replace("</", "<\\/")
    return _TEMPLATE.replace("__DATA__", payload)
