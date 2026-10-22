/* PhyVision shared front-end helpers.
   Author: Swayam Shakya  |  License: MIT
   Works in two modes:
     live    - served by `python app.py`, talks to the /api endpoints
     offline - opened straight from disk; uses web/assets/static_data.js     */
(function () {
  "use strict";
  const scriptSrc = document.currentScript && document.currentScript.src;
  const webBase = scriptSrc ? new URL("../", scriptSrc).href : "web/";
  const S = window.PHY_STATIC || {};
  const CLASS_ORDER = ["athletic", "normal", "obese", "skinny"];
  const LABEL = { athletic: "Athletic", normal: "Normal", obese: "Obese", skinny: "Skinny" };

  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (x, d = 1) => (x == null ? "n/a" : (x * 100).toFixed(d) + "%");
  const $ = (sel, root = document) => root.querySelector(sel);

  let online = null;
  async function api(path, opts = {}) {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), opts.timeout || 60000);
    try {
      const r = await fetch(path, { ...opts, signal: ctrl.signal });
      const ct = r.headers.get("content-type") || "";
      const body = ct.includes("json") ? await r.json() : null;
      if (!r.ok) throw new Error((body && body.error) || `Request failed (${r.status})`);
      return body;
    } finally { clearTimeout(t); }
  }

  async function status() {
    if (location.protocol === "file:") { online = false; return null; }
    try { const s = await api("/api/status", { timeout: 8000 }); online = true; return s; }
    catch (e) { online = false; return null; }
  }

  async function guides() {
    if (online) { try { return await api("/api/guides"); } catch (e) { /* fall through */ } }
    return S.guides;
  }
  async function demos() {
    if (online) { try { return await api("/api/demos"); } catch (e) { /* fall through */ } }
    return S.demos || [];
  }
  async function metrics() {
    if (online) { try { return await api("/api/metrics"); } catch (e) { /* fall through */ } }
    return S.metrics || null;
  }
  async function files() {
    if (online) { try { return await api("/api/files"); } catch (e) { /* fall through */ } }
    return S.manifest || [];
  }

  /* ---- rendering helpers ---- */
  function bars(probs, host) {
    host.innerHTML = CLASS_ORDER.map((c) => `
      <div class="bar c-${c}" role="img" aria-label="${LABEL[c]} ${pct(probs[c])}">
        <span>${LABEL[c]}</span><div class="track"><div class="fill" data-w="${(probs[c] * 100).toFixed(2)}"></div></div>
        <span class="pct">${pct(probs[c])}</span></div>`).join("");
    requestAnimationFrame(() => host.querySelectorAll(".fill").forEach((f) => { f.style.width = f.dataset.w + "%"; }));
  }

  function guideHTML(g, key, opts = {}) {
    const list = (arr) => `<ul>${arr.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`;
    const week = g.training.weekly_plan.map((d) => `<div><b>${esc(d.day)}</b>${esc(d.session)}</div>`).join("");
    return `
    <article class="panel guide c-${key}" id="guide-${key}">
      <div class="row" style="justify-content:space-between">
        <h2 style="margin:0">${opts.heading ? esc(opts.heading) : esc(g.title) + " fitness guide"}</h2>
        <span class="chip">${esc(g.tagline)}</span>
      </div>
      <p class="muted" style="margin-top:12px">${esc(g.summary)}</p>
      <div class="guide-cols">
        <div><h3>Your goals</h3>${list(g.goals)}</div>
        <div><h3>Training focus</h3><p>${esc(g.training.focus)}</p><p class="small muted">${esc(g.training.progression)}</p></div>
      </div>
      <h3>A sample training week</h3>
      <div class="week">${week}</div>
      <h3>Key exercises</h3>
      <ul class="pills">${g.training.key_exercises.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>
      <div class="guide-cols">
        <div><h3>Nutrition</h3>${list(g.nutrition)}</div>
        <div><h3>A sample day of eating</h3>${list(g.sample_day)}</div>
      </div>
      <div class="guide-cols">
        <div><h3>Lifestyle and recovery</h3>${list(g.lifestyle)}</div>
        <div><h3>Stay safe</h3>${list(g.cautions)}</div>
      </div>
    </article>`;
  }

  function toast(msg) {
    const t = document.createElement("div");
    t.className = "toast"; t.setAttribute("role", "status"); t.textContent = msg;
    document.body.appendChild(t); setTimeout(() => t.remove(), 3200);
  }

  function initTheme() {
    const btn = $("[data-theme-toggle]");
    let saved = null;
    try { saved = localStorage.getItem("pv-theme"); } catch (e) { /* storage blocked */ }
    if (saved) document.documentElement.dataset.theme = saved;
    if (!btn) return;
    const label = () => {
      const dark = document.documentElement.dataset.theme === "dark" ||
        (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
      btn.textContent = dark ? "Light mode" : "Dark mode";
      return dark;
    };
    label();
    btn.addEventListener("click", () => {
      const next = label() ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem("pv-theme", next); } catch (e) { /* ignore */ }
      label();
    });
  }

  /* ---- photo-guide diagrams (inline SVG, theme aware) ---- */
  const FIG = `
    <circle cx="100" cy="40" r="15"/>
    <rect x="94" y="52" width="12" height="10" rx="3"/>
    <path d="M74 64 Q100 56 126 64 L123 100 Q119 122 122 150 L78 150 Q81 122 77 100 Z"/>
    <path d="M78 150 L122 150 L118 168 L82 168 Z"/>
    <g fill="none" stroke-linecap="round">
      <path d="M77 67 L66 108 L60 146" stroke-width="10"/>
      <path d="M123 67 L134 108 L140 146" stroke-width="10"/>
      <path d="M90 160 L88 218 L87 270" stroke-width="15"/>
      <path d="M110 160 L112 218 L113 270" stroke-width="15"/>
    </g>
    <ellipse cx="84" cy="276" rx="9" ry="5"/><ellipse cx="116" cy="276" rx="9" ry="5"/>`;

  function frameDiagram(opts = {}) {
    const labels = opts.labels !== false;
    return `<svg class="frame-diagram" viewBox="${labels ? "-6 -6 330 312" : "-4 -4 208 308"}" role="img"
      aria-label="Ideal photo: portrait 2:3 frame, whole body from head to feet filling about 85 percent of the height, small space above the head and below the feet, person centred, arms slightly away from the body">
      <rect x="0" y="0" width="200" height="300" rx="10" class="fd-frame"/>
      <line x1="100" y1="0" x2="100" y2="300" class="fd-guide"/>
      <g class="fd-figure">${FIG}</g>
      <line x1="0" y1="25" x2="200" y2="25" class="fd-mark"/>
      <line x1="0" y1="281" x2="200" y2="281" class="fd-mark"/>
      ${labels ? `
      <g class="fd-label">
        <path d="M206 1 L212 1 L212 24 L206 24" class="fd-brace"/><text x="216" y="16">5-10% space</text>
        <path d="M206 26 L212 26 L212 280 L206 280" class="fd-brace"/><text x="216" y="140">Body fills</text><text x="216" y="156">75-95% of</text><text x="216" y="172">the height</text>
        <path d="M206 282 L212 282 L212 299 L206 299" class="fd-brace"/><text x="216" y="295">feet visible</text>
        <text x="100" y="-0" text-anchor="middle" class="fd-small" dy="-1"></text>
      </g>` : ""}
    </svg>`;
  }

  function framingStrip(acc) {
    // cut = body y-coordinate (in the 0-300 figure space) where the photo ends
    const levels = [["full", "Head to feet", 300, "Best"], ["knees", "Head to knees", 222, "Good"],
      ["thigh", "Head to mid-thigh", 192, "OK"], ["waist", "Waist up", 162, "Least reliable"]];
    return levels.map(([k, name, cut, word]) => {
      const a = acc && acc[k] ? `: ${(acc[k].accuracy * 100).toFixed(1)}% on test photos` : "";
      // zoom so the visible part fills the frame, as in a real cropped photo
      const kz = k === "full" ? 1 : (300 - 14) / (cut - 25);
      const ty = k === "full" ? 0 : 14 - 25 * kz;
      return `<figure class="framing-card fr-${k}">
        <svg viewBox="0 0 200 300" aria-hidden="true">
          <defs><clipPath id="clip-${k}"><rect x="0" y="0" width="200" height="300" rx="10"/></clipPath></defs>
          <rect x="0" y="0" width="200" height="300" rx="10" class="fd-frame"/>
          <g clip-path="url(#clip-${k})"><g class="fd-figure" transform="translate(100 ${ty}) scale(${kz}) translate(-100 0)">${FIG}</g></g>
          <rect x="0" y="0" width="200" height="300" rx="10" class="fd-frame" style="fill:none"/>
        </svg>
        <figcaption><b>${name}</b><span>${word}${a}</span></figcaption>
      </figure>`;
    }).join("");
  }

  /* demo image URLs are stored relative to web/ */
  const webUrl = (rel) => new URL(rel, webBase).href;

  document.addEventListener("DOMContentLoaded", initTheme);
  window.PV = { api, status, guides, demos, metrics, files, bars, guideHTML, toast, esc, pct, webUrl, webBase, frameDiagram, framingStrip,
    CLASS_ORDER, LABEL, get online() { return online; }, static: S };
})();
