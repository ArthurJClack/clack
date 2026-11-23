// shellm live viewer: the model's actual parts in 3d, updated for every token.
//
//   x axis  = token position (the token being processed sits at the right end of the wall)
//   y axis  = depth: token row at the bottom, then one row per transformer layer
//   wall    = the residual stream: one cube per (token, layer), brighter = larger norm
//   arcs    = attention: where the current token reads from, one colour per head
//   panels  = that layer's mlp neurons firing for the current token (orange +, blue -)
//   tiles   = the layer's weight matrices (q k v o | gate up down); pink = edited
//
// everything is driven by GET /api/events, so api calls from anywhere show up here.
import * as THREE from 'three';
import { OrbitControls } from '/static/vendor/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const DX = 0.6, DY = 2.05, Y0 = 1.9, VIS = 28;
const PANEL_X = 1.4, PANEL_W = 3.8, PANEL_H = 1.3, TILE = 0.84, TILE_GAP = 0.12;
const TILES_X = PANEL_X + PANEL_W + 0.7;
const MATS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'];
const MAT_LABEL = ['q', 'k', 'v', 'o', 'gate', 'up', 'down'];
const TOPK = 4, SEG = 16, THR = 0.05;

const S = {
  info: null, L: 0, H: 0, F: 0, tokens: [], nPrompt: 0, steps: [], shown: -1, follow: true,
  rate: 6, lastAdvance: 0, resid: [], headOn: [], stepTime: 0, steer: new Map(), edited: new Set(),
  genAbort: null, source: '', spans: [], hoverAbs: -1, cur: -1,
};
const yOf = (l) => Y0 + l * DY;
const xOfK = (k) => -(VIS - 1 - k) * DX;
const show = (s) => (s === '\n' ? '⏎' : s.replace(/\n/g, '⏎'));

// ------------------------------------------------------------------ three setup
const canvas = $('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0c11);
scene.fog = new THREE.Fog(0x0a0c11, 60, 140);
const camera = new THREE.PerspectiveCamera(36, 1, 0.1, 400);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.addEventListener('start', () => (S.userMoved = true));
canvas.addEventListener('dblclick', () => { S.userMoved = false; fitCamera(); });
scene.add(new THREE.AmbientLight(0xffffff, 1.5));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(8, 18, 14);
scene.add(sun);

// the panels cover the left, right and bottom edges; everything is framed inside what is left
function freeArea() {
  const w = window.innerWidth, h = window.innerHeight, wide = w > 980;
  const padL = wide ? 324 : 284, padR = wide ? 274 : 12, padB = 174, padT = 12;
  return { w, h, padL, padR, padB, padT, fw: Math.max(200, w - padL - padR), fh: Math.max(200, h - padB - padT) };
}
function resize() {
  const a = freeArea();
  renderer.setSize(a.w, a.h, false);
  camera.aspect = a.w / a.h;
  camera.setViewOffset(a.w, a.h, -(a.padL - a.padR) / 2, (a.padB - a.padT) / 2, a.w, a.h);
  camera.updateProjectionMatrix();
}
// pull the camera back until the whole model fits the free area, at a slight angle so depth reads
function fitCamera() {
  if (!S.bounds) return;
  const a = freeArea(), b = S.bounds, t = Math.tan((camera.fov * Math.PI) / 360);
  const dW = ((b.x1 - b.x0) / (2 * t)) * (a.h / a.fw), dH = ((b.y1 - b.y0) / (2 * t)) * (a.h / a.fh);
  const d = Math.max(dW, dH) * 1.1, yaw = 0.2, pitch = 0.2;
  controls.target.set((b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2, 0);
  camera.position.set(controls.target.x + d * Math.sin(yaw) * Math.cos(pitch), controls.target.y + d * Math.sin(pitch), d * Math.cos(yaw) * Math.cos(pitch));
  controls.update();
}
window.addEventListener('resize', () => { resize(); if (!S.userMoved) fitCamera(); });

// ------------------------------------------------------------------ html labels pinned to 3d points
const labels = [];
const _v = new THREE.Vector3();
function mkLabel(cls, align = 'c') {
  const el = document.createElement('div');
  el.className = 'lab ' + cls;
  el.style.display = 'none';
  $('labels').appendChild(el);
  const o = { el, pos: new THREE.Vector3(), on: false, align };
  labels.push(o);
  return o;
}
function placeLabels() {
  const w = window.innerWidth, h = window.innerHeight;
  for (const o of labels) {
    if (!o.on) { if (o.el.style.display !== 'none') o.el.style.display = 'none'; continue; }
    _v.copy(o.pos).project(camera);
    if (_v.z > 1 || _v.x < -1.2 || _v.x > 1.2 || _v.y < -1.2 || _v.y > 1.2) { o.el.style.display = 'none'; continue; }
    const x = (_v.x * 0.5 + 0.5) * w, y = (-_v.y * 0.5 + 0.5) * h;
    const shift = o.align === 'l' ? 'translate(0,-50%)' : o.align === 'r' ? 'translate(-100%,-50%)' : 'translate(-50%,-50%)';
    o.el.style.display = '';
    o.el.style.transform = `translate(${x.toFixed(1)}px,${y.toFixed(1)}px) ${shift}`;
  }
}

// ------------------------------------------------------------------ scene
const G = {};  // meshes for the current model
const headColor = (h) => new THREE.Color().setHSL(h / S.H, 0.78, 0.6);
const C_COLD = new THREE.Color(0x151c29), C_HOT = new THREE.Color(0x7fd4ff), C_WHITE = new THREE.Color(0xffffff);
const C_TILE = new THREE.Color(0x1b2331), C_READ = new THREE.Color(0xffd27a);
const _c = new THREE.Color();

function buildScene() {
  if (G.root) {
    scene.remove(G.root);
    G.root.traverse((o) => { o.geometry?.dispose(); o.material?.map?.dispose(); o.material?.dispose(); });
  }
  for (const o of labels.splice(0)) o.el.remove();
  const { L, H, F } = S;
  const root = (G.root = new THREE.Group());
  scene.add(root);
  const m4 = new THREE.Matrix4();
  const yTop = yOf(L);

  // token row
  G.tiles = new THREE.InstancedMesh(new THREE.BoxGeometry(DX * 0.84, 0.2, 0.9), new THREE.MeshLambertMaterial(), VIS);
  for (let k = 0; k < VIS; k++) { G.tiles.setMatrixAt(k, m4.makeTranslation(xOfK(k), 0, 0)); G.tiles.setColorAt(k, C_TILE); }
  root.add(G.tiles);

  // residual wall
  G.wall = new THREE.InstancedMesh(new THREE.BoxGeometry(DX * 0.78, 0.62, 0.55), new THREE.MeshLambertMaterial(), L * VIS);
  for (let l = 0; l < L; l++) for (let k = 0; k < VIS; k++) {
    G.wall.setMatrixAt(l * VIS + k, m4.makeTranslation(xOfK(k), yOf(l), 0));
    G.wall.setColorAt(l * VIS + k, C_COLD);
  }
  root.add(G.wall);

  // the column being computed: a faint spine and a pulse that climbs it
  G.spine = new THREE.Mesh(new THREE.BoxGeometry(DX * 0.96, yTop + 0.3, 0.72),
    new THREE.MeshBasicMaterial({ color: 0x7fd4ff, transparent: true, opacity: 0.07, depthWrite: false }));
  G.spine.position.set(0, (yTop + 0.3) / 2 - 0.15, 0);
  root.add(G.spine);
  G.pulse = new THREE.Mesh(new THREE.SphereGeometry(0.2, 20, 14), new THREE.MeshBasicMaterial({ color: 0xffffff }));
  G.pulse.position.set(0, 0, 0.6);
  G.pulse.visible = false;
  root.add(G.pulse);
  G.hover = new THREE.Mesh(new THREE.BoxGeometry(DX * 0.96, yTop + 0.3, 0.72),
    new THREE.MeshBasicMaterial({ color: 0xffd27a, transparent: true, opacity: 0.12, depthWrite: false }));
  G.hover.visible = false;
  root.add(G.hover);

  // attention ribbons, one dynamic mesh per layer
  G.arcs = [];
  const maxVerts = H * TOPK * SEG * 6;
  for (let l = 0; l < L; l++) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(maxVerts * 3), 3).setUsage(THREE.DynamicDrawUsage));
    geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(maxVerts * 3), 3).setUsage(THREE.DynamicDrawUsage));
    geo.setDrawRange(0, 0);
    const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
      vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }));
    mesh.frustumCulled = false;
    root.add(mesh);
    G.arcs.push(mesh);
  }

  // mlp panels
  S.gridC = Math.ceil(Math.sqrt(F * PANEL_W / PANEL_H));
  S.gridR = Math.ceil(F / S.gridC);
  G.panels = [];
  for (let l = 0; l < L; l++) {
    const tex = new THREE.DataTexture(new Uint8Array(S.gridC * S.gridR * 4), S.gridC, S.gridR, THREE.RGBAFormat);
    tex.magFilter = THREE.NearestFilter; tex.minFilter = THREE.NearestFilter; tex.colorSpace = THREE.SRGBColorSpace;
    const back = new THREE.Mesh(new THREE.PlaneGeometry(PANEL_W + 0.12, PANEL_H + 0.12), new THREE.MeshBasicMaterial({ color: 0x10151f }));
    back.position.set(PANEL_X + PANEL_W / 2, yOf(l), -0.02);
    root.add(back);
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(PANEL_W, PANEL_H), new THREE.MeshBasicMaterial({ map: tex, transparent: true }));
    mesh.position.set(PANEL_X + PANEL_W / 2, yOf(l), 0);
    root.add(mesh);
    G.panels.push(mesh);
  }

  // weight tiles
  G.wt = {};
  for (let l = 0; l < L; l++) MATS.forEach((mname, i) => {
    const name = S.info.layers[l].tensors[mname];
    const tex = new THREE.DataTexture(new Uint8Array(4), 1, 1, THREE.RGBAFormat);
    tex.magFilter = THREE.NearestFilter; tex.colorSpace = THREE.SRGBColorSpace; tex.needsUpdate = true;
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(TILE, TILE), new THREE.MeshBasicMaterial({ map: tex }));
    const x = TILES_X + i * (TILE + TILE_GAP) + (i >= 4 ? 0.3 : 0) + TILE / 2;
    mesh.position.set(x, yOf(l), 0);
    root.add(mesh);
    const frame = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(TILE + 0.1, TILE + 0.1)),
      new THREE.LineBasicMaterial({ color: 0xff5fb8 }));
    frame.position.copy(mesh.position); frame.position.z = 0.01; frame.visible = false;
    root.add(frame);
    G.wt[name] = { mesh, frame, attn: i < 4 };
    loadTile(name);
  });

  // steering markers
  G.steerBars = [];
  for (let l = 0; l < L; l++) {
    const bar = new THREE.Mesh(new THREE.BoxGeometry(VIS * DX, 0.07, 0.6), new THREE.MeshBasicMaterial({ color: 0xff5fb8 }));
    bar.position.set(xOfK(0) + (VIS - 1) * DX / 2, yOf(l) + 0.42, 0);
    bar.visible = false;
    root.add(bar);
    const lab = mkLabel('mod', 'r');
    lab.pos.set(xOfK(0) - 0.6, yOf(l) + 0.8, 0);
    G.steerBars.push({ bar, lab });
  }

  const grid = new THREE.GridHelper(120, 120, 0x1a2130, 0x121722);
  grid.position.set(-8, -0.4, 0);
  root.add(grid);

  // pinned labels
  G.lens = [];
  const xLens = TILES_X + 7 * (TILE + TILE_GAP) + 0.7;
  for (let l = 0; l < L; l++) {
    const tag = mkLabel('', 'r');
    tag.pos.set(xOfK(0) - 0.6, yOf(l), 0);
    tag.el.textContent = `layer ${l}`; tag.on = true;
    const lens = mkLabel('lens', 'l');
    lens.pos.set(xLens, yOf(l), 0);
    G.lens.push(lens);
  }
  const head = (text, x, y = yTop - 0.75) => { const o = mkLabel('head'); o.pos.set(x, y, 0); o.el.textContent = text; o.on = true; };
  head('residual stream · one cube per token per layer', xOfK(0) + 7 * DX, yTop - 0.75);
  head('mlp neurons', PANEL_X + PANEL_W / 2);
  head('attention weights q k v o', TILES_X + 2 * (TILE + TILE_GAP), yTop - 0.2);
  head('mlp weights', TILES_X + 5.5 * (TILE + TILE_GAP) + 0.3);
  head('guess so far', xLens + 1.3, yTop - 0.2);
  const tk = mkLabel('', 'r'); tk.pos.set(xOfK(0) - 0.6, 0, 0); tk.el.textContent = 'tokens'; tk.on = true;
  G.next = mkLabel('next');
  G.next.pos.set(0, yTop - 0.3, 0.4);
  G.tokLabs = [];
  for (let i = 0; i < TOPK + 2; i++) G.tokLabs.push(mkLabel('tok'));
  G.hint = mkLabel('head');
  G.hint.pos.set(-4, yTop + 0.15, 0);
  G.hint.el.textContent = 'press generate, or call the api: it shows up here';
  G.hint.on = true;

  S.bounds = { x0: xOfK(0) - 3.4, x1: xLens + 3.6, y0: -2.9, y1: yTop + 0.5 };
  S.userMoved = false;
  fitCamera();
  refreshMods();
}

async function loadTile(name) {
  const t = G.wt[name];
  if (!t) return;
  const edited = S.edited.has(name);
  const r = await fetch(`/api/weights?name=${encodeURIComponent(name)}&size=28${edited ? '&diff=1' : ''}`).then((x) => x.json());
  const raw = Uint8Array.from(atob(r.data), (c) => c.charCodeAt(0));
  const px = new Uint8Array(r.rows * r.cols * 4);
  const tint = edited ? [255, 95, 184] : t.attn ? [120, 205, 215] : [235, 185, 115];
  for (let i = 0; i < raw.length; i++) {
    const v = Math.pow(raw[i] / 255, edited ? 0.45 : 1.6);
    px[i * 4] = 14 + tint[0] * v * 0.92; px[i * 4 + 1] = 18 + tint[1] * v * 0.92; px[i * 4 + 2] = 26 + tint[2] * v * 0.9; px[i * 4 + 3] = 255;
  }
  const tex = new THREE.DataTexture(px, r.cols, r.rows, THREE.RGBAFormat);
  tex.magFilter = THREE.NearestFilter; tex.colorSpace = THREE.SRGBColorSpace; tex.needsUpdate = true;
  t.mesh.material.map?.dispose();
  t.mesh.material.map = tex;
  t.mesh.material.needsUpdate = true;
  t.frame.visible = edited;
}

function refreshMods() {
  if (!G.steerBars) return;
  G.steerBars.forEach(({ bar, lab }, l) => {
    const a = S.steer.get(l);
    bar.visible = lab.on = a !== undefined;
    if (a !== undefined) lab.el.textContent = `steered ×${a.toFixed(1)} ▸`;
  });
}

// ------------------------------------------------------------------ one step -> visuals
function addArc(pos, col, n, ax, ay, bx, by, w, color, hOff) {
  const dist = Math.abs(ax - bx);
  const cx = (ax + bx) / 2, cy = (ay + by) / 2 + 0.1, cz = 0.75 + 0.55 * Math.sqrt(dist) + hOff;
  const hw = 0.01 + 0.055 * w, k = 0.5 * (0.2 + 0.8 * Math.pow(w, 0.8));
  const r = color.r * k, g = color.g * k, b = color.b * k;
  let px = 0, py = 0, pz = 0, ox = 0, oy = 0;
  for (let s = 0; s <= SEG; s++) {
    const t = s / SEG, u = 1 - t;
    const x = u * u * ax + 2 * u * t * cx + t * t * bx;
    const y = u * u * ay + 2 * u * t * cy + t * t * by;
    const z = 0.34 + 2 * u * t * cz;
    // ribbon width runs across the curve, in the plane facing the viewer
    let tx = 2 * u * (cx - ax) + 2 * t * (bx - cx), ty = 2 * u * (cy - ay) + 2 * t * (by - cy);
    let wx = ty, wy = -tx;
    const len = Math.hypot(wx, wy) || 1;
    wx = (wx / len) * hw; wy = (wy / len) * hw;
    if (s > 0) {
      const q = [px - ox, py - oy, pz, px + ox, py + oy, pz, x - wx, y - wy, z, px + ox, py + oy, pz, x + wx, y + wy, z, x - wx, y - wy, z];
      for (let i = 0; i < 18; i++) pos[n + i] = q[i];
      for (let i = 0; i < 6; i++) { col[n + i * 3] = r; col[n + i * 3 + 1] = g; col[n + i * 3 + 2] = b; }
      n += 18;
    }
    px = x; py = y; pz = z; ox = wx; oy = wy;
  }
  return n;
}

function render(i) {
  const st = S.steps[i];
  if (!st || !G.root) return;
  S.shown = i;
  const { L, H, F } = S;
  const tr = st.trace, Sx = tr.S, cur = tr.off + Sx - 1;
  S.cur = cur;
  const first = cur - VIS + 1;  // absolute index shown in wall column 0
  G.hint.on = false;

  // how much each context token is read right now (sum over the heads that are switched on)
  const recv = new Float32Array(Sx);
  let recvMax = 1e-9;
  for (let l = 0; l < L; l++) for (let h = 0; h < H; h++) {
    if (!S.headOn[h]) continue;
    const o = (l * H + h) * Sx;
    for (let j = 0; j < Sx; j++) recv[j] += tr.attn[o + j] / 255;
  }
  for (let j = 0; j < Sx; j++) recvMax = Math.max(recvMax, recv[j]);

  // token row + residual wall
  for (let k = 0; k < VIS; k++) {
    const abs = first + k, j = abs - tr.off;
    const r = j >= 0 && j < Sx ? recv[j] / recvMax : 0;
    if (abs < 0) _c.setRGB(0.03, 0.035, 0.05);
    else if (abs === cur) _c.copy(C_HOT);
    else _c.copy(C_TILE).lerp(C_READ, Math.pow(r, 0.7));
    G.tiles.setColorAt(k, _c);
  }
  G.tiles.instanceColor.needsUpdate = true;
  for (let l = 0; l < L; l++) {
    let lo = Infinity, hi = -Infinity;
    for (let k = 0; k < VIS; k++) { const v = first + k > 0 ? S.resid[l][first + k] : undefined; if (v !== undefined) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }
    for (let k = 0; k < VIS; k++) {
      const abs = first + k, v = abs >= 0 && abs <= cur ? S.resid[l][abs] : undefined;
      if (v === undefined) _c.setRGB(0.03, 0.035, 0.05);
      else {
        const t = hi > lo ? Math.max(0, Math.min(1, (v - lo) / (hi - lo))) : 0.5;
        _c.copy(C_COLD).lerp(C_HOT, 0.1 + 0.9 * t * t);
        if (abs === cur) _c.lerp(C_WHITE, 0.45);
      }
      G.wall.setColorAt(l * VIS + k, _c);
    }
  }
  G.wall.instanceColor.needsUpdate = true;

  // attention ribbons: from the layer below at the source token, up into this layer at the current token
  const best = new Int32Array(TOPK), bestW = new Float32Array(TOPK);
  for (let l = 0; l < L; l++) {
    const geo = G.arcs[l].geometry, pos = geo.attributes.position.array, col = geo.attributes.color.array;
    let n = 0;
    const ya = l === 0 ? 0.12 : yOf(l - 1), yb = yOf(l);
    for (let h = 0; h < H; h++) {
      if (!S.headOn[h]) continue;
      const o = (l * H + h) * Sx;
      bestW.fill(0); best.fill(-1);
      for (let j = Math.max(0, first - tr.off); j < Sx; j++) {
        const w = tr.attn[o + j] / 255;
        if (w < THR) continue;
        let m = 0;
        for (let q = 1; q < TOPK; q++) if (bestW[q] < bestW[m]) m = q;
        if (w > bestW[m]) { bestW[m] = w; best[m] = j; }
      }
      const color = headColor(h);
      for (let q = 0; q < TOPK; q++) {
        if (best[q] < 0) continue;
        const ax = (best[q] + tr.off - cur) * DX;
        n = addArc(pos, col, n, ax, ya, 0, yb, bestW[q], color, (h / H) * 0.5);
      }
    }
    geo.setDrawRange(0, n / 3);
    geo.attributes.position.needsUpdate = true;
    geo.attributes.color.needsUpdate = true;
  }

  // mlp panels
  for (let l = 0; l < L; l++) {
    const tex = G.panels[l].material.map, px = tex.image.data;
    for (let f = 0; f < F; f++) {
      const v = tr.mlp[l * F + f] / 127, mag = Math.pow(Math.abs(v), 0.6);
      const row = S.gridR - 1 - Math.floor(f / S.gridC), p = (row * S.gridC + (f % S.gridC)) * 4;
      if (v >= 0) { px[p] = 255; px[p + 1] = 174; px[p + 2] = 87; } else { px[p] = 95; px[p + 1] = 160; px[p + 2] = 255; }
      px[p + 3] = 255 * (0.05 + 0.95 * mag);
    }
    tex.needsUpdate = true;
  }

  // pinned labels: each layer's early guess, the tokens being read, the token that came out
  tr.lens.forEach((g, l) => {
    const o = G.lens[l];
    o.on = true;
    o.el.className = 'lab lens' + (g.id === st.id ? ' hit' : '');
    o.el.innerHTML = '';
    const b = document.createElement('b');
    b.textContent = show(g.text);
    o.el.append(b, ` ${(g.p * 100).toFixed(0)}%`);
  });
  const order = Array.from(recv.keys()).filter((j) => j + tr.off !== cur && j + tr.off >= first && j + tr.off > 0)
    .sort((a, b) => recv[b] - recv[a]).slice(0, TOPK).sort((a, b) => a - b);
  G.tokLabs.forEach((o) => (o.on = false));
  const put = (o, abs, cls, lane) => {
    o.on = true;
    o.el.className = 'lab tok' + cls;
    o.el.textContent = show(S.tokens[abs]?.text ?? '?');
    o.pos.set((abs - cur) * DX, -0.8 - lane * 0.66, 0.3);
  };
  put(G.tokLabs[0], cur, ' cur', 0);
  order.forEach((j, n) => put(G.tokLabs[n + 1], j + tr.off, '', 1 + (n % 3)));
  G.next.on = true;
  G.next.el.textContent = `→ ${show(st.text)}`;

  // side panel + text strip
  $('top').replaceChildren(...tr.top.map((t) => {
    const d = document.createElement('div');
    d.className = 'bar' + (t.id === st.id ? ' pick' : '');
    const a = document.createElement('span'); a.className = 't'; a.textContent = show(t.text);
    const b = document.createElement('span'); b.className = 'b';
    const bi = document.createElement('i'); bi.style.width = `${(t.p * 100).toFixed(1)}%`; b.append(bi);
    const p = document.createElement('span'); p.className = 'p'; p.textContent = `${(t.p * 100).toFixed(0)}%`;
    d.append(a, b, p);
    return d;
  }));
  $('lens').replaceChildren(...tr.lens.map((g, l) => {
    const d = document.createElement('div');
    d.className = 'lensrow' + (g.id === st.id ? ' hit' : '');
    const a = document.createElement('span'); a.className = 'l'; a.textContent = `L${l}`;
    const b = document.createElement('span'); b.className = 't'; b.textContent = show(g.text);
    const p = document.createElement('span'); p.className = 'p'; p.textContent = `${(g.p * 100).toFixed(0)}%`;
    d.append(a, b, p);
    return d;
  }).reverse());
  S.spans.forEach((sp, abs) => {
    const j = abs - tr.off, r = j >= 0 && j < Sx ? recv[j] / recvMax : 0;
    sp.style.background = abs <= cur && r > 0.02 ? `rgba(255,210,122,${(0.5 * Math.pow(r, 0.8)).toFixed(3)})` : '';
    sp.classList.toggle('cur', abs === cur);
    sp.classList.toggle('nxt', abs === cur + 1);
    sp.classList.toggle('pending', abs > cur + 1);
  });
  $('stepline').textContent = `token ${i + 1} of ${S.steps.length}${S.source ? ' · from ' + S.source : ''} · the model read “${show(S.tokens[cur]?.text ?? '')}” and wrote “${show(st.text)}”`;
  S.stepTime = performance.now();
}

// ------------------------------------------------------------------ events from the server
const b64u8 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));

function addSpan(tok, cls) {
  const sp = document.createElement('span');
  sp.textContent = tok.text === '<s>' ? '' : tok.text;
  sp.className = cls;
  const abs = S.spans.length;
  sp.addEventListener('mouseenter', () => (S.hoverAbs = abs));
  sp.addEventListener('mouseleave', () => (S.hoverAbs = -1));
  sp.addEventListener('click', () => {
    const i = Math.max(0, Math.min(S.steps.length - 1, abs - S.nPrompt));
    if (!S.steps.length) return;
    S.follow = false;
    $('live').disabled = false;
    render(i);
  });
  S.spans.push(sp);
  $('text').append(sp);
}

function onEvent(ev) {
  if (ev.type === 'hello') {
    S.info = ev.model;
    const c = ev.model.config;
    const rebuild = S.L !== c.n_layers || S.H !== c.n_heads || S.F !== c.ffn_dim || !G.root;
    S.L = c.n_layers; S.H = c.n_heads; S.F = c.ffn_dim;
    $('modelline').textContent = `${ev.model.name} · ${(ev.model.n_params / 1e6).toFixed(1)}m params · ${c.n_layers} layers × ${c.n_heads} heads · ${ev.model.device}`;
    S.steer = new Map(ev.state.steering.map((s) => [s.layer, s.alpha]));
    S.edited = new Set(ev.state.changed);
    if (rebuild) {
      S.headOn = Array(S.H).fill(true);
      S.resid = Array.from({ length: S.L }, () => []);
      buildScene();
      buildHeads();
    } else { refreshMods(); Object.keys(G.wt).forEach(loadTile); }
  } else if (ev.type === 'start') {
    S.tokens = ev.tokens.slice(); S.nPrompt = ev.tokens.length; S.steps = []; S.shown = -1; S.follow = true;
    S.resid = Array.from({ length: S.L }, () => []);
    S.source = ev.source; S.spans = [];
    $('text').replaceChildren();
    ev.tokens.forEach((t) => addSpan(t, 'p'));
    $('live').disabled = true;
    $('stats').textContent = 'generating…';
  } else if (ev.type === 'token' && ev.trace) {
    const t = ev.trace, sh = t.shape;
    const tr = { S: sh.ctx, off: t.ctx_offset, attn: b64u8(t.attn), mlp: new Int8Array(b64u8(t.mlp).buffer), lens: t.lens, top: t.top };
    const cur = tr.off + tr.S - 1;
    t.resid.forEach((row, l) => row.forEach((v, n) => (S.resid[l][cur - row.length + 1 + n] = v)));
    S.tokens.push({ id: ev.id, text: ev.text });
    addSpan({ text: ev.text }, 'pending');
    S.steps.push({ id: ev.id, text: ev.text, p: ev.p, trace: tr });
  } else if (ev.type === 'done') {
    const u = ev.usage;
    $('stats').textContent = `${u.completion_tokens} tokens · ${u.tokens_per_s.toFixed(0)} tok/s · via ${S.source || '?'}`;
  } else if (ev.type === 'steer') {
    S.steer.set(ev.layer, ev.alpha); refreshMods();
  } else if (ev.type === 'steer_cleared') {
    S.steer.clear(); refreshMods();
  } else if (ev.type === 'edit') {
    S.edited.add(ev.tensor); loadTile(ev.tensor);
  } else if (ev.type === 'reset') {
    S.steer.clear(); const was = [...S.edited]; S.edited.clear(); refreshMods(); was.forEach(loadTile);
  }
}

function connect() {
  const es = new EventSource('/api/events');
  es.onopen = () => $('conn').classList.add('on');
  es.onerror = () => { $('conn').classList.remove('on'); $('modelline').textContent = 'server not reachable, retrying…'; };
  es.onmessage = (m) => { try { onEvent(JSON.parse(m.data)); } catch (e) { console.error(e); } };
}

// ------------------------------------------------------------------ controls
function buildHeads() {
  $('heads').replaceChildren(...S.headOn.map((_, h) => {
    const b = document.createElement('button');
    b.textContent = h;
    b.style.background = '#' + headColor(h).getHexString();
    b.title = `head ${h}: click to hide or show its attention`;
    b.addEventListener('click', () => {
      S.headOn[h] = !S.headOn[h];
      b.classList.toggle('off', !S.headOn[h]);
      if (S.shown >= 0) render(S.shown);
    });
    return b;
  }));
}

async function post(path, body, method = 'POST') {
  const r = await fetch(path, { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error?.message || r.statusText);
  return j;
}
const note = (id, text, cls = '') => { const n = $(id); n.textContent = text; n.className = 'note ' + cls; };
const lines = (id) => $(id).value.split('\n').map((s) => s.trim()).filter(Boolean);

$('go').addEventListener('click', async () => {
  S.genAbort?.abort();
  const ac = (S.genAbort = new AbortController());
  $('go').disabled = true; $('stop').disabled = false;
  try {
    const r = await fetch('/api/generate', { method: 'POST', signal: ac.signal, headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: $('prompt').value, max_tokens: +$('maxtok').value || 60, temperature: +$('temp').value, stream: true, source: 'viewer' }) });
    const rd = r.body.getReader();
    while (!(await rd.read()).done) { /* the visuals come from /api/events; just drain this */ }
  } catch (e) { if (e.name !== 'AbortError') $('stats').textContent = 'error: ' + e.message; }
  if (S.genAbort === ac) { $('go').disabled = false; $('stop').disabled = true; }
});
$('stop').addEventListener('click', () => S.genAbort?.abort());
$('prompt').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) $('go').click(); });
$('rate').addEventListener('input', () => { S.rate = +$('rate').value; $('ratev').textContent = S.rate > 30 ? 'live' : `${S.rate}/s`; });
$('alpha').addEventListener('input', () => ($('alphav').textContent = (+$('alpha').value).toFixed(1)));
$('live').addEventListener('click', () => { S.follow = true; $('live').disabled = true; });
$('steer').addEventListener('click', async () => {
  note('steernote', 'finding the direction…');
  try {
    const r = await post('/api/steer', { pos: lines('spos'), neg: lines('sneg'), alpha: +$('alpha').value });
    note('steernote', `pushing layer ${r.layer} along it (×${r.alpha.toFixed(1)}). generate again to see it.`, 'good');
  } catch (e) { note('steernote', e.message, 'bad'); }
});
$('unsteer').addEventListener('click', async () => { await post('/api/steer', {}, 'DELETE'); note('steernote', 'steering cleared'); });
$('edit').addEventListener('click', async () => {
  note('editnote', 'solving the rank-one update… (a few seconds)');
  $('edit').disabled = true;
  try {
    const r = await post('/api/edit', { prompt: $('fprompt').value, target: $('ftarget').value });
    note('editnote', `written into layer ${r.layer}'s down matrix: log-prob ${r.logp_before.toFixed(2)} → ${r.logp_after.toFixed(2)}`, 'good');
    $('prompt').value = $('fprompt').value;
  } catch (e) { note('editnote', e.message, 'bad'); }
  $('edit').disabled = false;
});
$('reset').addEventListener('click', async () => { await post('/api/reset', {}); note('editnote', ''); note('steernote', 'model reset to the weights on disk'); });

// ------------------------------------------------------------------ frame loop
function frame(now) {
  requestAnimationFrame(frame);
  if (G.root) {
    if (S.follow && S.shown < S.steps.length - 1) {
      if (S.rate > 30) render(S.steps.length - 1);
      else if (now - S.lastAdvance >= 1000 / S.rate) { S.lastAdvance = now; render(S.shown + 1); }
    }
    // the reveal: activity climbs the stack layer by layer after each token
    const dur = Math.min(520, 900 / Math.min(S.rate, 30)), stag = (dur * 0.75) / S.L, fade = dur * 0.4;
    const el = now - S.stepTime;
    for (let l = 0; l < S.L; l++) {
      const t = S.shown < 0 ? 0 : Math.max(0, Math.min(1, (el - l * stag) / fade));
      G.arcs[l].material.opacity = t;
      G.panels[l].material.opacity = 0.15 + 0.85 * t;
    }
    const pt = Math.max(0, Math.min(1, el / dur));
    G.pulse.visible = S.shown >= 0 && pt < 1;
    G.pulse.position.y = pt * yOf(S.L - 1);
    const hv = S.hoverAbs >= 0 && S.cur >= 0 && S.hoverAbs <= S.cur && S.hoverAbs > S.cur - VIS;
    G.hover.visible = hv;
    if (hv) G.hover.position.set((S.hoverAbs - S.cur) * DX, G.spine.position.y, 0);
  }
  controls.update();
  renderer.render(scene, camera);
  placeLabels();
}

resize();
connect();
requestAnimationFrame(frame);
window.__shellm = { S, G, render };  // handy from the browser console
