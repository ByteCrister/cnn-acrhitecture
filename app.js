'use strict';

/* ================================================================
   GLOBALS
   ================================================================ */
let model = null;
let vizModel = null;
let trainData = null;
let testData  = null;
let trainArrays = null;
let testArrays  = null;

let isTraining = false;
let isPaused   = false;
let stopFlag   = false;
let continueResolver = null;

let lossChartInst = null;
let accChartInst  = null;

let convInputData  = [];
let convKernelData = [];
let convAnimHandle = null;

let poolData = [];
let flatData = [];

let drawing = false;
let drawCtx = null;

/* ================================================================
   UTILITIES
   ================================================================ */
function drawHeatmap(canvas, data2d, opts = {}) {
  const rows = data2d.length;
  const cols = data2d[0].length;
  canvas.width  = opts.w || canvas.width;
  canvas.height = opts.h || canvas.height;
  const ctx = canvas.getContext('2d');
  const cw = canvas.width / cols;
  const ch = canvas.height / rows;

  let mn = Infinity, mx = -Infinity;
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      mn = Math.min(mn, data2d[r][c]);
      mx = Math.max(mx, data2d[r][c]);
    }
  if (mx === mn) mx = mn + 1e-6;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const t = (data2d[r][c] - mn) / (mx - mn);
      const [rr, gg, bb] = heatColor(t);
      ctx.fillStyle = `rgb(${rr},${gg},${bb})`;
      ctx.fillRect(c * cw, r * ch, cw, ch);
    }
  }
}

function heatColor(t) {
  if (t < 0.33) {
    const s = t / 0.33;
    return [Math.round(s * 108), Math.round(s * 99), Math.round(s * 255)];
  } else if (t < 0.66) {
    const s = (t - 0.33) / 0.33;
    return [Math.round(108 + s * (73 - 108)), Math.round(99 + s * (182 - 99)), Math.round(255 + s * (229 - 255))];
  } else {
    const s = (t - 0.66) / 0.34;
    return [Math.round(73 + s * (247 - 73)), Math.round(182 + s * (201 - 182)), Math.round(229 + s * 26)];
  }
}

function flat2d(arr, h, w) {
  const out = [];
  for (let r = 0; r < h; r++) {
    out.push(Array.from(arr.slice(r * w, r * w + w)));
  }
  return out;
}

function drawVectorStrip(canvas, values, h = 24) {
  const n = values.length;
  canvas.width  = n * 3;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  let mn = Math.min(...values), mx = Math.max(...values);
  if (mx === mn) mx = mn + 1e-6;
  for (let i = 0; i < n; i++) {
    const t = (values[i] - mn) / (mx - mn);
    const [rr, gg, bb] = heatColor(t);
    ctx.fillStyle = `rgb(${rr},${gg},${bb})`;
    ctx.fillRect(i * 3, 0, 3, h);
  }
}

function renderProbBars(container, probs, predicted) {
  container.innerHTML = '';
  for (let i = 0; i < 10; i++) {
    const pct = (probs[i] * 100).toFixed(1);
    const row = document.createElement('div');
    row.className = 'prob-row';
    const isTop = (i === predicted);
    row.innerHTML = `
      <div class="prob-lbl">${i}</div>
      <div class="prob-bar-bg">
        <div class="prob-bar-fill${isTop?' top':''}" style="width:${pct}%"></div>
      </div>
      <div class="prob-pct">${pct}%</div>`;
    container.appendChild(row);
  }
}

function makeHeatmapEl(data2d, size, label) {
  const wrap = document.createElement('div');
  wrap.className = 'heatmap-wrap';
  const c = document.createElement('canvas');
  c.width  = size;
  c.height = size;
  drawHeatmap(c, data2d, {w: size, h: size});
  const lbl = document.createElement('div');
  lbl.className = 'heatmap-label';
  lbl.textContent = label || '';
  wrap.appendChild(c);
  wrap.appendChild(lbl);
  return wrap;
}

/* ================================================================
   SECTION TOGGLE
   ================================================================ */
function toggleSection(id) {
  const header = document.querySelector(`#${id} .section-header`);
  const body   = document.querySelector(`#${id} .section-body`);
  const open   = body.classList.contains('open');
  if (open) {
    body.classList.remove('open');
    header.classList.remove('open');
  } else {
    body.classList.add('open');
    header.classList.add('open');
  }
}

['sec1','sec2','sec3','sec4','sec5','sec6'].forEach(id => {
  const hdr = document.querySelector(`#${id} .section-header`);
  const bdy = document.querySelector(`#${id} .section-body`);
  if (hdr) hdr.classList.add('open');
  if (bdy) bdy.classList.add('open');
});

/* ================================================================
   SECTION 1 — CNN BASICS
   ================================================================ */
const CONV_ROWS = 6, CONV_COLS = 6;
const KERN_SIZE = 3;

function initConvDemo() {
  convInputData = Array.from({length: CONV_ROWS}, () =>
    Array.from({length: CONV_COLS}, () => Math.round(Math.random() * 9))
  );
  convKernelData = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]];
  renderConvGrid();
}

function renderConvGrid() {
  const inputDiv = document.getElementById('convInput');
  inputDiv.innerHTML = '';
  inputDiv.style.gridTemplateColumns = `repeat(${CONV_COLS}, 38px)`;
  for (let r = 0; r < CONV_ROWS; r++) {
    for (let c = 0; c < CONV_COLS; c++) {
      const inp = document.createElement('input');
      inp.type = 'number'; inp.min = '0'; inp.max = '9'; inp.step = '1';
      inp.value = convInputData[r][c];
      inp.style.cssText = 'width:38px;height:38px;text-align:center;padding:0;font-size:0.85rem;';
      inp.style.background = cellColor(convInputData[r][c], 0, 9);
      inp.style.color = convInputData[r][c] > 5 ? '#111' : '#eee';
      inp.addEventListener('input', () => {
        convInputData[r][c] = parseFloat(inp.value) || 0;
        inp.style.background = cellColor(convInputData[r][c], 0, 9);
        inp.style.color = convInputData[r][c] > 5 ? '#111' : '#eee';
        computeConvOutput();
      });
      inputDiv.appendChild(inp);
    }
  }

  const kernDiv = document.getElementById('convKernel');
  kernDiv.innerHTML = '';
  kernDiv.style.gridTemplateColumns = `repeat(${KERN_SIZE}, 44px)`;
  for (let r = 0; r < KERN_SIZE; r++) {
    for (let c = 0; c < KERN_SIZE; c++) {
      const inp = document.createElement('input');
      inp.type = 'number'; inp.step = '0.1';
      inp.value = convKernelData[r][c];
      inp.style.cssText = 'width:44px;height:44px;text-align:center;padding:0;font-size:0.85rem;';
      const mn = -3, mx = 3;
      inp.style.background = cellColor(convKernelData[r][c], mn, mx);
      inp.style.color = Math.abs(convKernelData[r][c]) < 1 ? '#eee' : '#111';
      inp.addEventListener('input', () => {
        convKernelData[r][c] = parseFloat(inp.value) || 0;
        computeConvOutput();
        renderConvGrid();
      });
      kernDiv.appendChild(inp);
    }
  }
  computeConvOutput();
}

function cellColor(val, mn, mx) {
  const t = (val - mn) / (mx - mn);
  const [r, g, b] = heatColor(Math.max(0, Math.min(1, t)));
  return `rgb(${r},${g},${b})`;
}

function computeConvOutput(highlightR = -1, highlightC = -1) {
  const stride  = parseInt(document.getElementById('strideSlider').value);
  const padding = document.getElementById('paddingCheck').checked;

  let inp = convInputData;
  let padded = inp;
  if (padding) {
    padded = Array.from({length: CONV_ROWS + 2}, (_, r) =>
      Array.from({length: CONV_COLS + 2}, (_, c) => {
        if (r === 0 || r === CONV_ROWS + 1 || c === 0 || c === CONV_COLS + 1) return 0;
        return inp[r - 1][c - 1];
      })
    );
  }
  const ph = padded.length, pw = padded[0].length;
  const oh = Math.floor((ph - KERN_SIZE) / stride) + 1;
  const ow = Math.floor((pw - KERN_SIZE) / stride) + 1;

  const output = [];
  for (let r = 0; r < oh; r++) {
    const row = [];
    for (let c = 0; c < ow; c++) {
      let sum = 0;
      for (let kr = 0; kr < KERN_SIZE; kr++)
        for (let kc = 0; kc < KERN_SIZE; kc++)
          sum += padded[r * stride + kr][c * stride + kc] * convKernelData[kr][kc];
      row.push(sum);
    }
    output.push(row);
  }

  const canvas = document.getElementById('convOutputCanvas');
  canvas.width = 120; canvas.height = 120;
  drawHeatmap(canvas, output, {w: 120, h: 120});

  if (highlightR >= 0 && highlightC >= 0) {
    const ctx = canvas.getContext('2d');
    const cw = 120 / ow, ch = 120 / oh;
    ctx.strokeStyle = '#f7c948';
    ctx.lineWidth = 2;
    ctx.strokeRect(highlightC * cw + 1, highlightR * ch + 1, cw - 2, ch - 2);
  }

  document.getElementById('convOutSize').textContent = `Output: ${oh}×${ow}`;
}

function randomiseConvInput() {
  convInputData = Array.from({length: CONV_ROWS}, () =>
    Array.from({length: CONV_COLS}, () => Math.round(Math.random() * 9))
  );
  renderConvGrid();
}

const KERNEL_PRESETS = {
  'edge-h':  [[-1,-1,-1],[0,0,0],[1,1,1]],
  'edge-v':  [[-1,0,1],[-1,0,1],[-1,0,1]],
  'blur':    [[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]],
  'sharpen': [[0,-1,0],[-1,5,-1],[0,-1,0]],
};
function setKernelPreset(name) {
  convKernelData = KERNEL_PRESETS[name].map(r => [...r]);
  renderConvGrid();
}

function updateConvParams() {
  document.getElementById('strideVal').textContent = document.getElementById('strideSlider').value;
  renderConvGrid();
}

let convAnimating = false;
async function animateConv() {
  if (convAnimating) return;
  convAnimating = true;
  const stride  = parseInt(document.getElementById('strideSlider').value);
  const padding = document.getElementById('paddingCheck').checked;
  let inp = convInputData;
  let padded = inp;
  if (padding) {
    padded = Array.from({length: CONV_ROWS + 2}, (_, r) =>
      Array.from({length: CONV_COLS + 2}, (_, c) => {
        if (r === 0 || r === CONV_ROWS + 1 || c === 0 || c === CONV_COLS + 1) return 0;
        return inp[r - 1][c - 1];
      })
    );
  }
  const ph = padded.length, pw = padded[0].length;
  const oh = Math.floor((ph - KERN_SIZE) / stride) + 1;
  const ow = Math.floor((pw - KERN_SIZE) / stride) + 1;

  for (let r = 0; r < oh; r++) {
    for (let c = 0; c < ow; c++) {
      computeConvOutput(r, c);
      await sleep(90);
    }
  }
  computeConvOutput();
  convAnimating = false;
}

/* ----- ACTIVATION DEMO ----- */
let actMatrix = [];
function initActivationDemo() { randomActMatrix(); }
function randomActMatrix() {
  actMatrix = Array.from({length: 5}, () =>
    Array.from({length: 5}, () => (Math.random() * 4 - 2).toFixed(2) * 1)
  );
  runActivation();
}
function runActivation() {
  const fn  = document.getElementById('actFnSelect').value;
  const cBefore = document.getElementById('actBeforeCanvas');
  const cAfter  = document.getElementById('actAfterCanvas');
  drawHeatmap(cBefore, actMatrix, {w: 130, h: 130});
  const after = actMatrix.map(row => row.map(v =>
    fn === 'relu' ? Math.max(0, v) : 1 / (1 + Math.exp(-v))
  ));
  drawHeatmap(cAfter, after, {w: 130, h: 130});
  const fEl = document.getElementById('actFormula');
  fEl.textContent = fn === 'relu' ? 'ReLU(x) = max(0, x)' : 'σ(x) = 1 / (1 + e^-x)';
}

/* ----- POOLING DEMO ----- */
function initPoolDemo() {
  poolData = Array.from({length: 4}, () =>
    Array.from({length: 4}, () => Math.round(Math.random() * 9))
  );
  runPool();
}
function runPool() {
  drawHeatmap(document.getElementById('poolInputCanvas'), poolData, {w: 130, h: 130});
  const out = [[0,0],[0,0]];
  const info = [];
  for (let r = 0; r < 2; r++) {
    for (let c = 0; c < 2; c++) {
      const vals = [
        poolData[r*2][c*2], poolData[r*2][c*2+1],
        poolData[r*2+1][c*2], poolData[r*2+1][c*2+1]
      ];
      out[r][c] = Math.max(...vals);
      info.push(`max(${vals.join(',')})=${out[r][c]}`);
    }
  }
  drawHeatmap(document.getElementById('poolOutputCanvas'), out, {w: 130, h: 130});
  document.getElementById('poolValues').innerHTML = info.map(s =>
    `<div style="color:var(--accent3)">${s}</div>`).join('');
}

/* ----- FLATTEN DEMO ----- */
function initFlatDemo() {
  flatData = Array.from({length: 2}, () =>
    Array.from({length: 2}, () => parseFloat((Math.random() * 2).toFixed(2)))
  );
  const W = Array.from({length: 3}, () =>
    Array.from({length: 4}, () => parseFloat((Math.random() * 2 - 1).toFixed(2)))
  );
  const b = Array.from({length: 3}, () => parseFloat((Math.random() * 0.5).toFixed(2)));
  const flat = flatData[0].concat(flatData[1]);
  const neurons = W.map((wRow, i) => wRow.reduce((s, w, j) => s + w * flat[j], b[i]));
  const relu = neurons.map(v => Math.max(0, v));

  drawHeatmap(document.getElementById('flatInputCanvas'), flatData, {w: 80, h: 80});
  drawVectorStrip(document.getElementById('flatVecCanvas'), flat, 28);
  document.getElementById('flatVecCanvas').width = 4 * 28;

  const nc = document.getElementById('flatDenseCanvas');
  nc.width = 40; nc.height = 90;
  const ctx = nc.getContext('2d');
  ctx.clearRect(0,0,40,90);
  relu.forEach((v, i) => {
    const t = Math.max(0, Math.min(1, v));
    const [r, g, b_] = heatColor(t);
    ctx.fillStyle = `rgb(${r},${g},${b_})`;
    ctx.beginPath(); ctx.arc(20, 15 + i * 30, 12, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.textAlign = 'center'; ctx.font = '9px JetBrains Mono';
    ctx.fillText(v.toFixed(1), 20, 19 + i * 30);
  });
  document.getElementById('fcFormula').textContent = `y = ReLU(Wx + b) → [${relu.map(v=>v.toFixed(2)).join(', ')}]`;
}

/* ================================================================
   FULL CNN PIPELINE (Section 1, bottom)
   ================================================================ */
const PIPELINE_INPUT_SIZE = 5;
const PIPELINE_KERN_SIZE = 3;
let pipelineInputData = [];
let pipelineKernelData = [];
let pipelineAnimRunning = false;
let pipelineAnimAbort = false;
let pipelinePaused = false;
/** @type {{resolve: Function|null}|null} */
let pipelinePauseWait = null;
let pipelineInputCells = []; // [r][c] -> HTMLElement

function pipelineClamp01(t) {
  return Math.max(0, Math.min(1, t));
}

function syncPipelinePaddingClass() {
  const grid = document.getElementById('pipelineInputGrid');
  if (!grid) return;
  const on = !!(document.getElementById('pipelinePadding')?.checked);
  grid.classList.toggle('pipeline-input-grid--pad', on);
}

function clearPipelineConvHighlight() {
  const grid = document.getElementById('pipelineInputGrid');
  if (!grid) return;
  grid.querySelectorAll('.pipeline-cell-conv-highlight').forEach(el => {
    el.classList.remove('pipeline-cell-conv-highlight');
  });
}

/** Visual + index layout matches conv tensor: 5×5 (no pad) or 7×7 with border of implicit zeros. */
function setPipelineConvHighlight(paddedOriginR, paddedOriginC, _stride, padding) {
  clearPipelineConvHighlight();
  const inpDiv = document.getElementById('pipelineInputGrid');
  if (!inpDiv) return;
  const dim = padding ? PIPELINE_INPUT_SIZE + 2 : PIPELINE_INPUT_SIZE;
  for (let kr = 0; kr < PIPELINE_KERN_SIZE; kr++) {
    for (let kc = 0; kc < PIPELINE_KERN_SIZE; kc++) {
      const pr = paddedOriginR + kr;
      const pc = paddedOriginC + kc;
      if (pr < 0 || pc < 0 || pr >= dim || pc >= dim) continue;
      const el = inpDiv.children[pr * dim + pc];
      if (el) el.classList.add('pipeline-cell-conv-highlight');
    }
  }
}

function pipelineAppendTrace(traceEl, idx, expr, val) {
  const row = document.createElement('div');
  row.className = 'calc-trace-row';
  row.innerHTML =
    `<span class="calc-trace-idx">${idx}</span>` +
    `<span class="calc-trace-expr">${expr}</span>` +
    `<span class="calc-trace-val">${val}</span>`;
  traceEl.appendChild(row);
  traceEl.scrollTop = traceEl.scrollHeight;
}

function pipelineDeterministicDenseParams(flatVec, nNeurons) {
  const nIn = flatVec.length;
  let h = 2166136261 >>> 0;
  flatVec.forEach((v, i) => {
    const iv = Math.floor((v + 999) * 10000 + (i + 1) * 97);
    h = Math.imul(h ^ iv, 16777619) >>> 0;
  });
  const W = [];
  for (let i = 0; i < nNeurons; i++) {
    const row = [];
    for (let j = 0; j < nIn; j++) {
      const seed = (((h >>> 0) ^ (i + 1) * 92837111 ^ j * 2654435761) >>> 0) / 4294967295;
      row.push(parseFloat((Math.sin(seed * Math.PI * 2) * 0.85).toFixed(4)));
      h ^= (j + i * 13) >>> 0;
    }
    W.push(row);
  }
  const b = [];
  for (let i = 0; i < nNeurons; i++) {
    const seed = (((h >>> 0) ^ (i + 1) * 1103515245) >>> 0) / 4294967295;
    b.push(parseFloat((Math.cos(seed * Math.PI * 2) * 0.25).toFixed(4)));
  }
  return { W, b };
}

/** Build padded tensor for convolution; pad = 1 when "same" ON, else 0. */
function pipelineBuildPadded(input, padding) {
  if (!padding) return input.map(r => [...r]);
  const s = PIPELINE_INPUT_SIZE;
  return Array.from({ length: s + 2 }, (_, r) =>
    Array.from({ length: s + 2 }, (_, c) => {
      if (r === 0 || r === s + 1 || c === 0 || c === s + 1) return 0;
      return input[r - 1][c - 1];
    })
  );
}

function pipelineGridMinMax(mat) {
  let mn = Infinity, mx = -Infinity;
  mat.forEach(row => row.forEach(v => {
    mn = Math.min(mn, v);
    mx = Math.max(mx, v);
  }));
  if (mx === mn) mx = mn + 1e-9;
  return { mn, mx };
}

/** Resize canvas to fit grid with sensible cell size caps. */
function pipelineSizeCanvasForGrid(canvas, rows, cols, minCell = 22, maxCell = 40) {
  const target = Math.round(Math.min(maxCell, Math.max(minCell,
    cols > rows ? 200 / cols : 200 / rows)));
  const cw = Math.max(Math.round(target * cols), cols * minCell * 0.5);
  const ch = Math.max(Math.round(target * rows), rows * minCell * 0.5);
  canvas.width = cw;
  canvas.height = ch;
}

/**
 * Heatmap grid with centered numeric labels; optional outer highlight rectangle
 * in OUTPUT grid coords (yellow border region).
 */
function pipelineDrawHeatNumberGrid(canvas, data2d, highlightOrNull) {
  const rows = data2d.length;
  const cols = data2d[0].length;
  pipelineSizeCanvasForGrid(canvas, rows, cols, 26, 48);
  const ctx = canvas.getContext('2d');
  const { mn, mx } = pipelineGridMinMax(data2d);
  const cw = canvas.width / cols;
  const ch = canvas.height / rows;

  ctx.fillStyle = '#1e1e2e';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = data2d[r][c];
      const t = pipelineClamp01((v - mn) / (mx - mn));
      const [rr, gg, bb] = heatColor(t);
      ctx.fillStyle = `rgb(${rr},${gg},${bb})`;
      ctx.fillRect(c * cw, r * ch, cw, ch);
      ctx.strokeStyle = 'rgba(255,255,255,0.18)';
      ctx.lineWidth = 1;
      ctx.strokeRect(c * cw + 0.5, r * ch + 0.5, cw - 1, ch - 1);
      ctx.fillStyle = t > 0.55 ? '#111' : '#eef';
      ctx.font = `600 ${Math.max(10, Math.min(cw, ch) * 0.32)}px JetBrains Mono, monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const label = typeof v === 'number' ? (Math.abs(v) >= 100 ? v.toFixed(0) :
        Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2)) : String(v);
      ctx.fillText(label, (c + 0.5) * cw, (r + 0.55) * ch);
    }
  }

  if (highlightOrNull && typeof highlightOrNull.r0 === 'number' && highlightOrNull.r0 >= 0) {
    const { r0, c0 } = highlightOrNull;
    const hr = Math.max(1, Math.min(rows - r0, highlightOrNull.h || 2));
    const hc = Math.max(1, Math.min(cols - c0, highlightOrNull.w || 2));
    ctx.strokeStyle = '#f7c948';
    ctx.lineWidth = 3;
    ctx.strokeRect(c0 * cw + 2, r0 * ch + 2, hc * cw - 4, hr * ch - 4);
  }
}

function pipelineDrawVectorStrip(canvas, values, revealedCount) {
  const n = values.length;
  const h = 28;
  if (n === 0) {
    canvas.width = 40;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#1e1e2e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    return;
  }
  const cellW = Math.max(22, Math.min(44, Math.floor(240 / n)));
  canvas.width = cellW * n;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  let mn = Math.min(...values), mx = Math.max(...values);
  if (mx === mn) mx = mn + 1e-9;
  for (let i = 0; i < n; i++) {
    const v = values[i];
    const t = pipelineClamp01((v - mn) / (mx - mn));
    const [rr, gg, bb] = heatColor(t);
    ctx.fillStyle = i < revealedCount ? `rgb(${rr},${gg},${bb})` : '#2a2a3c';
    ctx.fillRect(i * cellW, 0, cellW, h);
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.strokeRect(i * cellW + 0.5, 0.5, cellW - 1, h - 1);
    if (i < revealedCount) {
      ctx.fillStyle = t > 0.55 ? '#111' : '#eef';
      ctx.font = `600 ${Math.max(9, cellW * 0.34)}px JetBrains Mono, monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const label = Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2);
      ctx.fillText(label, (i + 0.5) * cellW, h / 2 + 2);
    }
  }
}

function initFullPipelineDemo() {
  pipelineInputData = Array.from({ length: PIPELINE_INPUT_SIZE }, () =>
    Array.from({ length: PIPELINE_INPUT_SIZE }, () => Math.round(Math.random() * 9))
  );
  pipelineKernelData = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]];
  pipelineInputCells = [];
  renderPipelineGrids();
  resetPipelineAnimation();
}

function renderPipelineGrids() {
  const inpDiv = document.getElementById('pipelineInputGrid');
  if (!inpDiv) return;
  inpDiv.innerHTML = '';
  inpDiv.style.display = 'inline-grid';
  inpDiv.style.gap = '2px';
  const padOn = !!document.getElementById('pipelinePadding')?.checked;
  const gridDim = padOn ? PIPELINE_INPUT_SIZE + 2 : PIPELINE_INPUT_SIZE;
  inpDiv.style.gridTemplateColumns = `repeat(${gridDim}, 38px)`;
  pipelineInputCells = [];
  for (let r = 0; r < PIPELINE_INPUT_SIZE; r++) pipelineInputCells[r] = [];

  const appendZeroPadCell = () => {
    const z = document.createElement('span');
    z.className = 'pipeline-pad-zero-cell';
    z.textContent = '0';
    z.title = 'Implicit zero (same padding — not editable)';
    z.style.background = cellColor(0, 0, 9);
    z.style.color = '#eee';
    inpDiv.appendChild(z);
  };

  const appendEditable = (ir, ic) => {
    const inp = document.createElement('input');
    inp.type = 'number';
    inp.min = '0';
    inp.max = '9';
    inp.step = '1';
    inp.className = 'pipeline-input-cell';
    inp.dataset.r = String(ir);
    inp.dataset.c = String(ic);
    inp.value = pipelineInputData[ir][ic];
    inp.disabled = !!(pipelineAnimRunning && !pipelinePaused);
    inp.style.background = cellColor(pipelineInputData[ir][ic], 0, 9);
    inp.style.color = pipelineInputData[ir][ic] > 5 ? '#111' : '#eee';
    inp.addEventListener('input', () => {
      if (pipelineAnimRunning && !pipelinePaused) return;
      let x = parseInt(inp.value, 10);
      if (!Number.isFinite(x)) x = 0;
      x = Math.max(0, Math.min(9, x));
      pipelineInputData[ir][ic] = x;
      inp.value = x;
      inp.style.background = cellColor(pipelineInputData[ir][ic], 0, 9);
      inp.style.color = pipelineInputData[ir][ic] > 5 ? '#111' : '#eee';
      resetPipelineAnimation();
    });
    inpDiv.appendChild(inp);
    pipelineInputCells[ir][ic] = inp;
  };

  if (padOn) {
    const last = PIPELINE_INPUT_SIZE + 1;
    for (let r = 0; r <= last; r++) {
      for (let c = 0; c <= last; c++) {
        const isBorder = (r === 0 || r === last || c === 0 || c === last);
        if (isBorder) appendZeroPadCell();
        else appendEditable(r - 1, c - 1);
      }
    }
  } else {
    for (let r = 0; r < PIPELINE_INPUT_SIZE; r++) {
      for (let c = 0; c < PIPELINE_INPUT_SIZE; c++) appendEditable(r, c);
    }
  }
  syncPipelinePaddingClass();

  const kernDiv = document.getElementById('pipelineKernelGrid');
  kernDiv.innerHTML = '';
  kernDiv.style.gridTemplateColumns = `repeat(${PIPELINE_KERN_SIZE}, 44px)`;
  for (let r = 0; r < PIPELINE_KERN_SIZE; r++) {
    for (let c = 0; c < PIPELINE_KERN_SIZE; c++) {
      const inp = document.createElement('input');
      inp.type = 'number';
      inp.step = '0.1';
      inp.value = pipelineKernelData[r][c];
      inp.disabled = !!(pipelineAnimRunning && !pipelinePaused);
      inp.style.background = cellColor(pipelineKernelData[r][c], -3, 3);
      inp.style.color = Math.abs(pipelineKernelData[r][c]) < 1 ? '#eee' : '#111';
      inp.addEventListener('input', () => {
        if (pipelineAnimRunning && !pipelinePaused) return;
        pipelineKernelData[r][c] = parseFloat(inp.value) || 0;
        resetPipelineAnimation();
        renderPipelineGrids();
      });
      kernDiv.appendChild(inp);
    }
  }
}

function randomizePipelineInput() {
  if (pipelineAnimRunning && !pipelinePaused) return;
  pipelineInputData = Array.from({ length: PIPELINE_INPUT_SIZE }, () =>
    Array.from({ length: PIPELINE_INPUT_SIZE }, () => Math.round(Math.random() * 9))
  );
  renderPipelineGrids();
  resetPipelineAnimation();
}

function setPipelineKernelPreset(name) {
  const presets = {
    'edge-h':  [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
    'edge-v':  [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
    'blur':    [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]],
    'sharpen': [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
  };
  pipelineKernelData = presets[name].map(rr => [...rr]);
  renderPipelineGrids();
  resetPipelineAnimation();
}

function updatePipelineParams() {
  syncPipelinePaddingClass();
  if (!pipelineAnimRunning) resetPipelineAnimation();
}

async function pipelineDelay() {
  const raw = parseInt(document.getElementById('pipelineSpeed').value, 10);
  const ms = Math.max(40, Math.min(500, Number.isFinite(raw) ? raw : 180));
  const end = Date.now() + ms;
  while (Date.now() < end) {
    if (pipelineAnimAbort) return false;
    while (pipelinePaused && !pipelineAnimAbort) {
      await new Promise(r => {
        pipelinePauseWait = { resolve: r };
      });
      pipelinePauseWait = null;
    }
    if (pipelineAnimAbort) return false;
    await sleep(Math.min(30, Math.max(0, end - Date.now())));
  }
  return true;
}

function pausePipelinePlayback() {
  if (!pipelineAnimRunning || pipelinePaused) return;
  pipelinePaused = true;
  const btn = document.getElementById('btnPlayPipeline');
  if (btn) btn.textContent = '▶ Resume';
}

function resumePipelinePlayback() {
  if (!pipelineAnimRunning || !pipelinePaused) return;
  pipelinePaused = false;
  if (pipelinePauseWait) {
    pipelinePauseWait.resolve();
    pipelinePauseWait = null;
  }
  const btn = document.getElementById('btnPlayPipeline');
  if (btn) btn.textContent = '⏸ Pause';
}

function togglePipelinePlayPause() {
  if (pipelineAnimRunning && pipelinePaused) {
    resumePipelinePlayback();
    return;
  }
  if (pipelineAnimRunning && !pipelinePaused) {
    pausePipelinePlayback();
    return;
  }
  startPipelineAnimation();
}

function pipelineSetPlayButtonRunning() {
  const btn = document.getElementById('btnPlayPipeline');
  if (btn) btn.textContent = pipelinePaused ? '▶ Resume' : '⏸ Pause';
}

function pipelineSetPlayButtonIdle() {
  const btn = document.getElementById('btnPlayPipeline');
  if (btn) btn.textContent = '▶ Play';
}

function resetPipelineAnimation() {
  pipelineAnimAbort = true;
  pipelinePaused = false;
  if (pipelinePauseWait) {
    pipelinePauseWait.resolve();
    pipelinePauseWait = null;
  }
  pipelineAnimRunning = false;

  clearPipelineConvHighlight();

  const trace = document.getElementById('pipelineCalcTrace');
  if (trace) trace.innerHTML = '';

  ['pipelineFeatureCanvas', 'pipelinePoolCanvas', 'pipelineFlattenCanvas'].forEach(id => {
    const c = document.getElementById(id);
    if (!c) return;
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.fillStyle = '#1e1e2e';
    ctx.fillRect(0, 0, c.width, c.height);
  });

  const fs = document.getElementById('pipelineFeatureSize');
  if (fs) fs.textContent = '';
  const ps = document.getElementById('pipelinePoolSize');
  if (ps) ps.textContent = '';
  const ft = document.getElementById('pipelineFlattenText');
  if (ft) ft.textContent = '';
  const dw = document.getElementById('pipelineDenseWeights');
  if (dw) dw.innerHTML = '';
  const dout = document.getElementById('pipelineDenseOutput');
  if (dout) dout.innerHTML = '';

  pipelineSetPlayButtonIdle();
  renderPipelineGrids();
}

function featureMapToNested(featureMap2d, oh, ow) {
  const m = [];
  for (let r = 0; r < oh; r++) {
    const row = [];
    for (let c = 0; c < ow; c++)
      row.push(featureMap2d[r][c]);
    m.push(row);
  }
  return m;
}

async function startPipelineAnimation() {
  if (pipelineAnimRunning) return;
  pipelineAnimRunning = true;
  pipelineAnimAbort = false;
  pipelinePaused = false;
  pipelineSetPlayButtonRunning();
  renderPipelineGrids();

  const stride = parseInt(document.getElementById('pipelineStride').value, 10) === 2 ? 2 : 1;
  const padding = !!document.getElementById('pipelinePadding').checked;
  const poolType = document.getElementById('pipelinePoolType').value;
  const trace = document.getElementById('pipelineCalcTrace');

  const input = pipelineInputData;
  const kernel = pipelineKernelData;
  const padded = pipelineBuildPadded(input, padding);
  const ph = padded.length;
  const pw = padded[0].length;

  let oh = Math.floor((ph - PIPELINE_KERN_SIZE) / stride) + 1;
  let ow = Math.floor((pw - PIPELINE_KERN_SIZE) / stride) + 1;

  const edgeMsg =
    ph < PIPELINE_KERN_SIZE || pw < PIPELINE_KERN_SIZE || oh < 1 || ow < 1;
  if (edgeMsg) {
    pipelineAppendTrace(trace, '—', 'Convolution output has no valid positions for this stride/padding/kernel.', '—');
    pipelineAnimRunning = false;
    pipelineSetPlayButtonIdle();
    renderPipelineGrids();
    return;
  }

  const featureMap = Array.from({ length: oh }, () => Array(ow).fill(0));
  const fCanvas = document.getElementById('pipelineFeatureCanvas');
  document.getElementById('pipelineFeatureSize').textContent = `${oh}×${ow}`;
  pipelineDrawHeatNumberGrid(fCanvas, featureMap.map(r => r.map(() => 0)));

  pipelineAppendTrace(trace, 'conv', `${padding ? 'same (1-pixel zero pad)' : 'no pad'}, stride=${stride}`, 'start');

  for (let r = 0; r < oh; r++) {
    for (let c = 0; c < ow; c++) {
      if (!(await pipelineDelay()) || pipelineAnimAbort) break;

      const pr0 = r * stride;
      const pc0 = c * stride;
      setPipelineConvHighlight(pr0, pc0, stride, padding);

      let sum = 0;
      const terms = [];
      for (let kr = 0; kr < PIPELINE_KERN_SIZE; kr++) {
        for (let kc = 0; kc < PIPELINE_KERN_SIZE; kc++) {
          const iv = padded[pr0 + kr][pc0 + kc];
          const kv = kernel[kr][kc];
          sum += iv * kv;
          terms.push(`${iv}×${kv.toFixed(3)}`);
        }
      }
      const val = parseFloat(sum.toFixed(4));
      featureMap[r][c] = val;

      pipelineDrawHeatNumberGrid(fCanvas, featureMapToNested(featureMap, oh, ow));
      pipelineAppendTrace(trace, `[${r}][${c}]`, terms.join(' + '), `sum = ${val.toFixed(4)}`);

      if (pipelineAnimAbort) break;
    }
    if (pipelineAnimAbort) break;
  }

  clearPipelineConvHighlight();

  if (pipelineAnimAbort) {
    pipelineAnimRunning = false;
    pipelineSetPlayButtonIdle();
    renderPipelineGrids();
    return;
  }

  pipelineAppendTrace(trace, 'conv', `Done. Feature map shape ${oh}×${ow}.`, `step ${stride}`);

  const poolPossible = oh >= 2 && ow >= 2;
  let flatVec = [];
  let pooled2d = null;

  const pCanvas = document.getElementById('pipelinePoolCanvas');
  if (!poolPossible) {
    pipelineAppendTrace(trace, 'pool', `⚠ Map ${oh}×${ow} → cannot tile 2×2 non-overlapping windows. Flatten raw map.`, '—');
    for (let r = 0; r < oh; r++)
      for (let c = 0; c < ow; c++)
        flatVec.push(featureMap[r][c]);
    document.getElementById('pipelinePoolSize').textContent = '—';
    const emptyPool = [[0]];
    pipelineDrawHeatNumberGrid(pCanvas, emptyPool);
  } else {
    const poolH = Math.floor(oh / 2);
    const poolW = Math.floor(ow / 2);
    pooled2d = Array.from({ length: poolH }, () => Array(poolW).fill(0));
    pipelineDrawHeatNumberGrid(pCanvas, pooled2d);
    document.getElementById('pipelinePoolSize').textContent = `${poolH}×${poolW}`;

    pipelineAppendTrace(trace, 'pool', `${poolType} pool 2×2, stride 2`, `windows ${poolH}×${poolW}`);

    for (let pr = 0; pr < poolH; pr++) {
      for (let pc = 0; pc < poolW; pc++) {
        if (!(await pipelineDelay()) || pipelineAnimAbort) break;

        const vals = [];
        for (let rr = 0; rr < 2; rr++)
          for (let cc = 0; cc < 2; cc++)
            vals.push(featureMap[pr * 2 + rr][pc * 2 + cc]);

        const poolVal =
          poolType === 'max' ? Math.max(...vals) :
            poolType === 'min' ? Math.min(...vals) : vals[0];
        pooled2d[pr][pc] = poolVal;

        const fmShow = featureMapToNested(featureMap, oh, ow);
        const hiR = pr * 2;
        const hiC = pc * 2;
        pipelineDrawHeatNumberGrid(fCanvas, fmShow, { r0: hiR, c0: hiC, h: 2, w: 2 });
        pipelineDrawHeatNumberGrid(pCanvas, pooled2d);

        const label = poolType === 'max' ? 'max' : 'min';
        pipelineAppendTrace(trace, `[${pr}][${pc}]`,
          `${label}(${vals.map(v => v.toFixed(4)).join(', ')})`,
          `${poolVal.toFixed(4)}`);
        if (pipelineAnimAbort) break;
      }
      if (pipelineAnimAbort) break;
    }

    pipelineDrawHeatNumberGrid(fCanvas, featureMapToNested(featureMap, oh, ow));

    flatVec = [];
    for (let r = 0; r < poolH; r++)
      for (let c = 0; c < poolW; c++)
        flatVec.push(pooled2d[r][c]);

    if (pipelineAnimAbort) {
      pipelineAnimRunning = false;
      pipelineSetPlayButtonIdle();
      renderPipelineGrids();
      return;
    }

    pipelineAppendTrace(trace, 'pool', `Done ${poolType} pooling. Flatten (${flatVec.length}).`, `${poolH}×${poolW}`);
  }

  if (pipelineAnimAbort) {
    pipelineAnimRunning = false;
    pipelineSetPlayButtonIdle();
    renderPipelineGrids();
    return;
  }

  const ftEl = document.getElementById('pipelineFlattenText');
  const flatCv = document.getElementById('pipelineFlattenCanvas');
  for (let k = 0; k <= flatVec.length; k++) {
    if (!(await pipelineDelay()) || pipelineAnimAbort) break;
    pipelineDrawVectorStrip(flatCv, flatVec, k);
    if (k === flatVec.length)
      ftEl.textContent = `[ ${flatVec.map(v => v.toFixed(4)).join(', ')} ]`;
    else ftEl.textContent = flatVec.slice(0, k).map(v => v.toFixed(4)).join(', ');
  }

  if (pipelineAnimAbort) {
    pipelineAnimRunning = false;
    pipelineSetPlayButtonIdle();
    renderPipelineGrids();
    return;
  }

  pipelineAppendTrace(trace, 'flat', ftEl.textContent, `dim=${flatVec.length}`);

  const nNeurons = 2;
  if (flatVec.length === 0) {
    pipelineAppendTrace(trace, 'dense', 'No values to flatten — dense layer skipped.', '—');
    pipelineAnimRunning = false;
    pipelineSetPlayButtonIdle();
    renderPipelineGrids();
    return;
  }

  const { W, b } = pipelineDeterministicDenseParams(flatVec, nNeurons);

  const wDisplay = document.getElementById('pipelineDenseWeights');
  wDisplay.innerHTML = '';

  const wTitle = document.createElement('div');
  wTitle.className = 'small dim mono';
  wTitle.textContent = `W (${nNeurons}×${flatVec.length})`;
  wDisplay.appendChild(wTitle);

  W.forEach((row, wi) => {
    const nrow = row.map((wv, ji) =>
      `<span title="W[${wi}][${ji}]">${wv}</span>`
    ).join(' ');
    const line = document.createElement('div');
    line.style.lineHeight = '1.55';
    line.innerHTML =
      `<span style="color:var(--accent3)">${wi}</span> &nbsp;` + nrow;
    wDisplay.appendChild(line);
  });
  const bLine = document.createElement('div');
  bLine.className = 'mt-8';
  bLine.style.color = 'var(--text-dim)';
  bLine.textContent = `b = [ ${b.map(x => x.toFixed(4)).join(', ')} ]`;
  wDisplay.appendChild(bLine);

  const denseOut = [];
  for (let i = 0; i < nNeurons; i++) {
    if (!(await pipelineDelay()) || pipelineAnimAbort) break;

    const pre = W[i].reduce((s, w, j) => s + w * flatVec[j], b[i]);
    const post = Math.max(0, parseFloat(pre.toFixed(4)));
    denseOut.push(post);

    const dots = W[i].map((w, j) => `${w.toFixed(4)}×(${flatVec[j].toFixed(4)})`).join(' + ');
    pipelineAppendTrace(trace,
      `dense[${i}]`,
      `( ${dots} ) + ${b[i].toFixed(4)}`,
      `raw=${pre.toFixed(4)} → ReLU=${post.toFixed(4)}`
    );

    const dout = document.getElementById('pipelineDenseOutput');
    dout.innerHTML = '';
    dout.style.gridTemplateColumns = `repeat(${nNeurons}, 54px)`;
    denseOut.forEach(v => {
      const cell = document.createElement('div');
      cell.className = 'num-cell';
      const tHeat = denseOut.length ? v / (Math.max(...denseOut) + 1e-9) : 0;
      const [rr, gg, bb] = heatColor(pipelineClamp01(tHeat));
      cell.style.background = `rgba(${rr},${gg},${bb},0.42)`;
      cell.textContent = v.toFixed(4);
      dout.appendChild(cell);
    });

    if (pipelineAnimAbort) break;
  }

  if (!pipelineAnimAbort)
    pipelineAppendTrace(trace, 'dense', `y = ReLU(Wx + b)`, `[${denseOut.map(v => v.toFixed(4)).join(', ')}]`);

  pipelineAnimRunning = false;
  pipelineSetPlayButtonIdle();
  renderPipelineGrids();
}

/* ================================================================
   DATASET GENERATION
   ================================================================ */
const IMG_SIZE    = 28;
const NUM_CLASSES = 10;
const TRAIN_SIZE  = 600;
const TEST_SIZE   = 150;

function generateDigitImage(digit) {
  const offscreen = document.createElement('canvas');
  offscreen.width = offscreen.height = IMG_SIZE;
  const ctx = offscreen.getContext('2d');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, IMG_SIZE, IMG_SIZE);

  const size  = 18 + Math.floor(Math.random() * 5);
  const fonts = ['Arial', 'Verdana', 'Georgia', 'Courier New', 'Times New Roman'];
  const font  = fonts[Math.floor(Math.random() * fonts.length)];
  ctx.font     = `bold ${size}px ${font}`;
  ctx.fillStyle = '#fff';
  ctx.textAlign    = 'center';
  ctx.textBaseline = 'middle';

  const ox = (Math.random() - 0.5) * 4;
  const oy = (Math.random() - 0.5) * 4;
  const angle = (Math.random() - 0.5) * 0.3;
  ctx.save();
  ctx.translate(IMG_SIZE / 2 + ox, IMG_SIZE / 2 + oy);
  ctx.rotate(angle);
  ctx.fillText(String(digit), 0, 0);
  ctx.restore();

  const imgData = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
  const out = new Float32Array(IMG_SIZE * IMG_SIZE);
  for (let i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
    const g = imgData.data[i * 4];
    out[i] = Math.min(1, g / 255 + Math.random() * 0.05);
  }
  return out;
}

async function generateDataset() {
  updateLoader('Generating synthetic digit dataset…');
  await sleep(50);

  const trainImages = [], trainLabels = [];
  const testImages  = [], testLabels  = [];
  const perClassTrain = Math.ceil(TRAIN_SIZE / NUM_CLASSES);
  const perClassTest  = Math.ceil(TEST_SIZE  / NUM_CLASSES);

  for (let digit = 0; digit < NUM_CLASSES; digit++) {
    for (let i = 0; i < perClassTrain; i++) {
      trainImages.push(generateDigitImage(digit));
      trainLabels.push(digit);
    }
    for (let i = 0; i < perClassTest; i++) {
      testImages.push(generateDigitImage(digit));
      testLabels.push(digit);
    }
    if (digit % 3 === 0) await sleep(10);
  }

  shuffleArrays(trainImages, trainLabels);

  trainArrays = { images: trainImages, labels: trainLabels };
  testArrays  = { images: testImages,  labels: testLabels  };

  document.getElementById('datasetCount').textContent =
    `${trainImages.length} training  •  ${testImages.length} test samples`;
  document.getElementById('datasetStatus').innerHTML =
    '<div class="pulse"></div> Dataset ready';
  document.getElementById('datasetStatus').className = 'badge badge-done';

  showMNISTGrid(trainImages, trainLabels, 25);
}

function shuffleArrays(a, b) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
    [b[i], b[j]] = [b[j], b[i]];
  }
}

function showMNISTGrid(images, labels, count) {
  const grid = document.getElementById('mnistGrid');
  grid.innerHTML = '';
  const idxs = [];
  while (idxs.length < Math.min(count, images.length)) {
    const r = Math.floor(Math.random() * images.length);
    if (!idxs.includes(r)) idxs.push(r);
  }
  idxs.forEach(idx => {
    const cell = document.createElement('div');
    cell.className = 'mnist-cell';
    const c = document.createElement('canvas');
    c.width = c.height = 48;
    const img2d = flat2d(images[idx], IMG_SIZE, IMG_SIZE);
    drawHeatmap(c, img2d, {w: 48, h: 48});
    const lbl = document.createElement('span');
    lbl.textContent = labels[idx];
    cell.appendChild(c);
    cell.appendChild(lbl);
    grid.appendChild(cell);
  });
}

function showNextBatch() {
  if (!trainArrays) return;
  showMNISTGrid(trainArrays.images, trainArrays.labels, 25);
}

/* ================================================================
   MODEL ARCHITECTURE
   ================================================================ */
function buildModel() {
  const inp = tf.input({shape: [IMG_SIZE, IMG_SIZE, 1]});
  const conv1 = tf.layers.conv2d({filters:8, kernelSize:3, padding:'same', activation:'relu', name:'conv1'}).apply(inp);
  const pool1 = tf.layers.maxPooling2d({poolSize:2, strides:2, name:'pool1'}).apply(conv1);
  const conv2 = tf.layers.conv2d({filters:16, kernelSize:3, padding:'same', activation:'relu', name:'conv2'}).apply(pool1);
  const pool2 = tf.layers.maxPooling2d({poolSize:2, strides:2, name:'pool2'}).apply(conv2);
  const flat  = tf.layers.flatten({name:'flatten'}).apply(pool2);
  const dense1= tf.layers.dense({units:64, activation:'relu', name:'dense1'}).apply(flat);
  const out   = tf.layers.dense({units:10, activation:'softmax', name:'output'}).apply(dense1);

  model = tf.model({inputs: inp, outputs: out});
  vizModel = tf.model({
    inputs: inp,
    outputs: [conv1, pool1, conv2, pool2, flat, dense1, out],
  });
  return model;
}

function renderArchDiagram() {
  const diag = document.getElementById('archDiagram');
  const layers = [
    {name:'Input',   shape:'28×28×1',  cls:'', params:0},
    {name:'Conv2D',  shape:'28×28×8',  cls:'conv', params:8*(3*3*1+1)},
    {name:'MaxPool', shape:'14×14×8',  cls:'pool', params:0},
    {name:'Conv2D',  shape:'14×14×16', cls:'conv', params:16*(3*3*8+1)},
    {name:'MaxPool', shape:'7×7×16',   cls:'pool', params:0},
    {name:'Flatten', shape:'784',       cls:'flat', params:0},
    {name:'Dense',   shape:'64',        cls:'dense', params:784*64+64},
    {name:'Dense',   shape:'10',        cls:'out', params:64*10+10},
  ];
  diag.innerHTML = '';
  layers.forEach((l, i) => {
    const blk = document.createElement('div');
    blk.className = 'arch-block';
    blk.innerHTML = `
      <div class="arch-layer ${l.cls}">${l.name}<br/><span style="font-size:0.65rem;opacity:0.8">${l.shape}</span></div>
      <div class="arch-params">${l.params > 0 ? l.params.toLocaleString()+' params' : 'no params'}</div>`;
    diag.appendChild(blk);
    if (i < layers.length - 1) {
      const arrow = document.createElement('div');
      arrow.className = 'arch-arrow';
      arrow.textContent = '→';
      diag.appendChild(arrow);
    }
  });

  const total = layers.reduce((s, l) => s + l.params, 0);
  const tbl = document.getElementById('archParamTable');
  tbl.innerHTML = `<div class="card-title">Layer Parameter Count</div>
    <table style="width:100%; border-collapse:collapse;">
    <thead><tr style="color:var(--accent3)"><th style="text-align:left">Layer</th><th>Output Shape</th><th>Parameters</th></tr></thead>
    <tbody>${layers.map(l => `
      <tr style="border-top:1px solid var(--border)">
        <td style="padding:4px 0">${l.name}</td>
        <td style="text-align:center">${l.shape}</td>
        <td style="text-align:right; color:var(--accent)">${l.params.toLocaleString()}</td>
      </tr>`).join('')}
      <tr style="border-top:2px solid var(--accent); font-weight:700; color:var(--accent3)">
        <td>Total</td><td></td><td style="text-align:right">${total.toLocaleString()}</td>
      </tr>
    </tbody></table>`;
}

/* ================================================================
   CHARTS
   ================================================================ */
function initCharts() {
  const chartOpts = (label, color) => ({
    type: 'line',
    data: { labels: [], datasets: [{ label, data: [], borderColor: color, tension: 0.35,
      pointRadius: 3, fill: false, borderWidth: 2 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#e0e0e0', font:{family:'JetBrains Mono',size:11} } } },
      scales: {
        x: { ticks:{color:'#888aaa'}, grid:{color:'rgba(255,255,255,0.05)'} },
        y: { ticks:{color:'#888aaa'}, grid:{color:'rgba(255,255,255,0.05)'} }
      }
    }
  });

  const lossCtx = document.getElementById('lossChart').getContext('2d');
  const accCtx  = document.getElementById('accChart').getContext('2d');
  lossChartInst = new Chart(lossCtx, chartOpts('Loss', '#6c63ff'));
  accChartInst  = new Chart(accCtx,  chartOpts('Accuracy', '#49B6E5'));
}

function updateCharts(epoch, loss, acc) {
  lossChartInst.data.labels.push(`E${epoch}`);
  lossChartInst.data.datasets[0].data.push(loss);
  lossChartInst.update('none');
  accChartInst.data.labels.push(`E${epoch}`);
  accChartInst.data.datasets[0].data.push(acc);
  accChartInst.update('none');
}

function resetCharts() {
  lossChartInst.data.labels = [];
  lossChartInst.data.datasets[0].data = [];
  lossChartInst.update('none');
  accChartInst.data.labels = [];
  accChartInst.data.datasets[0].data = [];
  accChartInst.update('none');
}

/* ================================================================
   TRAINING LOOP
   ================================================================ */
const OPTIMIZER_LR = () => parseFloat(document.getElementById('lrSelect').value);

async function startTraining() {
  if (!trainArrays) { alert('Dataset not ready yet!'); return; }
  if (isTraining) return;

  isTraining = true;
  isPaused   = false;
  stopFlag   = false;

  if (!model) { buildModel(); renderArchDiagram(); }

  const optimizer = tf.train.adam(OPTIMIZER_LR());
  const epochs    = parseInt(document.getElementById('epochsSlider').value);
  const batchSize = parseInt(document.getElementById('batchSelect').value);
  const doInspect = document.getElementById('inspectCheck').checked;

  setTrainStatus('running');
  document.getElementById('btnTrain').disabled = true;
  document.getElementById('btnPause').disabled = false;

  const n = trainArrays.images.length;
  const allXs = tf.tensor4d(
    Float32Array.from(trainArrays.images.flatMap(a => Array.from(a))),
    [n, IMG_SIZE, IMG_SIZE, 1]
  );
  const allYs = tf.oneHot(tf.tensor1d(trainArrays.labels, 'int32'), NUM_CLASSES).toFloat();
  const numBatches = Math.ceil(n / batchSize);

  for (let epoch = 0; epoch < epochs && !stopFlag; epoch++) {
    let epochLoss = 0, epochAcc = 0;
    document.getElementById('epochLabel').textContent = `Epoch ${epoch+1} / ${epochs}`;
    document.getElementById('epochProgress').style.width = `${((epoch) / epochs) * 100}%`;

    const indices = tf.util.createShuffledIndices(n);

    for (let b = 0; b < numBatches && !stopFlag; b++) {
      if (isPaused) {
        await waitForContinue();
        if (stopFlag) break;
      }

      document.getElementById('batchLabel').textContent  = `Batch ${b+1} / ${numBatches}`;
      document.getElementById('batchProgress').style.width = `${((b+1) / numBatches) * 100}%`;

      const start = b * batchSize;
      const end   = Math.min(start + batchSize, n);
      const bIdx  = Array.from(indices).slice(start, end);

      const batchResult = tf.tidy(() => {
        const bIdxT = tf.tensor1d(bIdx, 'int32');
        const bXs = tf.gather(allXs, bIdxT);
        const bYs = tf.gather(allYs, bIdxT);
        return [bXs, bYs];
      });
      const [bXs, bYs] = batchResult;

      let gradData = null;
      const {value: loss, grads} = optimizer.computeGradients(() => {
        return tf.tidy(() => {
          const preds = model.apply(bXs, {training: true});
          return tf.losses.softmaxCrossEntropy(bYs, preds).mean();
        });
      });

      optimizer.applyGradients(grads);

      const lossVal = await loss.data();
      epochLoss = lossVal[0];

      const accVal = await tf.tidy(() => {
        const preds  = model.apply(bXs);
        const predCls = preds.argMax(-1);
        const trueCls = bYs.argMax(-1);
        return predCls.equal(trueCls).mean().data();
      });
      epochAcc = accVal[0];

      tf.dispose([bXs, bYs, loss]);
      Object.values(grads).forEach(g => tf.dispose(g));

      if (doInspect && b === 0) {
        const sampleIdx = bIdx[0];
        await showInspectStep(sampleIdx, epochLoss, epochAcc);
        if (!stopFlag) {
          await waitForContinue();
        }
      }

      await sleep(0);
    }

    updateCharts(epoch + 1, epochLoss, epochAcc);
    document.getElementById('epochProgress').style.width = `${((epoch+1) / epochs) * 100}%`;
  }

  tf.dispose([allXs, allYs]);
  isTraining = false;
  setTrainStatus(stopFlag ? 'idle' : 'done');
  document.getElementById('btnTrain').disabled = false;
  document.getElementById('btnPause').disabled = true;
  document.getElementById('inspectContainer').style.display = 'none';
}

function pauseTraining() {
  if (!isTraining) return;
  if (isPaused) {
    isPaused = false;
    setTrainStatus('running');
    document.getElementById('btnPause').textContent = '⏸ Pause';
    if (continueResolver) { continueResolver(); continueResolver = null; }
  } else {
    isPaused = true;
    setTrainStatus('paused');
    document.getElementById('btnPause').textContent = '▶ Resume';
  }
}

function continueTraining() {
  document.getElementById('inspectContainer').style.display = 'none';
  if (continueResolver) { continueResolver(); continueResolver = null; }
}

function waitForContinue() {
  return new Promise(resolve => { continueResolver = resolve; });
}

function resetModel() {
  stopFlag = true;
  if (continueResolver) { continueResolver(); continueResolver = null; }
  isTraining = false; isPaused = false;
  if (model) { model.dispose(); model = null; }
  if (vizModel) { vizModel.dispose(); vizModel = null; }
  buildModel(); renderArchDiagram();
  resetCharts();
  setTrainStatus('idle');
  document.getElementById('btnTrain').disabled = false;
  document.getElementById('btnPause').disabled = true;
  document.getElementById('btnPause').textContent = '⏸ Pause';
  document.getElementById('inspectContainer').style.display = 'none';
  document.getElementById('epochLabel').textContent = 'Epoch — / —';
  document.getElementById('batchLabel').textContent = 'Batch — / —';
  document.getElementById('epochProgress').style.width = '0%';
  document.getElementById('batchProgress').style.width = '0%';
}

function setTrainStatus(state) {
  const el = document.getElementById('trainStatus');
  el.className = `badge badge-${state}`;
  const labels = {idle:'Idle', running:'Training…', paused:'Paused', done:'Complete ✓'};
  el.innerHTML = `<div class="pulse"></div> ${labels[state]||state}`;
}

/* ================================================================
   INSPECT STEP
   ================================================================ */
async function showInspectStep(sampleIdx, loss, acc) {
  const panel = document.getElementById('inspectPanel');
  panel.innerHTML = '<div class="small mono dim">Extracting intermediate outputs…</div>';
  document.getElementById('inspectContainer').style.display = 'block';

  const imgFlat = trainArrays.images[sampleIdx];
  const trueLabel = trainArrays.labels[sampleIdx];
  await renderInspectPanel(panel, imgFlat, trueLabel, loss);
}

async function renderInspectPanel(panel, imgFlat, trueLabel, lossVal) {
  panel.innerHTML = '';

  if (!model || !vizModel) {
    panel.innerHTML = '<div class="dim small">Train the model first to see visualisations.</div>';
    return;
  }

  // 1. Input image
  const inputSec = makeInspectSection('📷 Input Image (28×28 grayscale)');
  const img2d = flat2d(imgFlat, IMG_SIZE, IMG_SIZE);
  inputSec.appendChild(makeHeatmapEl(img2d, 112, '28×28'));
  panel.appendChild(inputSec);

  // Run vizModel
  const [c1Out, p1Out, c2Out, p2Out, flatOut, d1Out, softOut] = await tf.tidy(() => {
    const xT = tf.tensor4d(imgFlat, [1, IMG_SIZE, IMG_SIZE, 1]);
    const outs = vizModel.apply(xT);
    return outs.map(t => t.dataSync());
  });

  // 2. Conv1
  const c1Sec = makeInspectSection('🔲 Layer 1 — Conv2D (8 filters, 3×3, ReLU) + MaxPool → 14×14');
  const conv1Weights = model.getLayer('conv1').getWeights()[0];
  const conv1W = await conv1Weights.data();
  const kWrap = document.createElement('div');
  kWrap.innerHTML = '<div class="small mono dim" style="margin-bottom:6px;">Kernel Weights (8 × 3×3)</div>';
  const kGrid = document.createElement('div');
  kGrid.className = 'maps-grid';
  for (let f = 0; f < 8; f++) {
    const kern2d = Array.from({length:3}, (_, r) =>
      Array.from({length:3}, (_, c) => conv1W[r * 3 * 1 * 8 + c * 1 * 8 + 0 * 8 + f])
    );
    kGrid.appendChild(makeHeatmapEl(kern2d, 60, `K${f}`));
  }
  kWrap.appendChild(kGrid);
  c1Sec.appendChild(kWrap);
  c1Sec.appendChild(document.createElement('hr')).className = 'doodle';

  const fmWrap = document.createElement('div');
  fmWrap.innerHTML = '<div class="small mono dim" style="margin-bottom:6px;">Feature Maps after ReLU (28×28 each)</div>';
  const fmGrid = document.createElement('div');
  fmGrid.className = 'maps-grid';
  for (let f = 0; f < 8; f++) {
    const fm = Array.from({length: IMG_SIZE}, (_, r) =>
      Array.from({length: IMG_SIZE}, (_, c) => c1Out[r * IMG_SIZE * 8 + c * 8 + f])
    );
    fmGrid.appendChild(makeHeatmapEl(fm, 70, `FM${f}`));
  }
  fmWrap.appendChild(fmGrid);
  c1Sec.appendChild(fmWrap);

  const pmWrap = document.createElement('div');
  pmWrap.innerHTML = '<div class="small mono dim" style="margin-bottom:6px; margin-top:10px;">After MaxPool2D (14×14 each)</div>';
  const pmGrid = document.createElement('div');
  pmGrid.className = 'maps-grid';
  for (let f = 0; f < 8; f++) {
    const pm = Array.from({length: 14}, (_, r) =>
      Array.from({length: 14}, (_, c) => p1Out[r * 14 * 8 + c * 8 + f])
    );
    pmGrid.appendChild(makeHeatmapEl(pm, 60, `PM${f}`));
  }
  pmWrap.appendChild(pmGrid);
  c1Sec.appendChild(pmWrap);
  panel.appendChild(c1Sec);

  // 3. Conv2
  const c2Sec = makeInspectSection('🔲 Layer 2 — Conv2D (16 filters, 3×3, ReLU) + MaxPool → 7×7');
  const c2FmWrap = document.createElement('div');
  c2FmWrap.innerHTML = '<div class="small mono dim" style="margin-bottom:6px;">Feature Maps after ReLU (14×14 each)</div>';
  const c2FmGrid = document.createElement('div');
  c2FmGrid.className = 'maps-grid';
  for (let f = 0; f < 16; f++) {
    const fm = Array.from({length: 14}, (_, r) =>
      Array.from({length: 14}, (_, c) => c2Out[r * 14 * 16 + c * 16 + f])
    );
    c2FmGrid.appendChild(makeHeatmapEl(fm, 55, `FM${f}`));
  }
  c2FmWrap.appendChild(c2FmGrid);
  c2Sec.appendChild(c2FmWrap);

  const p2Wrap = document.createElement('div');
  p2Wrap.innerHTML = '<div class="small mono dim" style="margin-bottom:6px; margin-top:10px;">After MaxPool2D (7×7 each)</div>';
  const p2Grid = document.createElement('div');
  p2Grid.className = 'maps-grid';
  for (let f = 0; f < 16; f++) {
    const pm = Array.from({length: 7}, (_, r) =>
      Array.from({length: 7}, (_, c) => p2Out[r * 7 * 16 + c * 16 + f])
    );
    p2Grid.appendChild(makeHeatmapEl(pm, 50, `PM${f}`));
  }
  p2Wrap.appendChild(p2Grid);
  c2Sec.appendChild(p2Wrap);
  panel.appendChild(c2Sec);

  // 4. Flatten
  const flatSec = makeInspectSection('⬡ Flatten — 7×7×16 = 784-dimensional vector');
  const flatC = document.createElement('canvas');
  drawVectorStrip(flatC, Array.from(flatOut), 28);
  flatC.style.borderRadius = '4px';
  flatSec.appendChild(flatC);
  flatSec.appendChild(makeSmallNote('Each pixel in this strip = one value in the 784-element vector'));
  panel.appendChild(flatSec);

  // 5. Dense1
  const d1Sec = makeInspectSection('⬡ Dense Layer — 64 neurons, ReLU');
  const d1C = document.createElement('canvas');
  drawVectorStrip(d1C, Array.from(d1Out), 32);
  d1C.style.borderRadius = '4px';
  d1Sec.appendChild(d1C);
  d1Sec.appendChild(makeSmallNote('64 neuron activations after ReLU'));
  panel.appendChild(d1Sec);

  // 6. Output
  const outSec = makeInspectSection('🎯 Output — 10 Class Probabilities (Softmax)');
  const predicted = Array.from(softOut).indexOf(Math.max(...softOut));
  const tvp = document.createElement('div');
  tvp.className = 'row gap-8 mb-8';
  tvp.style.marginBottom = '10px';
  tvp.innerHTML = `
    <div class="card" style="padding:10px 16px;">
      <div class="small dim mono">True Label</div>
      <div style="font-size:2rem; color:var(--accent3); font-family:var(--font-mono)">${trueLabel >= 0 ? trueLabel : '?'}</div>
    </div>
    <div class="card" style="padding:10px 16px;">
      <div class="small dim mono">Predicted</div>
      <div style="font-size:2rem; color:${predicted === trueLabel || trueLabel < 0 ? 'var(--success)' : 'var(--danger)'}; font-family:var(--font-mono)">${predicted}</div>
    </div>`;
  outSec.appendChild(tvp);

  const probBars = document.createElement('div');
  probBars.className = 'prob-bars';
  renderProbBars(probBars, softOut, predicted);
  outSec.appendChild(probBars);

  if (trueLabel >= 0 && lossVal != null) {
    const p = softOut[trueLabel];
    const lossEl = document.createElement('div');
    lossEl.style.marginTop = '12px';
    lossEl.innerHTML = `
      <div class="small mono dim">Cross-Entropy Loss Formula</div>
      <div style="font-family:var(--font-mono); font-size:0.8rem; color:var(--accent3); margin-top:6px;">
        L = -log(p<sub>true</sub>) = -log(${p.toFixed(4)}) = <strong style="color:var(--accent)">${(-Math.log(p + 1e-9)).toFixed(4)}</strong>
      </div>
      <div class="small dim" style="margin-top:4px;">Batch loss: ${typeof lossVal === 'number' ? lossVal.toFixed(4) : '—'}</div>`;
    outSec.appendChild(lossEl);
  }
  panel.appendChild(outSec);

  // 7. Gradients
  if (trueLabel >= 0) {
    const gradSec = makeInspectSection('∇ Gradients — Output Layer Weights');
    try {
      const gradData = tf.tidy(() => {
        const xT = tf.tensor4d(imgFlat, [1, IMG_SIZE, IMG_SIZE, 1]);
        const yT = tf.oneHot(tf.tensor1d([trueLabel], 'int32'), 10).toFloat();
        const grads = tf.grad(x => {
          const preds = model.apply(x);
          return tf.losses.softmaxCrossEntropy(yT, preds).mean();
        })(xT);
        return grads.dataSync();
      });
      const gradImg2d = flat2d(gradData, IMG_SIZE, IMG_SIZE);
      const gWrap = document.createElement('div');
      gWrap.innerHTML = '<div class="small mono dim" style="margin-bottom:6px;">Input Gradient (sensitivity map — what the network attends to)</div>';
      gWrap.appendChild(makeHeatmapEl(gradImg2d, 112, 'grad w.r.t input'));
      gradSec.appendChild(gWrap);
    } catch(e) {
      gradSec.appendChild(makeSmallNote('Gradient computation unavailable for this view.'));
    }
    panel.appendChild(gradSec);
  }
}

function makeInspectSection(title) {
  const sec = document.createElement('div');
  sec.className = 'inspect-section';
  const t = document.createElement('div');
  t.className = 'inspect-title';
  t.textContent = title;
  sec.appendChild(t);
  return sec;
}

function makeSmallNote(text) {
  const el = document.createElement('div');
  el.className = 'small dim mono';
  el.style.marginTop = '6px';
  el.textContent = text;
  return el;
}

/* ================================================================
   DRAWING CANVAS
   ================================================================ */
function initDrawCanvas() {
  const canvas = document.getElementById('drawCanvas');
  drawCtx = canvas.getContext('2d');
  clearCanvas();

  canvas.addEventListener('mousedown',  e => { drawing = true;  draw(e); });
  canvas.addEventListener('mousemove',  e => { if (drawing) draw(e); });
  canvas.addEventListener('mouseup',    () => drawing = false);
  canvas.addEventListener('mouseleave', () => drawing = false);

  canvas.addEventListener('touchstart',  e => { e.preventDefault(); drawing = true; draw(e.touches[0]); }, {passive:false});
  canvas.addEventListener('touchmove',   e => { e.preventDefault(); if (drawing) draw(e.touches[0]); }, {passive:false});
  canvas.addEventListener('touchend',    () => drawing = false);
}

function draw(e) {
  const canvas = document.getElementById('drawCanvas');
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  drawCtx.fillStyle = '#fff';
  drawCtx.beginPath();
  drawCtx.arc(x, y, 14, 0, Math.PI * 2);
  drawCtx.fill();
}

function clearCanvas() {
  const canvas = document.getElementById('drawCanvas');
  drawCtx = drawCtx || canvas.getContext('2d');
  drawCtx.fillStyle = '#000';
  drawCtx.fillRect(0, 0, canvas.width, canvas.height);
  document.getElementById('predResultCard').style.display = 'none';
  document.getElementById('predInspectContainer').style.display = 'none';
}

function getDrawingPixels() {
  const bigCanvas = document.getElementById('drawCanvas');
  const small = document.createElement('canvas');
  small.width = small.height = IMG_SIZE;
  const ctx = small.getContext('2d');
  ctx.drawImage(bigCanvas, 0, 0, IMG_SIZE, IMG_SIZE);
  const imgData = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
  const out = new Float32Array(IMG_SIZE * IMG_SIZE);
  for (let i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
    out[i] = imgData.data[i * 4] / 255;
  }
  return out;
}

async function predictDrawing() {
  if (!model) { alert('Please train the model first!'); return; }
  const pixels = getDrawingPixels();
  await runPrediction(pixels, -1);
}

async function loadRandomTest() {
  if (!testArrays) { alert('Dataset not loaded yet!'); return; }
  const idx = Math.floor(Math.random() * testArrays.images.length);
  const img = testArrays.images[idx];
  const lbl = testArrays.labels[idx];

  const bigCanvas = document.getElementById('drawCanvas');
  const ctx = drawCtx || bigCanvas.getContext('2d');
  const imgData = ctx.createImageData(bigCanvas.width, bigCanvas.height);
  const scale = bigCanvas.width / IMG_SIZE;
  for (let r = 0; r < bigCanvas.height; r++) {
    for (let c = 0; c < bigCanvas.width; c++) {
      const sr = Math.floor(r / scale);
      const sc = Math.floor(c / scale);
      const v = Math.round(img[sr * IMG_SIZE + sc] * 255);
      const idx4 = (r * bigCanvas.width + c) * 4;
      imgData.data[idx4]   = v;
      imgData.data[idx4+1] = v;
      imgData.data[idx4+2] = v;
      imgData.data[idx4+3] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);
  if (model) await runPrediction(img, lbl);
}

async function runPrediction(imgFlat, trueLabel) {
  if (!model) return;
  const probs = tf.tidy(() => {
    const xT = tf.tensor4d(imgFlat, [1, IMG_SIZE, IMG_SIZE, 1]);
    return model.apply(xT).dataSync();
  });

  const predicted = Array.from(probs).indexOf(Math.max(...probs));
  const resultCard = document.getElementById('predResultCard');
  resultCard.style.display = 'block';
  document.getElementById('predDigit').textContent = predicted;
  document.getElementById('predDigit').style.color = (trueLabel < 0 || predicted === trueLabel)
    ? 'var(--accent3)' : 'var(--danger)';
  renderProbBars(document.getElementById('predProbs'), probs, predicted);

  const predPanel = document.getElementById('predInspectPanel');
  document.getElementById('predInspectContainer').style.display = 'block';
  await renderInspectPanel(predPanel, imgFlat, trueLabel, null);
}

/* ================================================================
   SECTION 6 — COMPUTATION WALKTHROUGH
   ================================================================ */

/**
 * Build a small self-contained example with a 4×4 input patch,
 * 3×3 kernel, two 2×2 pooling regions, a tiny flatten→dense→softmax chain.
 * All numbers are computed in plain JS so users can verify them step by step.
 */
function generateCompWalkthrough() {
  document.getElementById('compWalkthroughStatus').textContent = 'Generated!';
  const container = document.getElementById('compPipeline');
  container.innerHTML = '';

  /* ── Random seed data ── */
  const input4 = Array.from({length:4}, () =>
    Array.from({length:4}, () => Math.round(Math.random() * 9))
  );

  const kernel3 = Array.from({length:3}, () =>
    Array.from({length:3}, () => parseFloat((Math.random()*2-1).toFixed(2)))
  );

  const bias = parseFloat((Math.random()*0.4-0.2).toFixed(2));

  /* ── Step definitions ── */
  const steps = [
    buildStep_Input(input4),
    buildStep_Convolution(input4, kernel3, bias),
    buildStep_ReLU(input4, kernel3, bias),
    buildStep_Pooling(input4, kernel3, bias),
    buildStep_Flatten(input4, kernel3, bias),
    buildStep_Dense(input4, kernel3, bias),
    buildStep_Softmax(input4, kernel3, bias),
    buildStep_Loss(input4, kernel3, bias),
  ];

  const colors = ['#6c63ff','#49B6E5','#f7c948','#16A34A','#ec4899','#f97316','#7c3aed','#DC2626'];

  steps.forEach((step, i) => {
    // Build accordion item
    const item = document.createElement('div');
    item.className = 'comp-step';

    const hdr = document.createElement('div');
    hdr.className = 'comp-step-header';
    hdr.innerHTML = `
      <div class="comp-step-badge" style="background:${colors[i]}">${i+1}</div>
      <div class="comp-step-title">${step.title}</div>
      <div class="comp-step-shape">${step.shape}</div>
      <div class="comp-step-chevron">▾</div>`;
    hdr.addEventListener('click', () => {
      const bdy = item.querySelector('.comp-step-body');
      const open = bdy.classList.contains('open');
      bdy.classList.toggle('open', !open);
      hdr.classList.toggle('open', !open);
    });

    const bdy = document.createElement('div');
    bdy.className = 'comp-step-body';
    bdy.appendChild(step.content);

    item.appendChild(hdr);
    item.appendChild(bdy);
    container.appendChild(item);

    // Connector (not after last)
    if (i < steps.length - 1) {
      const conn = document.createElement('div');
      conn.className = 'comp-connector';
      conn.innerHTML = '<div class="comp-connector-arrow">▼</div>';
      container.appendChild(conn);
    }
  });

  // Auto-open first step
  const firstHdr  = container.querySelector('.comp-step-header');
  const firstBody = container.querySelector('.comp-step-body');
  if (firstHdr) { firstHdr.classList.add('open'); firstBody.classList.add('open'); }
}

/* ── Helper: render a numeric grid as HTML ── */
function renderNumGrid(data2d, highlightFn = null) {
  const rows = data2d.length, cols = data2d[0].length;
  const grid = document.createElement('div');
  grid.className = 'num-grid';
  grid.style.gridTemplateColumns = `repeat(${cols}, 36px)`;
  data2d.forEach((row, r) => row.forEach((val, c) => {
    const cell = document.createElement('div');
    cell.className = 'num-cell' + (highlightFn && highlightFn(r, c) ? ' highlight' : '');
    const [rr, gg, bb] = heatColor(Math.max(0, Math.min(1, (val + 10) / 20)));
    cell.style.background = `rgba(${rr},${gg},${bb},0.35)`;
    cell.textContent = typeof val === 'number' ? (Number.isInteger(val) ? val : val.toFixed(2)) : val;
    grid.appendChild(cell);
  }));
  return grid;
}

/* ── Helper: formula box ── */
function formulaBox(html) {
  const d = document.createElement('div');
  d.className = 'formula-box';
  d.innerHTML = html;
  return d;
}

/* ── Helper: trace table ── */
function traceTable(rows) {
  const d = document.createElement('div');
  d.className = 'calc-trace';
  rows.forEach(([idx, expr, val]) => {
    const row = document.createElement('div');
    row.className = 'calc-trace-row';
    row.innerHTML = `<span class="calc-trace-idx">${idx}</span><span class="calc-trace-expr">${expr}</span><span class="calc-trace-val">${val}</span>`;
    d.appendChild(row);
  });
  return d;
}

/* ── Helper: stat chips ── */
function statChips(items) {
  const d = document.createElement('div');
  d.className = 'stat-chips';
  items.forEach(([label, val]) => {
    d.innerHTML += `<div class="stat-chip"><span class="stat-chip-label">${label}</span><span class="stat-chip-val">${val}</span></div>`;
  });
  return d;
}

/* ── Step 1: Input ── */
function buildStep_Input(input4) {
  const content = document.createElement('div');
  content.innerHTML = `
    <div class="formula-box">
      <b>Input Grid</b> — 4×4 pixel values (0–9 scale, simulating grayscale 0–255 ÷ 28)
      <span class="formula-comment">In the real CNN, input is 28×28 normalised to [0, 1]. Here we show a 4×4 patch for clarity.</span>
    </div>`;

  content.appendChild(renderNumGrid(input4));

  const mn = Math.min(...input4.flat()), mx = Math.max(...input4.flat());
  const mean = (input4.flat().reduce((a, b) => a + b, 0) / 16).toFixed(2);
  content.appendChild(statChips([
    ['Grid size', '4×4 = 16 values'],
    ['Min pixel', mn],
    ['Max pixel', mx],
    ['Mean', mean],
  ]));

  content.innerHTML += `<div class="small dim mono" style="margin-top:10px;">
    Each cell is multiplied element-wise with the kernel filter during convolution.
    High values (bright) appear as yellows; low (dark) as blues/purples.
  </div>`;

  return { title: '📷 Input Grid', shape: '4×4', content };
}

/* ── Step 2: Convolution ── */
function buildStep_Convolution(input4, kernel3, bias) {
  const content = document.createElement('div');

  // Compute valid convolution (no padding, stride 1) → 2×2 output
  const outConv = [];
  const traceRows = [];
  for (let r = 0; r < 2; r++) {
    const row = [];
    for (let c = 0; c < 2; c++) {
      let sum = 0;
      const terms = [];
      for (let kr = 0; kr < 3; kr++) {
        for (let kc = 0; kc < 3; kc++) {
          const iv = input4[r+kr][c+kc];
          const kv = kernel3[kr][kc];
          sum += iv * kv;
          terms.push(`${iv}×${kv.toFixed(2)}`);
        }
      }
      sum += bias;
      row.push(parseFloat(sum.toFixed(3)));
      traceRows.push([
        `out[${r}][${c}]`,
        terms.join(' + ') + ` + bias(${bias})`,
        `= ${sum.toFixed(3)}`
      ]);
    }
    outConv.push(row);
  }

  content.appendChild(formulaBox(`
    <b>Convolution formula:</b>  out[r][c] = Σ<sub>kr,kc</sub> ( input[r+kr][c+kc] × kernel[kr][kc] ) + bias
    <span class="formula-comment">Kernel slides over the 4×4 input with stride=1, no padding → output is 2×2.
Output size = (Input − Kernel + 1) = (4 − 3 + 1) = 2 per dimension.</span>
  `));

  content.innerHTML += '<div style="display:flex;gap:20px;flex-wrap:wrap;align-items:flex-start;margin-bottom:10px;">';
  const row = document.createElement('div');
  row.style.cssText = 'display:flex;gap:20px;flex-wrap:wrap;align-items:flex-start;margin:8px 0;';

  const inputWrap = document.createElement('div');
  inputWrap.innerHTML = '<div class="small dim mono" style="margin-bottom:4px;">Input (4×4)</div>';
  inputWrap.appendChild(renderNumGrid(input4, (r,c) => r < 3 && c < 3));

  const kernWrap = document.createElement('div');
  kernWrap.innerHTML = '<div class="small dim mono" style="margin-bottom:4px;">Kernel (3×3) + bias=' + bias + '</div>';
  kernWrap.appendChild(renderNumGrid(kernel3));

  const outWrap = document.createElement('div');
  outWrap.innerHTML = '<div class="small dim mono" style="margin-bottom:4px;">Raw Output (2×2)</div>';
  outWrap.appendChild(renderNumGrid(outConv));

  row.appendChild(inputWrap);
  row.innerHTML += '<div style="align-self:center;font-size:1.4rem;color:var(--accent);">⊛</div>';
  row.appendChild(kernWrap);
  row.innerHTML += '<div style="align-self:center;font-size:1.4rem;color:var(--accent);">→</div>';
  row.appendChild(outWrap);
  content.appendChild(row);

  content.innerHTML += '<div class="small mono accent3" style="margin-bottom:6px;">Calculation trace (position [r][c]):</div>';
  content.appendChild(traceTable(traceRows));
  content.appendChild(statChips([
    ['Input shape', '4×4'],
    ['Kernel shape', '3×3'],
    ['Output shape', '2×2'],
    ['Bias', bias],
  ]));

  return { title: '⊛ Convolution', shape: '2×2 raw', content };
}

/* ── Step 3: ReLU ── */
function buildStep_ReLU(input4, kernel3, bias) {
  const content = document.createElement('div');

  // Recompute conv output
  const outConv = [];
  for (let r = 0; r < 2; r++) {
    const row = [];
    for (let c = 0; c < 2; c++) {
      let sum = bias;
      for (let kr = 0; kr < 3; kr++)
        for (let kc = 0; kc < 3; kc++)
          sum += input4[r+kr][c+kc] * kernel3[kr][kc];
      row.push(parseFloat(sum.toFixed(3)));
    }
    outConv.push(row);
  }

  const outReLU = outConv.map(row => row.map(v => Math.max(0, v)));

  const traceRows = [];
  outConv.forEach((row, r) => row.forEach((val, c) => {
    const relu = Math.max(0, val);
    traceRows.push([
      `[${r}][${c}]`,
      `ReLU(${val.toFixed(3)}) = max(0, ${val.toFixed(3)})`,
      `→ ${relu.toFixed(3)}`
    ]);
  }));

  content.appendChild(formulaBox(`
    <b>ReLU(x) = max(0, x)</b>
    <span class="formula-comment">All negative values are clipped to 0. Positive values pass through unchanged.
This introduces non-linearity — without it, stacking layers would collapse to a single linear transform.</span>
  `));

  const row = document.createElement('div');
  row.style.cssText = 'display:flex;gap:20px;flex-wrap:wrap;align-items:flex-start;margin:8px 0;';

  const preWrap = document.createElement('div');
  preWrap.innerHTML = '<div class="small dim mono" style="margin-bottom:4px;">Pre-ReLU (raw conv output)</div>';
  preWrap.appendChild(renderNumGrid(outConv));

  const postWrap = document.createElement('div');
  postWrap.innerHTML = '<div class="small dim mono" style="margin-bottom:4px;">Post-ReLU (negatives zeroed)</div>';
  postWrap.appendChild(renderNumGrid(outReLU));

  row.appendChild(preWrap);
  row.innerHTML += '<div style="align-self:center;font-size:1.4rem;color:var(--accent);">→</div>';
  row.appendChild(postWrap);
  content.appendChild(row);

  content.appendChild(traceTable(traceRows));

  const zeroed = outConv.flat().filter(v => v < 0).length;
  content.appendChild(statChips([
    ['Values zeroed', zeroed + ' / 4'],
    ['Values passed', (4 - zeroed) + ' / 4'],
  ]));

  return { title: '⚡ ReLU Activation', shape: '2×2 activated', content };
}

/* ── Step 4: Pooling ── */
function buildStep_Pooling(input4, kernel3, bias) {
  const content = document.createElement('div');

  // Recompute ReLU output — 2×2, so pool window is the whole thing
  const outConv = [];
  for (let r = 0; r < 2; r++) {
    const row = [];
    for (let c = 0; c < 2; c++) {
      let sum = bias;
      for (let kr = 0; kr < 3; kr++)
        for (let kc = 0; kc < 3; kc++)
          sum += input4[r+kr][c+kc] * kernel3[kr][kc];
      row.push(Math.max(0, parseFloat(sum.toFixed(3))));
    }
    outConv.push(row);
  }

  // Pool the 2×2 ReLU output with a 2×2 window → single scalar
  const flat4 = outConv.flat();
  const maxVal = Math.max(...flat4);
  const avgVal = flat4.reduce((a, b) => a + b, 0) / flat4.length;
  const winnerIdx = flat4.indexOf(maxVal);
  const winnerR = Math.floor(winnerIdx / 2), winnerC = winnerIdx % 2;

  content.appendChild(formulaBox(`
    <b>Max Pooling</b>  — pool_size=2×2, stride=2
    <span class="formula-comment">Takes the maximum value within each non-overlapping 2×2 window.
Reduces spatial dimensions: 2×2 → 1×1 here (in real CNN: 28×28 → 14×14 → 7×7).
Benefits: translation invariance, reduces parameters, controls overfitting.</span>
  `));

  // Show pool window visual
  const poolWrap = document.createElement('div');
  poolWrap.className = 'pool-visual-wrap';

  const quadDiv = document.createElement('div');
  quadDiv.innerHTML = '<div class="small dim mono" style="margin-bottom:4px;">2×2 Pool Window</div>';
  const quadGrid = document.createElement('div');
  quadGrid.className = 'pool-quad';
  quadGrid.style.gridTemplateColumns = 'repeat(2, 32px)';
  flat4.forEach((v, i) => {
    const r = Math.floor(i/2), c = i%2;
    const cell = document.createElement('div');
    cell.className = 'pool-quad-cell ' + (v === maxVal ? 'winner' : 'loser');
    cell.textContent = v.toFixed(2);
    if (v === maxVal) cell.title = '← MAX selected';
    quadGrid.appendChild(cell);
  });
  quadDiv.appendChild(quadGrid);

  const outDiv = document.createElement('div');
  outDiv.innerHTML = `<div class="small dim mono" style="margin-bottom:4px;">Max Pool Output</div>
    <div class="num-grid" style="grid-template-columns:36px;">
      <div class="num-cell highlight" style="font-weight:700;color:var(--accent3);">${maxVal.toFixed(3)}</div>
    </div>`;

  poolWrap.appendChild(quadDiv);
  poolWrap.innerHTML += '<div style="align-self:center;font-size:1.4rem;color:var(--accent);">→</div>';
  poolWrap.appendChild(outDiv);
  content.appendChild(poolWrap);

  content.appendChild(traceTable([
    ['values', flat4.map(v => v.toFixed(3)).join(', '), ''],
    ['max', `max(${flat4.map(v=>v.toFixed(2)).join(', ')})`, `= ${maxVal.toFixed(3)}`],
    ['avg (ref)', `avg(${flat4.map(v=>v.toFixed(2)).join(', ')})`, `= ${avgVal.toFixed(3)}`],
  ]));

  content.appendChild(statChips([
    ['Winner position', `[${winnerR}][${winnerC}]`],
    ['Max value', maxVal.toFixed(3)],
    ['Compression', '4 values → 1'],
  ]));

  return { title: '🗜️ Max Pooling', shape: '1×1 scalar', content };
}

/* ── Step 5: Flatten ── */
function buildStep_Flatten(input4, kernel3, bias) {
  const content = document.createElement('div');

  // In our toy example we have a 2×2 ReLU feature map
  const outConv = [];
  for (let r = 0; r < 2; r++) {
    const row = [];
    for (let c = 0; c < 2; c++) {
      let sum = bias;
      for (let kr = 0; kr < 3; kr++)
        for (let kc = 0; kc < 3; kc++)
          sum += input4[r+kr][c+kc] * kernel3[kr][kc];
      row.push(Math.max(0, parseFloat(sum.toFixed(3))));
    }
    outConv.push(row);
  }
  const flatVec = outConv.flat();

  content.appendChild(formulaBox(`
    <b>Flatten</b>  — reshape (H × W × C) → (H·W·C,)
    <span class="formula-comment">Converts multi-dimensional feature maps to a 1D vector so Dense layers can process them.
In the real CNN: 7×7×16 = 784 values. Here: 2×2×1 = 4 values.</span>
  `));

  const row = document.createElement('div');
  row.style.cssText = 'display:flex;gap:20px;flex-wrap:wrap;align-items:center;margin:8px 0;';

  const matWrap = document.createElement('div');
  matWrap.innerHTML = '<div class="small dim mono" style="margin-bottom:4px;">2D feature map (2×2)</div>';
  matWrap.appendChild(renderNumGrid(outConv));

  const vecWrap = document.createElement('div');
  vecWrap.innerHTML = '<div class="small dim mono" style="margin-bottom:4px;">Flattened 1D vector (4 elements)</div>';
  const vecGrid = document.createElement('div');
  vecGrid.className = 'num-grid';
  vecGrid.style.gridTemplateColumns = `repeat(${flatVec.length}, 56px)`;
  flatVec.forEach((v, i) => {
    const cell = document.createElement('div');
    cell.className = 'num-cell';
    const [rr, gg, bb] = heatColor(Math.max(0, Math.min(1, v / (Math.max(...flatVec) + 1e-6))));
    cell.style.background = `rgba(${rr},${gg},${bb},0.35)`;
    cell.textContent = v.toFixed(3);
    vecGrid.appendChild(cell);
  });
  vecWrap.appendChild(vecGrid);

  row.appendChild(matWrap);
  row.innerHTML += '<div style="font-size:1.4rem;color:var(--accent);">→ flatten →</div>';
  row.appendChild(vecWrap);
  content.appendChild(row);

  content.appendChild(traceTable(flatVec.map((v, i) => [
    `v[${i}]`,
    `from [${Math.floor(i/2)}][${i%2}]`,
    v.toFixed(3)
  ])));

  content.appendChild(statChips([
    ['Input shape', '2×2'],
    ['Output shape', `${flatVec.length} values`],
    ['Operation', 'row-major reshape'],
  ]));

  return { title: '⬡ Flatten', shape: `${flatVec.length}-vector`, content };
}

/* ── Step 6: Dense Layer ── */
function buildStep_Dense(input4, kernel3, bias) {
  const content = document.createElement('div');

  // Get flat vector
  const outConv = [];
  for (let r = 0; r < 2; r++) {
    const row = [];
    for (let c = 0; c < 2; c++) {
      let sum = bias;
      for (let kr = 0; kr < 3; kr++)
        for (let kc = 0; kc < 3; kc++)
          sum += input4[r+kr][c+kc] * kernel3[kr][kc];
      row.push(Math.max(0, parseFloat(sum.toFixed(3))));
    }
    outConv.push(row);
  }
  const x = outConv.flat(); // 4 inputs
  const nOut = 3; // 3 neurons for demo

  // Fixed reproducible weights seeded from input
  const seed = x.reduce((a,b) => a+b, 0);
  const W = Array.from({length: nOut}, (_, i) =>
    Array.from({length: x.length}, (_, j) =>
      parseFloat(((Math.sin(seed * (i+1) * (j+1) * 0.37) * 0.8)).toFixed(3))
    )
  );
  const b2 = Array.from({length: nOut}, (_, i) =>
    parseFloat((Math.cos(seed * (i+1) * 0.51) * 0.2).toFixed(3))
  );

  const preReLU = W.map((wRow, i) => {
    const dot = wRow.reduce((s, w, j) => s + w * x[j], b2[i]);
    return parseFloat(dot.toFixed(3));
  });
  const postReLU = preReLU.map(v => Math.max(0, v));

  content.appendChild(formulaBox(`
    <b>Dense layer:</b>  y<sub>i</sub> = ReLU( Σ<sub>j</sub> W<sub>ij</sub> · x<sub>j</sub> + b<sub>i</sub> )
    <span class="formula-comment">Each of the ${nOut} neurons computes a weighted sum of ALL ${x.length} inputs plus a bias, then applies ReLU.
In the real CNN: 784 → 64 neurons means 784×64 + 64 = 50,240 parameters.</span>
  `));

  // Show weight matrix
  const matDiv = document.createElement('div');
  matDiv.innerHTML = '<div class="small dim mono" style="margin-bottom:4px;">Weight matrix W (3×4) — rows=neurons, cols=inputs</div>';
  matDiv.appendChild(renderNumGrid(W));
  content.appendChild(matDiv);

  const traceRows = preReLU.map((v, i) => {
    const terms = W[i].map((w, j) => `${w}×${x[j].toFixed(2)}`).join(' + ');
    return [
      `neuron ${i}`,
      `${terms} + ${b2[i]} → ReLU(${v})`,
      `= ${Math.max(0, v).toFixed(3)}`
    ];
  });
  content.appendChild(traceTable(traceRows));

  const vecRow = document.createElement('div');
  vecRow.style.cssText = 'display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start;margin:10px 0;';

  const inVWrap = document.createElement('div');
  inVWrap.innerHTML = '<div class="small dim mono" style="margin-bottom:4px;">Input vector x</div>';
  const inVGrid = document.createElement('div');
  inVGrid.className = 'num-grid';
  inVGrid.style.gridTemplateColumns = `repeat(${x.length}, 54px)`;
  x.forEach(v => {
    const cell = document.createElement('div');
    cell.className = 'num-cell';
    const [rr, gg, bb] = heatColor(Math.max(0, Math.min(1, v / (Math.max(...x)+1e-6))));
    cell.style.background = `rgba(${rr},${gg},${bb},0.35)`;
    cell.textContent = v.toFixed(3);
    inVGrid.appendChild(cell);
  });
  inVWrap.appendChild(inVGrid);

  const outVWrap = document.createElement('div');
  outVWrap.innerHTML = '<div class="small dim mono" style="margin-bottom:4px;">Output neurons (post-ReLU)</div>';
  const outVGrid = document.createElement('div');
  outVGrid.className = 'num-grid';
  outVGrid.style.gridTemplateColumns = `repeat(${nOut}, 54px)`;
  postReLU.forEach(v => {
    const cell = document.createElement('div');
    cell.className = 'num-cell';
    const [rr, gg, bb] = heatColor(Math.max(0, Math.min(1, v / (Math.max(...postReLU)+1e-6))));
    cell.style.background = `rgba(${rr},${gg},${bb},0.35)`;
    cell.textContent = v.toFixed(3);
    outVGrid.appendChild(cell);
  });
  outVWrap.appendChild(outVGrid);

  vecRow.appendChild(inVWrap);
  vecRow.innerHTML += '<div style="align-self:center;font-size:1.4rem;color:var(--accent);">×W+b →</div>';
  vecRow.appendChild(outVWrap);
  content.appendChild(vecRow);

  content.appendChild(statChips([
    ['Input dim', x.length],
    ['Output neurons', nOut],
    ['Total weights', x.length * nOut + nOut],
    ['Activation', 'ReLU'],
  ]));

  return { title: '🔗 Dense Layer', shape: `${x.length}→${nOut}`, content };
}

/* ── Step 7: Softmax ── */
function buildStep_Softmax(input4, kernel3, bias) {
  const content = document.createElement('div');

  // Simulate 3 logits for demo (from dense step seed)
  const outConv = [];
  for (let r = 0; r < 2; r++) {
    const row = [];
    for (let c = 0; c < 2; c++) {
      let sum = bias;
      for (let kr = 0; kr < 3; kr++)
        for (let kc = 0; kc < 3; kc++)
          sum += input4[r+kr][c+kc] * kernel3[kr][kc];
      row.push(Math.max(0, parseFloat(sum.toFixed(3))));
    }
    outConv.push(row);
  }
  const x = outConv.flat();
  const seed = x.reduce((a,b) => a+b, 0);

  const nClasses = 5; // 5 classes for demo clarity
  const logits = Array.from({length: nClasses}, (_, i) =>
    parseFloat(((Math.sin(seed*(i+1)*0.37)*2).toFixed(3)))
  );

  const maxLogit = Math.max(...logits);
  const exps = logits.map(z => Math.exp(z - maxLogit)); // numerically stable
  const sumExp = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map(e => e / sumExp);
  const predicted = probs.indexOf(Math.max(...probs));

  content.appendChild(formulaBox(`
    <b>Softmax:</b>  P(class i) = exp(z<sub>i</sub>) / Σ<sub>j</sub> exp(z<sub>j</sub>)
    <span class="formula-comment">Converts raw logits (any real number) into probabilities that sum to 1.
We subtract max(z) before exponentiating for numerical stability — this doesn't change the result.
In the real CNN: ${nClasses} classes shown here → 10 classes (digits 0–9) in actual model.</span>
  `));

  // Show logit → exp → prob table
  const tableDiv = document.createElement('div');
  tableDiv.innerHTML = `<div class="small dim mono" style="margin-bottom:6px;">Computation per class:</div>`;
  tableDiv.appendChild(traceTable(logits.map((z, i) => [
    `class ${i}`,
    `exp(${z.toFixed(3)} − ${maxLogit.toFixed(3)}) = exp(${(z-maxLogit).toFixed(3)}) / ${sumExp.toFixed(3)}`,
    `P = ${(probs[i]*100).toFixed(1)}%`
  ])));
  content.appendChild(tableDiv);

  // Softmax bar chart
  const barDiv = document.createElement('div');
  barDiv.innerHTML = '<div class="small dim mono" style="margin-bottom:6px;margin-top:10px;">Probability distribution:</div>';
  probs.forEach((p, i) => {
    const row = document.createElement('div');
    row.className = 'softmax-row';
    const isTop = i === predicted;
    row.innerHTML = `
      <div class="softmax-lbl">${i}</div>
      <div class="softmax-bar-bg">
        <div class="softmax-bar-fill${isTop?' top-cls':''}" style="width:${(p*100).toFixed(1)}%"></div>
      </div>
      <div class="softmax-pct">${(p*100).toFixed(2)}%</div>
      <div class="softmax-logit">z=${logits[i].toFixed(3)}</div>`;
    barDiv.appendChild(row);
  });
  content.appendChild(barDiv);

  content.appendChild(statChips([
    ['Predicted class', predicted],
    ['Confidence', (probs[predicted]*100).toFixed(1)+'%'],
    ['Σ probabilities', probs.reduce((a,b)=>a+b,0).toFixed(4)],
  ]));

  return { title: '📊 Softmax Output', shape: `${nClasses} probs`, content };
}

/* ── Step 8: Loss ── */
function buildStep_Loss(input4, kernel3, bias) {
  const content = document.createElement('div');

  // Use same logits as softmax step
  const outConv = [];
  for (let r = 0; r < 2; r++) {
    const row = [];
    for (let c = 0; c < 2; c++) {
      let sum = bias;
      for (let kr = 0; kr < 3; kr++)
        for (let kc = 0; kc < 3; kc++)
          sum += input4[r+kr][c+kc] * kernel3[kr][kc];
      row.push(Math.max(0, parseFloat(sum.toFixed(3))));
    }
    outConv.push(row);
  }
  const x = outConv.flat();
  const seed = x.reduce((a,b)=>a+b,0);
  const nClasses = 5;
  const logits = Array.from({length:nClasses},(_, i) =>
    parseFloat(((Math.sin(seed*(i+1)*0.37)*2).toFixed(3)))
  );
  const maxLogit = Math.max(...logits);
  const exps = logits.map(z => Math.exp(z - maxLogit));
  const sumExp = exps.reduce((a,b)=>a+b,0);
  const probs = exps.map(e => e / sumExp);
  const predicted = probs.indexOf(Math.max(...probs));
  // Randomly pick a true label (different from predicted sometimes)
  const trueLabel = (predicted + Math.round(seed)) % nClasses;
  const p_true = probs[trueLabel];
  const loss = -Math.log(p_true + 1e-9);

  content.appendChild(formulaBox(`
    <b>Cross-Entropy Loss:</b>  L = −log( P(y<sub>true</sub>) )
    <span class="formula-comment">Measures how surprised the model is by the correct answer.
If P(true class) = 1.0 → loss = 0 (perfect). If P(true class) → 0 → loss → ∞.
During training, gradients of this loss flow backwards through all layers (backpropagation)
to update weights W so the model improves.</span>
  `));

  content.appendChild(traceTable([
    ['true label', `class ${trueLabel}`, ''],
    ['P(true)', `softmax probability of class ${trueLabel}`, p_true.toFixed(4)],
    ['loss', `-log(${p_true.toFixed(4)} + 1e-9)`, loss.toFixed(4)],
    ['predicted', `class ${predicted} (${(probs[predicted]*100).toFixed(1)}% confidence)`, predicted === trueLabel ? '✓ correct' : '✗ wrong'],
  ]));

  // Show where loss comes from
  const lossVizDiv = document.createElement('div');
  lossVizDiv.innerHTML = `
    <div class="formula-box" style="margin-top:12px; border-left-color:${predicted===trueLabel?'var(--success)':'var(--danger)'};">
      <b>Result:</b>
      True label = <span style="color:var(--accent3)">${trueLabel}</span> &nbsp;|&nbsp;
      Predicted = <span style="color:${predicted===trueLabel?'var(--success)':'var(--danger)'}">${predicted}</span> &nbsp;|&nbsp;
      L = <span style="color:var(--accent)">${loss.toFixed(4)}</span>
      <span class="formula-comment">${predicted===trueLabel
        ? '✅ Correct prediction! Backprop will make small adjustments to maintain this.'
        : '❌ Wrong prediction. Backprop will push probabilities toward class '+trueLabel+'.'}
      </span>
    </div>`;
  content.appendChild(lossVizDiv);

  // Gradient intuition
  content.appendChild(formulaBox(`
    <b>Gradient for backprop:</b>  ∂L/∂z<sub>i</sub> = P<sub>i</sub> − 1<sub>[i=true]</sub>
    <span class="formula-comment">The gradient is simply (predicted prob − one-hot target).
For correct class: P<sub>true</sub> − 1 (negative → increase this logit).
For other classes: P<sub>i</sub> − 0 (positive → decrease these logits).</span>
  `));

  const gradRow = document.createElement('div');
  gradRow.className = 'calc-trace';
  gradRow.innerHTML = '<div class="calc-trace-row" style="color:var(--accent3);font-weight:700;"><span class="calc-trace-idx">class</span><span class="calc-trace-expr">∂L/∂z = P − y_true</span><span class="calc-trace-val">gradient</span></div>';
  probs.forEach((p, i) => {
    const y = i === trueLabel ? 1 : 0;
    const g = (p - y).toFixed(4);
    const row = document.createElement('div');
    row.className = 'calc-trace-row';
    const isTrue = i === trueLabel;
    row.innerHTML = `<span class="calc-trace-idx" style="${isTrue?'color:var(--accent3)':''}">${i}${isTrue?' ←true':''}</span><span class="calc-trace-expr">${p.toFixed(4)} − ${y}</span><span class="calc-trace-val" style="color:${parseFloat(g)<0?'var(--success)':'var(--danger)'}">${g}</span>`;
    gradRow.appendChild(row);
  });
  content.appendChild(gradRow);

  content.appendChild(statChips([
    ['Loss value', loss.toFixed(4)],
    ['True class prob', (p_true*100).toFixed(2)+'%'],
    ['Result', predicted===trueLabel ? '✅ Correct' : '❌ Incorrect'],
  ]));

  return { title: '📉 Cross-Entropy Loss & Gradients', shape: 'scalar', content };
}

/* ================================================================
   KEYBOARD SHORTCUTS
   ================================================================ */
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  switch (e.key) {
    case ' ': e.preventDefault(); pauseTraining(); break;
    case 'r': case 'R': if (!e.ctrlKey) resetModel(); break;
    case 'c': case 'C': clearCanvas(); break;
    case 'p': case 'P': predictDrawing(); break;
  }
});

/* ================================================================
   UTILS
   ================================================================ */
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function updateLoader(msg) {
  document.getElementById('loaderText').textContent = msg;
}

/* ================================================================
   INITIALISATION
   ================================================================ */
async function init() {
  updateLoader('Initialising TensorFlow.js…');
  await tf.ready();
  updateLoader('Building CNN model…');
  buildModel();
  renderArchDiagram();
  updateLoader('Initialising charts…');
  initCharts();
  updateLoader('Setting up drawing canvas…');
  initDrawCanvas();
  updateLoader('Initialising CNN demos…');
  initConvDemo();
  initActivationDemo();
  initPoolDemo();
  initFlatDemo();
  initFullPipelineDemo();
  updateLoader('Generating training dataset…');
  await generateDataset();
  updateLoader('Generating computation walkthrough…');
  generateCompWalkthrough();
  updateLoader('Ready!');
  await sleep(400);
  document.getElementById('loaderOverlay').classList.add('hidden');
}

init().catch(err => {
  console.error(err);
  document.getElementById('loaderText').textContent = 'Error: ' + err.message;
});