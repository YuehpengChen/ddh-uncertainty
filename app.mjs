import { LANDMARKS, grafAngles, landmarkContour, seededRandom } from './geometry.mjs';

const $ = id => document.getElementById(id);
const canvas = $('image-canvas');
const context = canvas.getContext('2d');
const state = {
  manifest: null, modelReady: false, worker: null, busy: false, image: null,
  width: 720, height: 620, uploaded: false, points: [], originalPoints: [],
  prediction: null, moved: false, dragIndex: null, displayAngle: 'alpha', sequence: 0,
};
const illustration = [[.14, .40], [.50, .40], [.72, .79], [.54, .43], [.72, .19]];
const COLORS = { points: '#b3d2ff', mixture: '#175cc5', full: '#242829', diagonal: '#8c9297' };

function say(message) {
  $('message').textContent = message;
  $('message').hidden = false;
  clearTimeout(say.timer);
  say.timer = setTimeout(() => { $('message').hidden = true; }, 8000);
}

function status(message, mode = '') {
  $('model-status-text').textContent = message;
  $('model-status').className = `model-status ${mode}`;
}

function updateControls() {
  const ready = state.manifest?.status === 'ready';
  $('load-model').disabled = !ready || state.busy || state.modelReady;
  $('load-model').textContent = state.modelReady ? 'Model loaded on this device' : 'Load browser model';
  $('predict').disabled = !state.modelReady || !state.uploaded || state.busy;
  $('pass-count').disabled = state.busy;
  $('cancel-predict').hidden = !state.busy;
  $('restore-prediction').hidden = !state.prediction || !state.moved;
  $('show-contours').disabled = !state.prediction || state.moved;
  $('show-raters').disabled = state.uploaded;
  for (const input of $('coordinate-fields').querySelectorAll('input')) input.disabled = state.busy;
}

function syntheticImage() {
  const source = document.createElement('canvas');
  source.width = 720; source.height = 620;
  const ctx = source.getContext('2d');
  const pixels = ctx.createImageData(720, 620), random = seededRandom(421);
  for (let y = 0; y < 620; y++) for (let x = 0; x < 720; x++) {
    const nearArc = Math.exp(-Math.pow((Math.hypot((x - 420) * .85, y - 290) - 180) / 32, 2));
    const soft = 10 + 19 * Math.exp(-Math.pow((x - 400) / 310, 2) - Math.pow((y - 290) / 280, 2));
    const noise = (random() - .5) * (26 + 30 * nearArc);
    const value = Math.max(4, Math.min(180, soft + nearArc * 33 + noise));
    const i = (y * 720 + x) * 4;
    pixels.data[i] = value; pixels.data[i + 1] = value + 2; pixels.data[i + 2] = value + 3; pixels.data[i + 3] = 255;
  }
  ctx.putImageData(pixels, 0, 0);
  ctx.strokeStyle = '#c2c9cc'; ctx.lineCap = 'round'; ctx.shadowColor = '#bec6c9'; ctx.shadowBlur = 15;
  ctx.lineWidth = 9; ctx.beginPath(); ctx.moveTo(40, 246); ctx.bezierCurveTo(150, 248, 290, 247, 390, 264); ctx.stroke();
  ctx.strokeStyle = '#a7afb1'; ctx.lineWidth = 7; ctx.beginPath(); ctx.moveTo(387, 266); ctx.bezierCurveTo(421, 321, 450, 405, 518, 490); ctx.stroke();
  ctx.strokeStyle = '#6e787d'; ctx.lineWidth = 5; ctx.beginPath(); ctx.moveTo(393, 266); ctx.quadraticCurveTo(465, 190, 520, 118); ctx.stroke();
  ctx.shadowBlur = 0; ctx.strokeStyle = '#3e474b'; ctx.lineWidth = 1;
  for (let x = 20; x < 720; x += 60) { ctx.beginPath(); ctx.moveTo(x, 10); ctx.lineTo(x, 18); ctx.stroke(); }
  return source;
}

function clearPrediction() {
  state.prediction = null; state.moved = false;
  $('uncertainty-empty').hidden = false; $('uncertainty-results').hidden = true;
  $('uncertainty-badge').textContent = 'Awaiting a model prediction';
}

function resetImage() {
  if (state.busy) cancelComputation(false);
  if (state.image?.close) state.image.close();
  state.image = syntheticImage(); state.width = 720; state.height = 620; state.uploaded = false;
  state.points = illustration.map(([x, y]) => [x * state.width, y * state.height]);
  state.originalPoints = state.points.map(p => [...p]);
  clearPrediction();
  $('image-source').textContent = 'Demonstration image, not a patient scan';
  $('image-note').textContent = 'This synthetic image and its example points explain the interface. They are not model predictions.';
  $('show-raters').checked = false;
  $('image-file').value = '';
  resizeCanvas(); buildCoordinateFields(); updateAngles(); updateControls(); draw();
}

function resizeCanvas() {
  const scale = Math.min(1, 1600 / Math.max(state.width, state.height));
  canvas.width = Math.round(state.width * scale); canvas.height = Math.round(state.height * scale);
  $('canvas-wrap').style.aspectRatio = `${state.width}/${state.height}`;
}

function draw() {
  const w = canvas.width, h = canvas.height, sx = w / state.width, sy = h / state.height;
  context.clearRect(0, 0, w, h);
  context.drawImage(state.image, 0, 0, w, h);
  context.save(); context.scale(sx, sy);
  const unit = Math.max(state.width, state.height) / 720;
  if (state.prediction && !state.moved && $('show-contours').checked) {
    state.prediction.points.forEach((p, k) => {
      const contour = landmarkContour(state.prediction.total, k);
      context.beginPath(); contour.forEach(([x, y], i) => i ? context.lineTo(p[0] + x, p[1] + y) : context.moveTo(p[0] + x, p[1] + y));
      context.closePath(); context.fillStyle = '#d5dce033'; context.strokeStyle = '#e0e4e5bb'; context.lineWidth = 1 * unit; context.fill(); context.stroke();
    });
  }
  if ($('show-raters').checked && !state.uploaded) {
    const shifts = [[-7, -5], [5, -3], [-3, 6], [6, 4], [1, 0]];
    context.fillStyle = '#e2e2e2';
    state.points.forEach((p, k) => shifts.forEach(([x, y], j) => {
      const scale = k === 3 ? .5 : 1;
      context.beginPath(); context.arc(p[0] + (x + j % 2) * scale, p[1] + y * scale, 2.1 * unit, 0, 2 * Math.PI); context.fill();
    }));
  }
  if ($('show-lines').checked) {
    const pairs = [[0, 1], [3, 2], [3, 4]];
    context.strokeStyle = '#bad6ff'; context.lineWidth = 1.6 * unit;
    for (const [i, j] of pairs) { context.beginPath(); context.moveTo(...state.points[i]); context.lineTo(...state.points[j]); context.stroke(); }
    // Only the geometric continuation of P1–P2 is dashed.
    const [a, b] = state.points;
    context.setLineDash([6 * unit, 6 * unit]); context.strokeStyle = '#bad6ff99';
    context.beginPath(); context.moveTo(...b); context.lineTo(b[0] + (b[0] - a[0]) * .7, b[1] + (b[1] - a[1]) * .7); context.stroke(); context.setLineDash([]);
  }
  state.points.forEach(([x, y], index) => {
    context.beginPath(); context.arc(x, y, 6 * unit, 0, Math.PI * 2); context.fillStyle = COLORS.points; context.fill();
    context.strokeStyle = '#18232d'; context.lineWidth = 1.5 * unit; context.stroke();
    const label = LANDMARKS[index];
    context.font = `600 ${13 * unit}px Arial`; context.textBaseline = 'middle'; context.lineWidth = 3.4 * unit; context.strokeStyle = '#111820';
    const tx = Math.min(state.width - 29 * unit, Math.max(5 * unit, x + 11 * unit)), ty = Math.max(15 * unit, Math.min(state.height - 16 * unit, y - 12 * unit));
    context.strokeText(label, tx, ty); context.fillStyle = '#e7f0ff'; context.fillText(label, tx, ty);
  });
  context.restore();
}

function buildCoordinateFields() {
  const container = $('coordinate-fields'); container.replaceChildren();
  state.points.forEach((p, i) => {
    const group = document.createElement('div'), title = document.createElement('strong'); title.textContent = LANDMARKS[i]; group.append(title);
    ['x', 'y'].forEach((axis, k) => {
      const label = document.createElement('label'); label.textContent = axis;
      const input = document.createElement('input'); input.type = 'number'; input.step = '.1'; input.min = '0'; input.max = String(k ? state.height : state.width);
      input.value = p[k].toFixed(1); input.id = `coordinate-${i}-${k}`; input.setAttribute('aria-label', `${LANDMARKS[i]} ${axis} coordinate in pixels`);
      input.addEventListener('change', () => {
        if (!Number.isFinite(input.valueAsNumber)) { input.value = state.points[i][k].toFixed(1); return; }
        state.points[i][k] = Math.max(0, Math.min(k ? state.height : state.width, input.valueAsNumber));
        geometryChanged();
      });
      label.append(input); group.append(label);
    }); container.append(group);
  });
}

function updateAngles() {
  const angles = grafAngles(state.points);
  for (const key of ['alpha', 'beta']) $(`${key}-value`).textContent = angles[key] === null ? '—' : `${angles[key].toFixed(1)}°`;
  $('measurement-mode').textContent = state.prediction && !state.moved ? 'Model mean coordinates' : 'Geometry exploration';
  $('overlay-key').textContent = state.prediction && !state.moved ? 'Model landmarks · original image coordinates' : state.uploaded ? 'Manual landmarks · geometry only' : 'Illustrative landmarks · drag to explore';
  state.points.forEach((p, i) => p.forEach((v, k) => { const input = $(`coordinate-${i}-${k}`); if (input && document.activeElement !== input) input.value = v.toFixed(1); }));
}

function geometryChanged() {
  if (state.prediction) { state.moved = true; $('uncertainty-badge').textContent = 'Model distribution · unchanged by dragging'; }
  updateAngles(); updateControls(); draw();
}

function displayedImageBox() {
  const box = canvas.getBoundingClientRect();
  const scale = Math.min(box.width / canvas.width, box.height / canvas.height);
  const width = canvas.width * scale, height = canvas.height * scale;
  return { left: box.left + (box.width - width) / 2, top: box.top + (box.height - height) / 2, width, height };
}

function eventPoint(event) {
  const box = displayedImageBox();
  return [(event.clientX - box.left) / box.width * state.width, (event.clientY - box.top) / box.height * state.height];
}

canvas.addEventListener('pointerdown', event => {
  if (state.busy) return;
  const p = eventPoint(event), radius = 24 * state.width / displayedImageBox().width;
  const distances = state.points.map(([x, y]) => Math.hypot(x - p[0], y - p[1]));
  const index = distances.indexOf(Math.min(...distances));
  if (distances[index] <= radius) { state.dragIndex = index; canvas.setPointerCapture(event.pointerId); }
});
canvas.addEventListener('pointermove', event => {
  if (state.dragIndex === null) return;
  const p = eventPoint(event); state.points[state.dragIndex] = [Math.max(0, Math.min(state.width, p[0])), Math.max(0, Math.min(state.height, p[1]))]; geometryChanged();
});
for (const event of ['pointerup', 'pointercancel', 'lostpointercapture']) canvas.addEventListener(event, () => { state.dragIndex = null; });

$('open-image').addEventListener('click', () => $('image-file').click());
$('image-file').addEventListener('change', async event => {
  const file = event.target.files[0]; if (!file) return;
  if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type) || file.size > 20 * 1024 * 1024) { say('Choose a PNG, JPEG or WebP image smaller than 20 MB.'); return; }
  try {
    const image = await createImageBitmap(file);
    if (image.width < 32 || image.height < 32 || image.width > 8192 || image.height > 8192) { image.close(); throw new Error('Image dimensions must be between 32 and 8192 pixels.'); }
    if (state.busy) cancelComputation(false);
    if (state.image?.close) state.image.close();
    state.image = image; state.width = image.width; state.height = image.height; state.uploaded = true;
    state.points = illustration.map(([x, y]) => [x * state.width, y * state.height]); state.originalPoints = state.points.map(p => [...p]); clearPrediction();
    $('image-source').textContent = 'Your local image · not uploaded';
    $('image-note').textContent = 'The initial points are manual placeholders, not model predictions. Run the model to replace them. Uploaded files stay in browser memory.';
    $('show-raters').checked = false;
    resizeCanvas(); buildCoordinateFields(); updateAngles(); updateControls(); draw();
    if (state.modelReady) say('Image opened locally. Select “Predict landmarks & intervals” to run the model.');
  } catch (error) { say(`Image could not be opened: ${error.message}`); }
});

function cancelComputation(announce = true) {
  state.sequence++; state.worker?.terminate(); state.worker = null; state.busy = false; state.modelReady = false;
  $('progress-wrap').hidden = true; status('Computation stopped. Reload the model to continue.'); updateControls();
  if (announce) say('Computation cancelled. No image has left this browser.');
}

function workerMessage(event) {
  const message = event.data;
  if (message.type === 'status') { status(message.message); return; }
  if (message.type === 'loaded') {
    state.modelReady = true; state.busy = false; $('progress-wrap').hidden = true; status('Model verified and ready on this device.', 'ready');
    $('model-details').textContent = `Two ONNX sessions · browser CPU (WASM) · weights verified with SHA-256. ${state.uploaded ? 'Ready for your image.' : 'Open your own image to enable prediction.'}`;
    updateControls(); return;
  }
  if (message.type === 'progress' && message.id === state.sequence) {
    $('inference-progress').max = message.total; $('inference-progress').value = message.completed;
    $('progress-label').textContent = message.message || `${message.completed} / ${message.total} dropout passes`; return;
  }
  if (message.type === 'result' && message.id === state.sequence) {
    state.busy = false; state.prediction = message.result; state.points = message.result.points.map(p => [...p]); state.moved = false;
    $('progress-wrap').hidden = true; $('image-note').textContent = 'Blue landmarks are the model’s average coordinates. Gray contours describe marginal Gaussian spread, not calibrated confidence regions. Dragging points switches to manual geometry.';
    if (state.points.some(([x, y]) => x < 0 || y < 0 || x > state.width || y > state.height)) $('image-note').textContent += ' Some predicted landmarks lie outside the image; they have not been clipped to the image boundary.';
    status(`${message.result.passes} stochastic passes completed on this device.`, 'ready');
    $('uncertainty-empty').hidden = true; $('uncertainty-results').hidden = false;
    $('uncertainty-badge').textContent = `${message.result.passes} passes · ${message.result.samples.toLocaleString()} samples`;
    buildCoordinateFields(); updateAngles(); updateControls(); draw(); renderUncertainty(); return;
  }
  if (message.type === 'error') {
    state.busy = false; $('progress-wrap').hidden = true;
    if (message.phase === 'load') { state.modelReady = false; state.worker?.terminate(); state.worker = null; }
    status(message.message || 'The computation failed.', 'error'); updateControls(); say(message.message || 'The computation failed. No placeholder prediction was substituted.');
  }
}

$('load-model').addEventListener('click', () => {
  if (state.busy || !state.manifest) return;
  state.busy = true; updateControls(); status('Downloading and verifying model files…');
  state.worker = new Worker(new URL('./inference-worker.mjs', import.meta.url), { type: 'module' });
  state.worker.addEventListener('message', workerMessage);
  state.worker.addEventListener('error', event => {
    state.busy = false; state.modelReady = false; state.worker?.terminate(); state.worker = null;
    status('The browser worker could not start. Check browser support and reload.', 'error'); updateControls(); say(event.message || 'Browser worker failed.');
  });
  state.worker.postMessage({ type: 'init', manifestURL: new URL('models/manifest.json', document.baseURI).href });
});

$('predict').addEventListener('click', () => {
  if (!state.modelReady || !state.uploaded || state.busy) return;
  const resized = document.createElement('canvas'); resized.width = 448; resized.height = 448;
  const ctx = resized.getContext('2d', { willReadFrequently: true }); ctx.drawImage(state.image, 0, 0, 448, 448);
  const image = ctx.getImageData(0, 0, 448, 448), passes = Number($('pass-count').value);
  state.busy = true; state.sequence++; updateControls();
  $('progress-wrap').hidden = false; $('inference-progress').max = passes; $('inference-progress').value = 0;
  $('progress-label').textContent = 'Extracting image features…'; status('Running locally in this visitor’s browser…');
  state.worker.postMessage({ type: 'predict', id: state.sequence, image, originalWidth: state.width, originalHeight: state.height, passes }, [image.data.buffer]);
});

function renderUncertainty() {
  if (!state.prediction) return;
  const prediction = state.prediction;
  const labels = { mixture: 'Same-pass Gaussian mixture', full: 'Gaussian · full covariance', diagonal: 'Gaussian · diagonal covariance' };
  const tbody = $('interval-rows'); tbody.replaceChildren();
  for (const key of ['mixture', 'full', 'diagonal']) {
    const row = document.createElement('tr'), name = document.createElement('td'), swatch = document.createElement('span');
    swatch.className = 'legend-swatch'; swatch.style.background = COLORS[key]; name.append(swatch, document.createTextNode(labels[key])); row.append(name);
    for (const angle of ['alpha', 'beta']) { const cell = document.createElement('td'), bounds = prediction.distributions[key].intervals[angle]; cell.textContent = bounds ? `${bounds[0].toFixed(1)}° – ${bounds[1].toFixed(1)}°` : 'Undefined'; row.append(cell); }
    tbody.append(row);
  }
  $('sampling-note').textContent = `${prediction.samples.toLocaleString()} joint configurations per distribution; 5th and 95th linearly interpolated percentiles. The dotted vertical line marks the angle at the mean model coordinates. The mixture selects one common dropout pass for all 10 coordinates. The two Gaussians use the same mean, coordinate variances and standard-normal draws; only cross-coordinate covariance changes. Mixture versus Gaussian also changes distributional shape. Intervals are nominal and uncalibrated. ${prediction.passes < 50 ? 'Fewer than the study’s 50 passes: exploratory setting. ' : ''}Units: original-image pixels and degrees.`;
  renderPlot();
}

function renderPlot() {
  if (!state.prediction) return;
  const svg = $('distribution-plot'), angle = state.displayAngle, ns = 'http://www.w3.org/2000/svg'; svg.replaceChildren();
  const all = Object.values(state.prediction.distributions).flatMap(d => d[angle]);
  const min = Math.max(0, Math.min(...all) - 2), max = Math.min(180, Math.max(...all) + 2), span = Math.max(max - min, 1);
  const left = 54, right = 930, top = 25, bottom = 195, bins = 64;
  const histograms = Object.entries(state.prediction.distributions).map(([name, distribution]) => {
    const counts = Array(bins).fill(0); for (const value of distribution[angle]) counts[Math.min(bins - 1, Math.max(0, Math.floor((value - min) / span * bins)))]++;
    return { name, counts: counts.map(n => n / Math.max(1, distribution[angle].length) / (span / bins)) };
  });
  const peak = Math.max(...histograms.flatMap(h => h.counts), .01);
  const node = (tag, attrs, text) => { const el = document.createElementNS(ns, tag); for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value); if (text !== undefined) el.textContent = text; svg.append(el); return el; };
  node('line', { x1: left, x2: right, y1: bottom, y2: bottom, stroke: '#777', 'stroke-width': 1 });
  for (let i = 0; i <= 3; i++) {
    const y = bottom - (bottom - top) * i / 3;
    node('text', { x: left - 9, y: y + 3, 'text-anchor': 'end', 'font-family': 'Arial', 'font-size': 10, fill: '#60686b' }, (peak * i / 3).toPrecision(2));
  }
  for (let i = 0; i <= 5; i++) {
    const x = left + (right - left) * i / 5, value = min + span * i / 5;
    node('line', { x1: x, x2: x, y1: bottom, y2: bottom + 5, stroke: '#777' });
    node('text', { x, y: bottom + 22, 'text-anchor': 'middle', 'font-family': 'Arial', 'font-size': 11, fill: '#60686b' }, `${value.toFixed(1)}°`);
  }
  node('text', { x: left, y: 14, 'font-family': 'Arial', 'font-size': 10, fill: '#60686b' }, 'Density (per degree)');
  node('text', { x: (left + right) / 2, y: 245, 'text-anchor': 'middle', 'font-family': 'Arial', 'font-size': 11, fill: '#60686b' }, angle === 'alpha' ? 'α angle (degrees)' : 'β angle (degrees)');
  for (const histogram of histograms.reverse()) {
    const points = histogram.counts.map((v, i) => `${left + (i + .5) / bins * (right - left)},${bottom - v / peak * (bottom - top)}`).join(' ');
    node('polyline', { points, fill: 'none', stroke: COLORS[histogram.name], 'stroke-width': histogram.name === 'mixture' ? 2.5 : 1.7, 'stroke-dasharray': histogram.name === 'diagonal' ? '6 4' : 'none', 'stroke-linejoin': 'round' });
  }
  const current = grafAngles(state.prediction.points)[angle];
  if (current !== null) { const x = left + (current - min) / span * (right - left); node('line', { x1: x, x2: x, y1: top, y2: bottom, stroke: '#175cc5', 'stroke-width': 1, 'stroke-dasharray': '2 4' }); }
}

for (const button of document.querySelectorAll('[data-angle]')) button.addEventListener('click', () => {
  state.displayAngle = button.dataset.angle;
  for (const other of document.querySelectorAll('[data-angle]')) { const selected = other === button; other.classList.toggle('active', selected); other.setAttribute('aria-pressed', String(selected)); }
  renderPlot();
});
for (const id of ['show-lines', 'show-raters', 'show-contours']) $(id).addEventListener('change', draw);
$('reset-image').addEventListener('click', resetImage);
$('cancel-predict').addEventListener('click', () => cancelComputation());
$('restore-prediction').addEventListener('click', () => { state.points = state.prediction.points.map(p => [...p]); state.moved = false; $('uncertainty-badge').textContent = `${state.prediction.passes} passes · ${state.prediction.samples.toLocaleString()} samples`; updateAngles(); updateControls(); draw(); });

async function inspectManifest() {
  try {
    const response = await fetch(new URL('models/manifest.json', document.baseURI), { cache: 'no-store' });
    if (!response.ok) throw new Error(`Model manifest returned HTTP ${response.status}.`);
    const manifest = await response.json(); state.manifest = manifest;
    if (manifest.repositoryURL && /^https:\/\/github\.com\//.test(manifest.repositoryURL)) $('repository-link').href = manifest.repositoryURL;
    if (manifest.status !== 'ready') { status('Model export and cloud validation are pending. Prediction is unavailable.'); $('model-details').textContent = manifest.message || 'The geometry workspace is available. It does not substitute for AI inference.'; updateControls(); return; }
    status('Published model available. Load it onto this device.');
    const downloads = manifest.models;
    const primary = Array.isArray(downloads) ? downloads[0] : downloads?.backbone;
    if (primary?.url) { const link = $('download-model'); link.href = new URL(primary.url, document.baseURI); link.hidden = false; link.textContent = 'Download backbone weights ↓'; }
    if (downloads?.head?.url) { const link = $('download-head'); link.href = new URL(downloads.head.url, document.baseURI); link.hidden = false; }
    const bytes = Array.isArray(downloads) ? downloads.reduce((s, m) => s + (m.bytes || 0), 0) : Object.values(downloads || {}).reduce((s, m) => s + (m.bytes || 0), 0);
    $('model-details').textContent = `${bytes ? `Model download: ${(bytes / 1024 / 1024).toFixed(0)} MB. ` : ''}CPU inference uses this visitor’s device; the author’s computer is not involved. Browser memory and runtime vary.`;
    updateControls();
  } catch (error) { status('The model is not available on this deployment.', 'error'); $('model-details').textContent = `${error.message} Geometry exploration remains available; no replacement predictions are generated.`; }
}

resetImage(); inspectManifest();
