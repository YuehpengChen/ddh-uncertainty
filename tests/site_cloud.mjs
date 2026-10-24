/** End-to-end frontend and genuine browser inference QA. GitHub cloud only. */
import assert from 'node:assert/strict';
import http from 'node:http';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright';

if (process.env.GITHUB_ACTIONS !== 'true' || !process.env.GITHUB_RUN_ID) {
  throw new Error('Refused: model and browser inference QA must execute in GitHub Actions, never on the author computer.');
}
const staticRoot = path.resolve(process.argv[2] || 'public');
const output = path.resolve(process.env.SITE_QA_OUTPUT || process.argv[3] || 'site-qa');
await mkdir(output, { recursive: true });
const mime = { '.html': 'text/html', '.css': 'text/css', '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm', '.onnx': 'application/octet-stream', '.png': 'image/png', '.svg': 'image/svg+xml' };
let server, browser, page;
let url = process.env.SITE_URL;
const requests = [], blockedRequests = [], errors = [], logs = [];
const UI_TIMEOUT = 90_000, MODEL_TIMEOUT = 30 * 60 * 1000;
const PAPER_TITLE = 'Estimating Landmark and Angle Uncertainty from Multiple Rater Annotations in Infant Hip Ultrasound';

async function instrumentOperationTimer(page) {
  await page.addInitScript(() => {
    const originalSetInterval = window.setInterval.bind(window), originalClearInterval = window.clearInterval.bind(window);
    const active = new Set();
    window.__qaTimers = { scheduled: 0, peakActive: 0, active: 0 };
    window.setInterval = (callback, delay, ...args) => {
      const id = originalSetInterval(callback, delay, ...args);
      if (Number(delay) === 1000) {
        active.add(id); window.__qaTimers.scheduled++;
        window.__qaTimers.active = active.size;
        window.__qaTimers.peakActive = Math.max(window.__qaTimers.peakActive, active.size);
      }
      return id;
    };
    window.clearInterval = id => {
      active.delete(id); window.__qaTimers.active = active.size;
      return originalClearInterval(id);
    };
  });
}

// Observe genuine worker messages and canvas calls without changing either.
// Full pass distributions are not copied; only the actual covariance summaries
// needed to verify the displayed landmark contours are retained.
async function instrumentContourGeometry(page) {
  await page.addInitScript(() => {
    window.__qaWorkerResults = []; window.__qaContourPaths = [];
    const NativeWorker = window.Worker;
    window.Worker = class extends NativeWorker {
      constructor(...args) {
        super(...args);
        this.addEventListener('message', event => {
          if (event.data?.type !== 'result') return;
          const { points, total, epistemic, conditional, passes, samples } = event.data.result;
          window.__qaWorkerResults.push(structuredClone({ id: event.data.id, points, total, epistemic, conditional, passes, samples }));
          window.__qaContourPaths = [];
        });
      }
    };
    const prototype = CanvasRenderingContext2D.prototype, paths = new WeakMap();
    for (const name of ['beginPath', 'moveTo', 'lineTo', 'closePath', 'stroke']) {
      const original = prototype[name];
      prototype[name] = function (...args) {
        if (this.canvas?.id === 'image-canvas') {
          if (name === 'beginPath') paths.set(this, { points: [], closed: false });
          const path = paths.get(this);
          if (path && ['moveTo', 'lineTo'].includes(name)) path.points.push(args.slice(0, 2));
          if (path && name === 'closePath') path.closed = true;
          if (path && name === 'stroke' && path.closed && path.points.length === 49) {
            window.__qaContourPaths.push(structuredClone(path.points));
          }
        }
        return Reflect.apply(original, this, args);
      };
    }
  });
}

async function verifyActualContourGeometry(page, label) {
  const observation = await page.evaluate(() => ({
    result: window.__qaWorkerResults.at(-1),
    paths: window.__qaContourPaths.slice(0, 5),
  }));
  const { result, paths } = observation;
  assert.ok(result, `${label}: a genuine inference worker result was observed.`);
  assert.equal(result.points.length, 5);
  assert.equal(paths.length, 5, `${label}: all five actual landmark contour paths were rendered.`);
  const rounded = value => Number(value.toPrecision(10));
  const matrix = (values, i) => [[values[i][i], values[i][i + 1]], [values[i + 1][i], values[i + 1][i + 1]]];
  const characteristics = ([[xx, xy], [yx, yy]]) => {
    assert.ok([xx, xy, yx, yy].every(Number.isFinite), 'Contour covariance entries are finite.');
    const root = Math.hypot(xx - yy, 2 * xy), maximum = (xx + yy + root) / 2, minimum = (xx + yy - root) / 2;
    return {
      covariance: [[xx, xy], [yx, yy]].map(row => row.map(rounded)),
      xyCorrelation: xx > 0 && yy > 0 ? rounded(xy / Math.sqrt(xx * yy)) : null,
      majorAxisDegreesFromImageX: rounded(.5 * Math.atan2(2 * xy, xx - yy) * 180 / Math.PI),
      majorToMinorAxisRatio: minimum > 0 ? rounded(Math.sqrt(maximum / minimum)) : null,
    };
  };
  const diagnostics = result.points.map((center, k) => {
    const i = k * 2, total = matrix(result.total, i), between = matrix(result.epistemic, i);
    const conditional = [[result.conditional[i], 0], [0, result.conditional[i + 1]]];
    for (let row = 0; row < 2; row++) for (let column = 0; column < 2; column++) {
      const expected = between[row][column] + conditional[row][column];
      assert.ok(Math.abs(total[row][column] - expected) <= 1e-9 * Math.max(1, Math.abs(expected)), `${label} P${k + 1}: total retains between-pass covariance plus conditional diagonal variance.`);
    }
    const [[xx, xy], [yx, yy]] = total, determinant = xx * yy - xy * yx;
    assert.ok(xx > 0 && yy > 0 && determinant > 0, `${label} P${k + 1}: actual contour covariance is positive definite.`);
    assert.ok(Math.abs(xy - yx) <= 1e-10 * Math.max(1, Math.abs(xy)), 'The observed covariance is symmetric.');
    const errors = paths[k].map(([x, y]) => {
      const dx = x - center[0], dy = y - center[1];
      // Independent inverse-matrix formula: no reuse of landmarkContour's
      // eigenvectors, axis lengths, rotation, or point generation.
      const squaredMahalanobis = (yy * dx * dx - (xy + yx) * dx * dy + xx * dy * dy) / determinant;
      return Math.abs(squaredMahalanobis - 4);
    });
    const maximumMahalanobisError = Math.max(...errors);
    assert.ok(maximumMahalanobisError < 1e-7, `${label} P${k + 1}: the actual rendered ellipse follows the full covariance, including xy, at radius 2.`);
    assert.ok(Math.hypot(paths[k][0][0] - paths[k].at(-1)[0], paths[k][0][1] - paths[k].at(-1)[1]) < 1e-8, `${label}: the contour closes.`);
    return {
      landmark: `P${k + 1}`, center: center.map(rounded),
      total: characteristics(total), conditional: characteristics(conditional),
      betweenPass: characteristics(between),
      renderedVertices: paths[k].length, radius: 2,
      maximumMahalanobisError: rounded(maximumMahalanobisError),
    };
  });
  return { observedResultId: result.id, passes: result.passes, actualRenderedCovarianceVerified: true, landmarks: diagnostics };
}

async function observeOperation(page, kind) {
  await page.evaluate(operation => {
    window.__qaObservation?.observer.disconnect();
    const records = [], scheduledBefore = window.__qaTimers.scheduled;
    const visible = node => !!node && !node.hidden && node.getClientRects().length > 0;
    let previous;
    const capture = () => {
      const wrap = document.getElementById(operation === 'load' ? 'load-progress-wrap' : 'progress-wrap');
      const bar = document.getElementById(operation === 'load' ? 'load-progress' : 'inference-progress');
      const elapsed = document.getElementById('operation-elapsed');
      if (!wrap || !bar || !elapsed) throw new Error('Operation progress and elapsed-time elements must exist.');
      const record = {
        phase: wrap.dataset.phase || '', visible: visible(wrap), value: bar.getAttribute('value'), max: bar.max,
        receivedBytes: Number(bar.dataset.receivedBytes || 0), totalBytes: Number(bar.dataset.totalBytes || 0),
        label: document.getElementById(operation === 'load' ? 'load-progress-label' : 'progress-label').textContent.trim(),
        elapsed: elapsed.textContent.trim(), elapsedVisible: visible(elapsed),
        ready: document.getElementById('model-status').classList.contains('ready'),
        resultsVisible: !document.getElementById('uncertainty-results').hidden,
      };
      const signature = JSON.stringify(record);
      if (signature !== previous) { records.push({ at: performance.now(), ...record }); previous = signature; }
    };
    const observer = new MutationObserver(capture);
    observer.observe(document.body, { subtree: true, childList: true, characterData: true, attributes: true,
      attributeFilter: ['value', 'max', 'data-phase', 'data-received-bytes', 'data-total-bytes', 'hidden', 'class'] });
    window.__qaObservation = { observer, records, capture, scheduledBefore };
    capture();
  }, kind);
}

async function finishObservation(page, label) {
  const proof = await page.evaluate(() => {
    const observation = window.__qaObservation; observation.capture(); observation.observer.disconnect();
    return { records: observation.records, timer: { ...window.__qaTimers, scheduledForOperation: window.__qaTimers.scheduled - observation.scheduledBefore } };
  });
  assert.equal(proof.timer.scheduledForOperation, 1, `${label}: one one-second elapsed timer is started.`);
  assert.equal(proof.timer.peakActive, 1, `${label}: elapsed timers never overlap.`);
  assert.equal(proof.timer.active, 0, `${label}: the elapsed timer stops when work stops.`);
  const busy = proof.records.filter(record => record.visible);
  assert.ok(busy.length > 0, `${label}: progress is visible while busy.`);
  let previousElapsed = 0;
  for (const record of busy) {
    assert.equal(record.elapsedVisible, true, `${label}: elapsed time is visible while busy.`);
    assert.match(record.elapsed, /\b\d+:[0-5]\d\b/, `${label}: elapsed time uses M:SS.`);
    const [, minutes, seconds] = record.elapsed.match(/\b(\d+):([0-5]\d)\b/);
    const elapsed = Number(minutes) * 60 + Number(seconds);
    assert.ok(elapsed >= previousElapsed, `${label}: elapsed time never moves backward while busy.`);
    previousElapsed = elapsed;
  }
  // Busy timers may be delayed by CI CPU contention. The stop handler updates
  // the final elapsed value synchronously, so compare that value to real time.
  const stopped = proof.records.at(-1), [, minutes, seconds] = stopped.elapsed.match(/\b(\d+):([0-5]\d)\b/);
  assert.ok(Math.abs(Number(minutes) * 60 + Number(seconds) - (stopped.at - busy[0].at) / 1000) < 2, `${label}: final elapsed time follows the actual browser clock.`);
  return proof;
}

function verifyDownloadProgress(proof) {
  const phased = proof.records.filter(record => ['download', 'verify', 'prepare', 'ready'].includes(record.phase));
  for (const record of phased.filter(record => record.totalBytes === 0)) assert.equal(record.value, null, 'An unknown download total remains indeterminate.');
  const records = phased.filter(record => record.totalBytes > 0);
  const downloads = records.filter(record => record.phase === 'download' && record.receivedBytes > 0);
  assert.ok(new Set(downloads.map(record => record.receivedBytes)).size >= 3, 'Several actual download byte updates are displayed.');
  assert.ok(new Set(downloads.filter(record => record.value !== null).map(record => record.receivedBytes)).size >= 3, 'Known byte totals produce several determinate download updates.');
  const total = records.at(-1).totalBytes;
  assert.ok(Number.isSafeInteger(total) && total > 0, 'Published model bytes provide a cumulative download denominator.');
  let received = 0;
  for (const record of records) {
    assert.equal(record.totalBytes, total, 'The cumulative model byte total stays fixed.');
    assert.ok(Number.isSafeInteger(record.receivedBytes) && record.receivedBytes >= received && record.receivedBytes <= total, 'Received model bytes are cumulative, monotonic, and bounded.');
    received = record.receivedBytes;
    if (record.phase === 'download' && record.value !== null) {
      assert.ok(Math.abs(Number(record.value) - 100 * record.receivedBytes / total) < 0.02, 'Determinate download percentage comes from received bytes.');
      if (record.receivedBytes < total) assert.ok(Number(record.value) < 100, 'Download never reaches 100% before all model bytes arrive.');
    }
    if (['verify', 'prepare'].includes(record.phase)) assert.equal(record.value, null, `${record.phase} remains indeterminate.`);
    if (record.phase !== 'ready') assert.equal(record.ready, false, 'A full download does not imply model readiness.');
  }
  for (const phase of ['verify', 'prepare']) assert.ok(records.some(record => record.phase === phase), `${phase} is reported separately from downloading.`);
  const ready = records.find(record => record.phase === 'ready' && record.ready);
  assert.ok(ready, 'Model load reports its ready phase.');
  assert.equal(Number(ready.value), 100); assert.equal(ready.max, 100);
  assert.equal(ready.receivedBytes, total, 'Ready retains the complete cumulative byte count.');
  assert.equal(ready.ready, true, 'The final 100% state is backed by loaded model sessions.');
}

function verifyInferenceProgress(proof, passes = 50) {
  const records = proof.records.filter(record => record.visible);
  for (const phase of ['preprocess', 'features', 'predict', 'intervals']) assert.ok(records.some(record => record.phase === phase), `${phase} is observed during actual inference.`);
  let completed = 0;
  for (const record of records) {
    if (record.phase === 'predict') {
      assert.equal(record.max, passes, 'Prediction progress uses the requested pass count.');
      assert.notEqual(record.value, null);
      const value = Number(record.value);
      assert.ok(Number.isInteger(value) && value >= completed && value <= passes, 'Only completed stochastic passes advance prediction progress.');
      assert.match(record.label, new RegExp(`\\b${value}\\s*\\/\\s*${passes}\\b`), 'Prediction text includes the completed/total pass count.');
      assert.match(record.label, /\d+(?:\.\d+)?\s*%/, 'Prediction text includes a percentage.');
      const percentage = Number(record.label.match(/(\d+(?:\.\d+)?)\s*%/)[1]);
      assert.ok(Math.abs(percentage - value / passes * 100) < 1, 'Prediction percentage matches the completed pass count.');
      completed = value;
    } else if (['preprocess', 'features', 'intervals'].includes(record.phase)) {
      assert.equal(record.value, null, `${record.phase} is indeterminate and never shows premature completion.`);
    }
  }
  assert.equal(completed, passes, 'All requested model passes are observed.');
  assert.ok(new Set(records.filter(record => record.phase === 'predict' && Number(record.value) > 0 && Number(record.value) < passes).map(record => Number(record.value))).size >= 3, 'Several distinct intermediate pass counts are observed.');
  assert.ok(new Set(records.map(record => record.elapsed)).size >= 2, 'Elapsed time advances during actual inference.');
}

async function verifyPaperMetadata(page) {
  assert.equal(await page.locator('h1#paper-title').textContent(), PAPER_TITLE);
  assert.equal(await page.locator('#paper-authors').textContent(), 'Authors: pending confirmation');
  assert.equal(await page.locator('#paper-info > details').evaluate(node => !node.open), true, 'Paper and licensing details start collapsed.');
  await page.locator('#paper-info > details > summary').click();
  const text = await page.locator('#paper-info').textContent();
  const licenses = await page.locator('#paper-info tbody tr').evaluateAll(nodes => nodes.map(node => [...node.querySelectorAll('td')].map(cell => cell.textContent.trim())));
  assert.match(licenses.find(([material]) => /paper.*code.*model/i.test(material))?.[1] || '', /Reuse license not specified/i, 'Paper, code, and model reuse is explicitly unspecified.');
  assert.match(licenses.find(([material]) => /example|sample/i.test(material))?.[1] || '', /Apache[\s\S]*2\.0/i, 'Public sample licensing is explicit.');
  assert.match(licenses.find(([material]) => /runtime/i.test(material))?.[1] || '', /\bMIT\b/i, 'Runtime licensing is explicit.');
  await page.locator('#paper-info > details > summary').click();
  return text;
}

async function verifyIntervalDetails(page) {
  assert.equal(await page.locator('#interval-details').evaluate(node => node.tagName === 'DETAILS' && !node.open), true, 'Detailed sampling information remains collapsed.');
  assert.equal(await page.locator('#sampling-note').isVisible(), false, 'The long sampling paragraph is hidden until requested.');
  assert.equal(await page.locator('#sample-count-details').isVisible(), false, 'Detailed sample counts are collapsed by default.');
  await page.locator('#interval-details > summary').click();
  assert.equal(await page.locator('#sampling-note').isVisible(), true);
  assert.equal(await page.locator('#sample-count-details').isVisible(), true);
  const counts = await page.locator('#sample-count-details').textContent();
  assert.equal(counts.match(/4,096 valid/g)?.length, 3, 'Every distribution retains its detailed valid count.');
  assert.equal(counts.match(/0 degenerate\s*\/\s*4,096 drawn/g)?.length, 3, 'Every distribution retains its detailed degeneracy and drawn counts.');
  await page.locator('#interval-details > summary').click();
}
try {
  if (!url) {
    server = http.createServer(async (request, response) => {
      try {
        let pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
        if (pathname.endsWith('/')) pathname += 'index.html';
        const target = path.resolve(staticRoot, '.' + pathname);
        if (!target.startsWith(staticRoot + path.sep)) throw new Error('Path outside static root.');
        const data = await readFile(target);
        response.writeHead(200, { 'Content-Type': mime[path.extname(target)] || 'application/octet-stream', 'Content-Length': data.byteLength, 'Cache-Control': 'no-store' });
        response.end(data);
      } catch (error) { response.writeHead(404); response.end(String(error.message)); }
    });
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    url = `http://127.0.0.1:${server.address().port}/`;
  }
  browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1, serviceWorkers: 'block' });
  const allowedOrigin = new URL(url).origin;
  // Browser-context interception includes the inference worker, unlike page-only hooks.
  await context.route('**/*', route => {
    const request = route.request(), requestURL = request.url();
    if (/^(blob:|data:)/.test(requestURL)) return route.continue();
    if (new URL(requestURL).origin !== allowedOrigin) {
      blockedRequests.push({ method: request.method(), url: requestURL });
      return route.abort('blockedbyclient');
    }
    return route.continue();
  });
  context.on('request', request => requests.push({ method: request.method(), url: request.url() }));
  page = await context.newPage();
  page.setDefaultTimeout(UI_TIMEOUT);
  await instrumentOperationTimer(page);
  await instrumentContourGeometry(page);
  page.on('pageerror', error => { errors.push(error.message); console.error('[site pageerror]', error.message); });
  page.on('console', message => { logs.push({ type: message.type(), text: message.text() }); if (message.type() === 'error') console.error('[site console]', message.text()); });
  await page.goto(url, { waitUntil: 'networkidle' });
  await page.waitForFunction(() => !document.getElementById('model-status-text').textContent.includes('Checking'));
  assert.equal(await page.locator('#predict').isDisabled(), true, 'AI inference is not automatic.');
  assert.equal(await page.locator('#load-model').isDisabled(), false, 'Published model must have passed its cloud release gate.');
  assert.equal(await page.locator('#image-source').textContent(), 'Demonstration image, not a patient scan');
  assert.equal(requests.filter(request => /\.onnx(?:\?|$)|\/runtime\//.test(request.url)).length, 0, 'No model download before a visitor explicitly requests it.');
  assert.equal(await page.locator('#uncertainty-results').isVisible(), false, 'No fabricated uncertainty result is prefilled.');
  assert.ok((await page.locator('#image-note').textContent()).includes('not model predictions'));
  const paperAndLicenseInformation = await verifyPaperMetadata(page);
  assert.equal(await page.locator('.model-provenance, #download-model, #download-head, #download-manifest').count(), 0, 'The model-details and download disclosure is removed.');
  await page.screenshot({ path: path.join(output, 'desktop-initial.png'), fullPage: true });

  // This generated canvas is explicitly synthetic and contains no patient data.
  const syntheticPNG = await page.evaluate(() => {
    const source = document.getElementById('image-canvas');
    return source.toDataURL('image/png').split(',')[1];
  });
  await page.locator('#image-file').setInputFiles({ name: 'synthetic-cloud-fixture.png', mimeType: 'image/png', buffer: Buffer.from(syntheticPNG, 'base64') });
  await page.waitForFunction(() => document.getElementById('image-source').textContent === 'Your local image · not uploaded');
  assert.ok((await page.locator('#image-note').textContent()).includes('manual placeholders'));
  assert.equal(await page.locator('#predict').isDisabled(), true);
  await observeOperation(page, 'load');
  const modelLoadStarted = Date.now();
  await page.locator('#load-model').click();
  console.log('Full website QA: explicit model download and integrity check started.');
  await page.waitForFunction(() => {
    const status = document.getElementById('model-status');
    return status.classList.contains('ready') || status.classList.contains('error');
  }, null, { timeout: MODEL_TIMEOUT });
  assert.equal(await page.locator('#model-status').evaluate(node => node.classList.contains('ready')), true, await page.locator('#model-status-text').textContent());
  const modelLoadProgress = await finishObservation(page, 'model load');
  verifyDownloadProgress(modelLoadProgress);
  const modelDownloadAndLoadSeconds = (Date.now() - modelLoadStarted) / 1000;
  assert.equal(await page.locator('#predict').isDisabled(), false);
  assert.ok((await page.locator('#model-details').textContent()).includes('SHA-256'));
  await page.locator('#pass-count').selectOption('50');
  await observeOperation(page, 'inference');
  const started = Date.now();
  await page.locator('#predict').click();
  assert.equal(await page.locator('#cancel-predict').isVisible(), true);
  assert.equal(await page.locator('#progress-wrap').isVisible(), true);
  console.log('Full website QA: running 50 genuine stochastic passes on the GitHub cloud CPU.');
  await page.waitForFunction(() => !document.getElementById('uncertainty-results').hidden || document.getElementById('model-status').classList.contains('error'), null, { timeout: MODEL_TIMEOUT });
  const inferenceAndIntervalSamplingSeconds = (Date.now() - started) / 1000;
  assert.equal(await page.locator('#model-status').evaluate(node => node.classList.contains('error')), false, await page.locator('#model-status-text').textContent());
  const inferenceProgress = await finishObservation(page, '50-pass inference');
  verifyInferenceProgress(inferenceProgress);
  assert.equal(await page.locator('#inference-progress').evaluate(node => node.value), 50);
  assert.equal(await page.locator('#progress-wrap').getAttribute('data-phase'), 'done');
  assert.equal(await page.locator('#progress-wrap').isVisible(), false, 'Completion hides the computation progress bar.');
  const values = {};
  for (const key of ['alpha', 'beta']) {
    const label = await page.locator(`#${key}-value`).textContent();
    const value = Number.parseFloat(label);
    assert.ok(Number.isFinite(value) && value >= 0 && value <= 180, `${key} must be a finite unsigned angle.`);
    values[key] = value;
  }
  assert.equal(await page.locator('#measurement-mode').textContent(), 'Model mean coordinates');
  const contourGeometry = await verifyActualContourGeometry(page, 'synthetic image genuine model inference');
  assert.ok((await page.locator('#uncertainty-badge').textContent()).includes('50 passes'));
  assert.ok((await page.locator('#sampling-note').textContent()).includes('4,096'));
  await verifyIntervalDetails(page);
  const intervalText = await page.locator('#interval-rows tr').allTextContents();
  assert.equal(intervalText.length, 3);
  for (const row of intervalText) assert.ok(!/Undefined|NaN|Infinity/.test(row) && row.match(/°/g)?.length === 4, 'All three distributions expose finite alpha and beta intervals.');
  for (const row of intervalText) assert.ok(row.includes('4,096 valid') && row.includes('0 degenerate / 4,096 drawn'), 'Every reported distribution has a complete, explicitly counted sample set.');
  assert.equal(await page.locator('#distribution-plot polyline').count(), 3);
  await page.locator('[data-angle="beta"]').click();
  assert.equal(await page.locator('[data-angle="beta"]').getAttribute('aria-pressed'), 'true');
  await page.screenshot({ path: path.join(output, 'desktop-prediction.png'), fullPage: true });
  await page.locator('.coordinates-details summary').click();
  const p4 = await page.locator('#coordinate-3-0').inputValue();
  await page.locator('#coordinate-3-0').fill(String(Number(p4) + 5));
  await page.locator('#coordinate-3-0').dispatchEvent('change');
  assert.equal(await page.locator('#measurement-mode').textContent(), 'Geometry exploration');
  assert.ok((await page.locator('#uncertainty-badge').textContent()).includes('unchanged by dragging'));
  assert.equal(await page.locator('#show-contours').isDisabled(), true);
  assert.deepEqual(await page.locator('#interval-rows tr').allTextContents(), intervalText, 'Geometry editing must not impersonate a new model inference.');
  await page.locator('#restore-prediction').click();
  assert.equal(await page.locator('#measurement-mode').textContent(), 'Model mean coordinates');
  assert.equal(await page.locator('#show-contours').isDisabled(), false);
  assert.equal(await page.locator('#coordinate-3-0').inputValue(), p4);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: path.join(output, 'mobile-prediction.png'), fullPage: true });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
  assert.equal(overflow, false, 'No horizontal overflow on a 390-pixel mobile viewport.');
  // Cancel only after observing at least one actual model pass; never simulate progress.
  await observeOperation(page, 'inference');
  await page.locator('#predict').click();
  await page.waitForFunction(() => !document.getElementById('progress-wrap').hidden && document.getElementById('progress-wrap').dataset.phase === 'predict' && document.getElementById('inference-progress').value >= 1, null, { timeout: MODEL_TIMEOUT });
  const completedBeforeCancellation = await page.locator('#inference-progress').evaluate(node => node.value);
  await page.locator('#cancel-predict').click();
  const cancellationProgress = await finishObservation(page, 'cancelled inference');
  assert.equal(await page.locator('#progress-wrap').isVisible(), false, 'Cancellation hides the computation progress bar.');
  assert.equal(await page.locator('#predict').isDisabled(), true, 'Cancelling terminates and unloads the worker.');
  assert.equal(await page.locator('#load-model').isDisabled(), false, 'A new worker requires an explicit model reload.');
  assert.ok((await page.locator('#model-status-text').textContent()).includes('stopped'));
  assert.ok((await page.locator('#uncertainty-badge').textContent()).includes('Previous completed result'));
  assert.deepEqual(await page.locator('#interval-rows tr').allTextContents(), intervalText, 'A cancelled partial run never replaces the previous valid intervals.');
  assert.deepEqual(errors, [], 'No unhandled frontend JavaScript errors.');
  assert.deepEqual(blockedRequests, [], 'Neither the page nor its worker may contact an external origin.');
  assert.equal(requests.filter(request => !['GET', 'HEAD'].includes(request.method)).length, 0, 'No image upload or inference service request.');
  const proof = {
    status: 'passed', execution: 'GitHub Actions hosted Chromium, complete website interface', runId: process.env.GITHUB_RUN_ID,
    url, patientDataUsed: false, input: 'Generated synthetic canvas, not a patient scan', authorComputerInference: false,
    passes: 50, intervalsPerDistributionSamples: 4096, originalImageCoordinates: true, finiteAngles: values,
    browserModelDownloadsExplicit: true, modelIntegrityVerified: true, imageUploadRequests: 0,
    manualGeometryKeptSeparateFromInference: true, restoredModelLandmarks: true, mobileHorizontalOverflow: overflow,
    cancellationAfterActualProgress: true, completedBeforeCancellation, cancelledWorkerUnloaded: true,
    javascriptErrors: errors, externalRequestsBlocked: blockedRequests, allRequestOriginsRestrictedTo: allowedOrigin,
    modelDownloadAndLoadSeconds, inferenceAndIntervalSamplingSeconds,
    modelLoadProgress, inferenceProgress, cancellationProgress, paperAndLicenseInformation, contourGeometry,
    secondsIncludingInferenceAndInteractions: (Date.now() - started) / 1000,
    intervalRows: intervalText, requests, browserConsole: logs,
  };
  await writeFile(path.join(output, 'site_validation.json'), JSON.stringify(proof, null, 2) + '\n');
  console.log(JSON.stringify(proof, null, 2));
} catch (error) {
  const operationProgress = page ? await page.evaluate(() => ({ records: window.__qaObservation?.records, timer: window.__qaTimers })).catch(() => null) : null;
  await writeFile(path.join(output, 'site_failure.json'), JSON.stringify({ error: error.message, stack: error.stack, operationProgress, errors, blockedRequests, requests, logs }, null, 2) + '\n');
  throw error;
} finally {
  if (browser) await browser.close();
  if (server) await new Promise(resolve => server.close(resolve));
}
