/** Public-example end-to-end QA with genuine inference. Execute in GitHub cloud only. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import http from 'node:http';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright';

// This check precedes browser creation, file serving, and every model operation.
if (process.env.GITHUB_ACTIONS !== 'true' || !process.env.GITHUB_RUN_ID) {
  throw new Error('Refused: public-example model and browser inference QA must run in GitHub Actions, never on the author computer or an SSH host.');
}

const staticRoot = path.resolve(process.argv[2] || '_site');
const output = path.resolve(process.env.EXAMPLES_QA_OUTPUT || process.argv[3] || 'examples-qa');
const expected = [
  { id: 'spj-01', label: 'Example 1' },
  { id: 'spj-02', label: 'Example 2' },
  { id: 'spj-03', label: 'Example 3' },
];
const mime = {
  '.html': 'text/html', '.css': 'text/css', '.mjs': 'text/javascript', '.js': 'text/javascript',
  '.json': 'application/json', '.wasm': 'application/wasm', '.onnx': 'application/octet-stream',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.svg': 'image/svg+xml', '.txt': 'text/plain',
};
const requests = [], blockedRequests = [], errors = [], logs = [], workerURLs = [];
const examples = [], layoutChecks = [];
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
    window.__qaWorkerResults = []; window.__qaWorkerRequests = []; window.__qaContourPaths = [];
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
      postMessage(message, ...args) {
        window.__qaWorkerRequests.push({ type: message?.type, id: message?.id });
        return super.postMessage(message, ...args);
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

async function verifyActualContourGeometry(page, label, mode = 'total') {
  const observation = await page.evaluate(() => ({
    result: window.__qaWorkerResults.at(-1),
    paths: window.__qaContourPaths.slice(-5),
    selectedMode: document.getElementById('contour-mode').value,
  }));
  const { result, paths } = observation;
  assert.equal(observation.selectedMode, mode, `${label}: the selected coordinate-spread mode is explicit.`);
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
    const selected = mode === 'epistemic' ? between : total;
    const [[xx, xy], [yx, yy]] = selected, determinant = xx * yy - xy * yx;
    const tolerance = 1e-8 * Math.max(1, Math.abs(xx), Math.abs(xy), Math.abs(yy));
    assert.ok(xx >= -tolerance && yy >= -tolerance && determinant >= -tolerance * Math.max(1, xx, yy), `${label} P${k + 1}: the selected covariance is positive semidefinite.`);
    assert.ok(Math.abs(xy - yx) <= 1e-10 * Math.max(1, Math.abs(xy)), 'The observed covariance is symmetric.');
    // Forty-eight equally spaced contour vertices have second moment
    // radius^2 / 2 times the selected covariance. This independent check
    // also covers a line or point produced by a rank-deficient covariance.
    const offsets = paths[k].slice(0, -1).map(([x, y]) => [x - center[0], y - center[1]]);
    const renderedCovariance = [[0, 0], [0, 0]], renderedCenterOffset = [0, 0];
    for (const offset of offsets) for (let row = 0; row < 2; row++) {
      renderedCenterOffset[row] += offset[row] / offsets.length;
      for (let column = 0; column < 2; column++) renderedCovariance[row][column] += offset[row] * offset[column] / (offsets.length * 2);
    }
    assert.ok(renderedCenterOffset.every(value => Math.abs(value) < 1e-7), `${label} P${k + 1}: switching modes keeps the contour centered at the model mean.`);
    let maximumCovarianceError = 0;
    for (let row = 0; row < 2; row++) for (let column = 0; column < 2; column++) maximumCovarianceError = Math.max(maximumCovarianceError, Math.abs(renderedCovariance[row][column] - selected[row][column]));
    assert.ok(maximumCovarianceError < tolerance, `${label} P${k + 1}: the actual vertices retain the complete selected covariance and the same radius-2 original-pixel scale.`);
    const fullRank = xx > 0 && yy > 0 && determinant > 1e-12 * xx * yy;
    const errors = fullRank ? paths[k].map(([x, y]) => {
      const dx = x - center[0], dy = y - center[1];
      // Independent inverse-matrix formula: no reuse of landmarkContour's
      // eigenvectors, axis lengths, rotation, or point generation.
      const squaredMahalanobis = (yy * dx * dx - (xy + yx) * dx * dy + xx * dy * dy) / determinant;
      return Math.abs(squaredMahalanobis - 4);
    }) : [];
    const maximumMahalanobisError = fullRank ? Math.max(...errors) : null;
    if (fullRank) assert.ok(maximumMahalanobisError < 1e-7, `${label} P${k + 1}: the actual rendered ellipse follows the full covariance, including xy, at radius 2.`);
    assert.ok(Math.hypot(paths[k][0][0] - paths[k].at(-1)[0], paths[k][0][1] - paths[k].at(-1)[1]) < 1e-8, `${label}: the contour closes.`);
    return {
      landmark: `P${k + 1}`, center: center.map(rounded),
      total: characteristics(total), conditional: characteristics(conditional),
      betweenPass: characteristics(between),
      renderedVertices: paths[k].length, radius: 2, renderedCovariance: renderedCovariance.map(row => row.map(rounded)),
      maximumCovarianceError: rounded(maximumCovarianceError),
      maximumMahalanobisError: fullRank ? rounded(maximumMahalanobisError) : null,
    };
  });
  return { mode, observedResultId: result.id, passes: result.passes, actualRenderedCovarianceVerified: true, landmarks: diagnostics };
}

async function verifyContourModeSwitch(page, label) {
  const control = page.locator('#contour-mode');
  assert.equal(await control.isDisabled(), false, `${label}: a completed prediction enables the spread selector.`);
  assert.deepEqual(await control.locator('option').evaluateAll(nodes => nodes.map(node => ({ value: node.value, label: node.textContent.trim() }))), [
    { value: 'total', label: 'Total uncertainty' }, { value: 'epistemic', label: 'Prediction center variation' },
  ]);
  const snapshot = () => page.evaluate(() => ({
    points: [...document.querySelectorAll('.coordinate-fields input')].map(node => node.value),
    alpha: document.getElementById('alpha-value').textContent, beta: document.getElementById('beta-value').textContent,
    intervals: document.getElementById('interval-rows').innerHTML, plot: document.getElementById('distribution-plot').innerHTML,
    results: window.__qaWorkerResults.length, requests: structuredClone(window.__qaWorkerRequests),
    pointsFromWorker: window.__qaWorkerResults.at(-1).points,
  }));
  const before = await snapshot(), networkRequestsBefore = requests.length;
  assert.equal(await control.inputValue(), 'total');
  const total = await verifyActualContourGeometry(page, `${label}: total`, 'total');
  await page.screenshot({ path: path.join(output, `${label}-coordinate-spread-total.png`), fullPage: true });
  const pathsBefore = await page.evaluate(() => window.__qaContourPaths.length);
  await control.selectOption('epistemic');
  assert.ok(await page.evaluate(previous => window.__qaContourPaths.length >= previous + 5, pathsBefore), `${label}: changing modes immediately redraws all five contours.`);
  const epistemic = await verifyActualContourGeometry(page, `${label}: prediction-center variation`, 'epistemic');
  await page.screenshot({ path: path.join(output, `${label}-coordinate-spread-epistemic.png`), fullPage: true });
  assert.deepEqual(await snapshot(), before, `${label}: the selector does not change landmark positions, angles, interval calculations, or worker messages/results.`);
  assert.equal(requests.length, networkRequestsBefore, `${label}: switching spread modes performs no network or model request.`);
  await page.locator('#show-contours').uncheck();
  assert.equal(await control.isDisabled(), true, `${label}: hiding coordinate spread disables its selector.`);
  assert.equal(await control.inputValue(), 'epistemic', `${label}: hiding spread preserves the chosen mode.`);
  await page.locator('#show-contours').check();
  assert.equal(await control.isDisabled(), false);
  assert.equal(await control.inputValue(), 'epistemic');
  await verifyActualContourGeometry(page, `${label}: preserved prediction-center mode`, 'epistemic');
  await control.selectOption('total');
  await verifyActualContourGeometry(page, `${label}: restored total`, 'total');
  assert.deepEqual(await snapshot(), before, `${label}: restoring total is also a display-only change.`);
  assert.equal(requests.length, networkRequestsBefore);
  return { total, epistemic, sameCentersAndPixelScale: true, anglesAndIntervalsUnchanged: true, inferenceWorkerMessagesUnchanged: true,
    workerResultCount: before.results, workerRequestCount: before.requests.length, newNetworkRequests: 0, selectedModePreservedWhenHidden: true };
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
const started = Date.now();
let server, browser, page;
let url = process.env.SITE_URL;
await mkdir(output, { recursive: true });

function isModelAsset(requestURL) {
  const pathname = new URL(requestURL).pathname;
  return /\.onnx$|\.wasm$|\/runtime\//i.test(pathname);
}

function finiteAngle(value, message) {
  assert.ok(Number.isFinite(value) && value >= 0 && value <= 180, message);
  return value;
}

async function verifyPlainLayout(page, name) {
  const result = await page.evaluate(() => {
    const selectors = ['html', 'body', 'main', '.paper-header', '.workspace', '.controls-panel', '.uncertainty-panel'];
    const surfaces = selectors.map(selector => {
      const element = document.querySelector(selector);
      if (!element) throw new Error(`Required layout surface is missing: ${selector}`);
      const style = getComputedStyle(element);
      return { selector, backgroundColor: style.backgroundColor, backgroundImage: style.backgroundImage };
    });
    const heading = getComputedStyle(document.querySelector('h1'));
    return {
      viewport: { width: innerWidth, height: innerHeight }, surfaces,
      h1FontSize: Number.parseFloat(heading.fontSize), h1FontFamily: heading.fontFamily, h1TextShadow: heading.textShadow,
      horizontalOverflow: document.documentElement.scrollWidth > innerWidth + 1,
    };
  });
  assert.equal(result.surfaces[0].backgroundColor, 'rgb(255, 255, 255)', `${name}: the page background is white.`);
  for (const surface of result.surfaces) {
    assert.ok(['rgb(255, 255, 255)', 'rgba(0, 0, 0, 0)'].includes(surface.backgroundColor), `${name}: ${surface.selector} has a plain white or transparent background.`);
    assert.equal(surface.backgroundImage, 'none', `${name}: ${surface.selector} has no decorative background image.`);
  }
  assert.ok(result.h1FontSize > 0 && result.h1FontSize <= 26, `${name}: the heading is at most 26 pixels.`);
  assert.match(result.h1FontFamily, /Arial/i, `${name}: the heading uses the plain Arial typeface.`);
  assert.equal(result.h1TextShadow, 'none', `${name}: the heading has no decorative text shadow.`);
  assert.equal(result.horizontalOverflow, false, `${name}: there is no horizontal page overflow.`);
  layoutChecks.push({ name, ...result });
}

async function verifyAwaitingExample(page, item) {
  await page.waitForFunction(label => document.getElementById('image-source').textContent === `Public example · ${label}`, item.label);
  assert.equal(await page.locator('#alpha-value').textContent(), '—', `${item.id}: no alpha result before prediction.`);
  assert.equal(await page.locator('#beta-value').textContent(), '—', `${item.id}: no beta result before prediction.`);
  assert.equal(await page.locator('#measurement-mode').textContent(), 'Awaiting model prediction');
  assert.equal(await page.locator('#uncertainty-results').isVisible(), false, `${item.id}: changing the image clears the previous result.`);
  assert.equal(await page.locator('#interval-rows tr').count(), 0, `${item.id}: previous interval rows are cleared.`);
  assert.equal(await page.locator('#show-contours').isDisabled(), true);
  assert.equal(await page.locator('#contour-mode').isDisabled(), true, `${item.id}: choosing a new image disables the spread selector until a genuine prediction is available.`);
  assert.equal(await page.locator('#restore-prediction').isVisible(), false);
  assert.equal(await page.locator('#example-source-details').evaluate(node => node.tagName === 'DETAILS' && !node.open), true, `${item.id}: example attribution starts collapsed.`);
  assert.equal(await page.locator('#example-credit').isVisible(), false, `${item.id}: source details do not expand the default interface.`);
  await page.locator('#example-source-details > summary').click();
  assert.equal(await page.locator('#example-credit').isVisible(), true);
  const credit = await page.locator('#example-credit').textContent();
  assert.match(credit, /Apache[\s\S]*2\.0/i, `${item.id}: the displayed credit states the Apache 2.0 license.`);
  const creditLinks = await page.locator('#example-credit a[href]').evaluateAll(nodes => nodes.map(node => ({ text: node.textContent.trim(), href: node.href })));
  assert.ok(creditLinks.some(link => /^https:\/\//.test(link.href)), `${item.id}: the displayed credit links to its public source.`);
  await page.locator('#example-source-details > summary').click();

  // Compare every decoded pixel, so an illustrative point, contour, or line on a
  // public image fails this check even if no corresponding DOM overlay exists.
  const canvasCheck = await page.evaluate(async imageURL => {
    const response = await fetch(imageURL, { cache: 'no-store' });
    if (!response.ok) throw new Error(`Raw example fetch failed: ${response.status}`);
    const bitmap = await createImageBitmap(await response.blob(), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
    const raw = document.createElement('canvas');
    raw.width = bitmap.width; raw.height = bitmap.height;
    const rawContext = raw.getContext('2d', { willReadFrequently: true });
    rawContext.drawImage(bitmap, 0, 0);
    const displayed = document.getElementById('image-canvas');
    const dimensionsMatch = displayed.width === bitmap.width && displayed.height === bitmap.height;
    const originalPixels = rawContext.getImageData(0, 0, raw.width, raw.height).data;
    const displayedPixels = displayed.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, displayed.width, displayed.height).data;
    let differingChannels = 0;
    if (dimensionsMatch) for (let i = 0; i < originalPixels.length; i++) differingChannels += originalPixels[i] !== displayedPixels[i];
    bitmap.close();
    return { width: displayed.width, height: displayed.height, dimensionsMatch, differingChannels };
  }, item.absoluteURL);
  assert.equal(canvasCheck.dimensionsMatch, true, `${item.id}: the original example dimensions are displayed.`);
  assert.equal(canvasCheck.differingChannels, 0, `${item.id}: the public image has no pre-prediction landmark or measurement overlay.`);
  return { credit, creditLinks, canvasCheck };
}

async function verifyBusyControls(page, label) {
  const controls = await page.evaluate(() => {
    const selectors = ['#load-model', '#predict', '#pass-count', 'button[data-example-id]'];
    return selectors.flatMap(selector => [...document.querySelectorAll(selector)].map(node => ({ selector, id: node.id || node.dataset.exampleId, disabled: node.disabled })));
  });
  assert.equal(controls.filter(control => control.selector === 'button[data-example-id]').length, 3);
  for (const control of controls) assert.equal(control.disabled, true, `${label}: ${control.id} is disabled while busy.`);
  return controls;
}

try {
  if (!url) {
    server = http.createServer(async (request, response) => {
      try {
        if (!['GET', 'HEAD'].includes(request.method)) { response.writeHead(405); response.end(); return; }
        let pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
        if (pathname.endsWith('/')) pathname += 'index.html';
        const target = path.resolve(staticRoot, '.' + pathname);
        if (!target.startsWith(staticRoot + path.sep)) throw new Error('Path outside static root.');
        const data = await readFile(target);
        response.writeHead(200, { 'Content-Type': mime[path.extname(target)] || 'application/octet-stream', 'Content-Length': data.byteLength, 'Cache-Control': 'no-store' });
        response.end(request.method === 'HEAD' ? undefined : data);
      } catch (error) { response.writeHead(404); response.end(String(error.message)); }
    });
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    url = `http://127.0.0.1:${server.address().port}/`;
  }
  const allowedOrigin = new URL(url).origin;
  browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1, serviceWorkers: 'block' });
  // Context interception covers the worker as well as the page. No response is
  // fulfilled or mocked: every allowed model, runtime, and image is fetched.
  await context.route('**/*', route => {
    const request = route.request(), requestURL = request.url();
    if (/^(blob:|data:)/.test(requestURL)) return route.continue();
    const parsed = new URL(requestURL);
    if (parsed.origin !== allowedOrigin || !['GET', 'HEAD'].includes(request.method()) || /\/(?:api|v1)(?:\/|$)/i.test(parsed.pathname)) {
      blockedRequests.push({ method: request.method(), url: requestURL });
      return route.abort('blockedbyclient');
    }
    return route.continue();
  });
  context.on('request', request => requests.push({ method: request.method(), url: request.url(), resourceType: request.resourceType() }));
  page = await context.newPage();
  page.setDefaultTimeout(UI_TIMEOUT);
  await instrumentOperationTimer(page);
  await instrumentContourGeometry(page);
  page.on('worker', worker => workerURLs.push(worker.url()));
  page.on('pageerror', error => { errors.push(error.message); console.error('[examples pageerror]', error.message); });
  page.on('console', message => { logs.push({ type: message.type(), text: message.text() }); if (message.type() === 'error') console.error('[examples console]', message.text()); });
  await page.goto(url, { waitUntil: 'networkidle' });
  await page.waitForFunction(() => !document.getElementById('model-status-text').textContent.includes('Checking'));
  await page.locator('button[data-example-id]').last().waitFor({ state: 'visible' });
  const buttons = await page.locator('button[data-example-id]').evaluateAll(nodes => nodes.map(node => ({ id: node.dataset.exampleId, label: node.textContent.trim() })));
  assert.deepEqual(buttons, expected, 'Exactly the three labeled public examples are available.');
  assert.equal(await page.locator('#predict').isDisabled(), true);
  assert.equal(await page.locator('#load-model').isDisabled(), false, 'The cloud-verified model is available for an explicit load.');
  assert.equal(requests.filter(request => isModelAsset(request.url)).length, 0, 'The initial page does not download the neural model or runtime.');
  assert.equal(workerURLs.length, 0, 'The initial page does not create an inference worker.');
  assert.equal(await page.locator('#contour-mode').inputValue(), 'total', 'Total uncertainty is the default coordinate-spread mode.');
  assert.equal(await page.locator('#contour-mode').isDisabled(), true);
  assert.ok((await page.locator('#image-note').textContent()).includes('not model predictions'));
  const paperAndLicenseInformation = await verifyPaperMetadata(page);
  assert.equal(await page.locator('.model-provenance, #download-model, #download-head, #download-manifest').count(), 0, 'The model-details and download disclosure is removed.');
  await verifyPlainLayout(page, 'desktop-initial');
  await page.screenshot({ path: path.join(output, 'desktop-initial.png'), fullPage: true });

  const manifestURL = new URL('examples/manifest.json', page.url()).href;
  const manifest = await page.evaluate(async target => {
    const response = await fetch(target, { cache: 'no-store' });
    if (!response.ok) throw new Error(`Example manifest fetch failed: ${response.status}`);
    return response.json();
  }, manifestURL);
  const manifestItems = Array.isArray(manifest) ? manifest : manifest.items || manifest.examples;
  assert.ok(Array.isArray(manifestItems), 'The example manifest has an array of items.');
  assert.equal(manifestItems.length, 3);

  // Download and decode each unmodified source file in the cloud browser, then
  // verify its raw-byte hash independently with Node crypto as well.
  for (const expectedItem of expected) {
    const item = manifestItems.find(candidate => candidate.id === expectedItem.id);
    assert.ok(item, `${expectedItem.id}: a manifest entry exists.`);
    assert.equal(item.label, expectedItem.label);
    assert.equal(item.width, 256); assert.equal(item.height, 256);
    assert.ok(Number.isSafeInteger(item.bytes) && item.bytes > 0);
    assert.match(item.sha256, /^[a-f0-9]{64}$/i);
    const absoluteURL = new URL(item.url, page.url()).href;
    assert.equal(new URL(absoluteURL).origin, allowedOrigin, 'Public example image files use the hosting origin.');
    const fetched = await page.evaluate(async imageURL => {
      const response = await fetch(imageURL, { cache: 'no-store' });
      if (!response.ok) throw new Error(`Example image fetch failed: ${response.status}`);
      const data = await response.arrayBuffer();
      const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', data))].map(byte => byte.toString(16).padStart(2, '0')).join('');
      const image = await createImageBitmap(new Blob([data]), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
      const result = { bytes: [...new Uint8Array(data)], byteLength: data.byteLength, sha256: hash, width: image.width, height: image.height, responseURL: response.url };
      image.close(); return result;
    }, absoluteURL);
    assert.equal(new URL(fetched.responseURL).origin, allowedOrigin);
    assert.equal(fetched.byteLength, item.bytes, `${item.id}: byte count matches the manifest.`);
    assert.equal(fetched.width, 256); assert.equal(fetched.height, 256);
    assert.equal(fetched.sha256, item.sha256.toLowerCase(), `${item.id}: browser SHA-256 matches the manifest.`);
    assert.equal(createHash('sha256').update(Buffer.from(fetched.bytes)).digest('hex'), item.sha256.toLowerCase(), `${item.id}: independent Node SHA-256 matches the manifest.`);
    const validatedItem = { ...item, absoluteURL };
    examples.push({ id: item.id, label: item.label, url: absoluteURL, sha256: fetched.sha256, bytes: fetched.byteLength, width: fetched.width, height: fetched.height });
    await page.locator(`button[data-example-id="${item.id}"]`).click();
    const beforePrediction = await verifyAwaitingExample(page, validatedItem);
    examples.at(-1).beforePrediction = beforePrediction;
    assert.equal(await page.locator('#predict').isDisabled(), true, 'Selecting an image never loads the model.');
    assert.equal(requests.filter(request => isModelAsset(request.url)).length, 0, `${item.id}: example selection did not download any model or runtime.`);
    assert.equal(workerURLs.length, 0);
    await page.screenshot({ path: path.join(output, `${item.id}-before-prediction.png`), fullPage: true });
  }

  await page.locator('button[data-example-id="spj-01"]').click();
  await verifyAwaitingExample(page, { ...manifestItems.find(item => item.id === 'spj-01'), absoluteURL: examples[0].url });
  await page.setViewportSize({ width: 390, height: 844 });
  await verifyPlainLayout(page, 'mobile-initial');
  await page.screenshot({ path: path.join(output, 'mobile-initial.png'), fullPage: true });
  await page.setViewportSize({ width: 1440, height: 1000 });

  await observeOperation(page, 'load');
  const modelLoadStarted = Date.now();
  await page.locator('#load-model').click();
  const modelLoadBusyControls = await verifyBusyControls(page, 'model load');
  console.log('Public-example QA: downloading and verifying the browser model once in GitHub cloud.');
  await page.waitForFunction(() => {
    const node = document.getElementById('model-status');
    return node.classList.contains('ready') || node.classList.contains('error');
  }, null, { timeout: MODEL_TIMEOUT });
  assert.equal(await page.locator('#model-status').evaluate(node => node.classList.contains('ready')), true, await page.locator('#model-status-text').textContent());
  const modelLoadProgress = await finishObservation(page, 'model load');
  verifyDownloadProgress(modelLoadProgress);
  assert.ok((await page.locator('#model-details').textContent()).includes('SHA-256'));
  assert.equal(workerURLs.filter(workerURL => /\/inference-worker\.mjs(?:\?|$)/.test(workerURL)).length, 1, 'One genuine inference worker is loaded explicitly.');
  const modelDownloadAndLoadSeconds = (Date.now() - modelLoadStarted) / 1000;
  const modelRequestsAfterLoad = requests.filter(request => isModelAsset(request.url)).length;
  assert.ok(modelRequestsAfterLoad >= 2, 'The real published model and runtime assets were fetched.');
  await page.locator('#pass-count').selectOption('50');

  for (let index = 0; index < examples.length; index++) {
    const record = examples[index], item = { ...manifestItems.find(candidate => candidate.id === record.id), absoluteURL: record.url };
    if (index) {
      assert.equal(await page.locator('#uncertainty-results').isVisible(), true);
      await page.locator(`button[data-example-id="${record.id}"]`).click();
      record.clearedPreviousPrediction = true;
    }
    record.awaitingImmediatelyBeforePrediction = await verifyAwaitingExample(page, item);
    assert.equal(await page.locator('#predict').isDisabled(), false, `${record.id}: the explicitly loaded model is ready for this selected image.`);
    assert.equal(requests.filter(request => isModelAsset(request.url)).length, modelRequestsAfterLoad, 'Switching examples does not reload model files.');
    await observeOperation(page, 'inference');
    const inferenceStarted = Date.now();
    await page.locator('#predict').click();
    record.busyControls = await verifyBusyControls(page, record.id);
    assert.equal(await page.locator('#cancel-predict').isVisible(), true);
    assert.equal(await page.locator('#progress-wrap').isVisible(), true);
    console.log(`Public-example QA: ${record.id}, 50 genuine stochastic passes on the GitHub cloud CPU.`);
    await page.waitForFunction(() => (document.getElementById('progress-wrap').dataset.phase === 'predict' && document.getElementById('inference-progress').value >= 1) || document.getElementById('model-status').classList.contains('error'), null, { timeout: MODEL_TIMEOUT });
    assert.equal(await page.locator('#model-status').evaluate(node => node.classList.contains('error')), false, await page.locator('#model-status-text').textContent());
    record.actualPassProgressObserved = await page.locator('#inference-progress').evaluate(node => node.value);
    assert.ok(record.actualPassProgressObserved >= 1);
    await page.waitForFunction(() => !document.getElementById('uncertainty-results').hidden || document.getElementById('model-status').classList.contains('error'), null, { timeout: MODEL_TIMEOUT });
    assert.equal(await page.locator('#model-status').evaluate(node => node.classList.contains('error')), false, await page.locator('#model-status-text').textContent());
    record.inferenceProgress = await finishObservation(page, record.id);
    verifyInferenceProgress(record.inferenceProgress);
    record.inferenceAndIntervalSamplingSeconds = (Date.now() - inferenceStarted) / 1000;
    record.finiteAngles = {};
    for (const angle of ['alpha', 'beta']) record.finiteAngles[angle] = finiteAngle(Number.parseFloat(await page.locator(`#${angle}-value`).textContent()), `${record.id}: ${angle} is a finite unsigned angle.`);
    assert.equal(await page.locator('#measurement-mode').textContent(), 'Model mean coordinates');
    record.contourGeometry = await verifyActualContourGeometry(page, record.id);
    record.contourModeSwitch = await verifyContourModeSwitch(page, record.id);
    assert.match(await page.locator('#uncertainty-badge').textContent(), /50 passes/);
    assert.match(await page.locator('#sampling-note').textContent(), /4,096/);
    await verifyIntervalDetails(page);
    record.completedPasses = await page.locator('#inference-progress').evaluate(node => node.value);
    assert.equal(record.completedPasses, 50, `${record.id}: the run completes all 50 passes.`);
    assert.equal(await page.locator('#progress-wrap').getAttribute('data-phase'), 'done');
    assert.equal(await page.locator('#progress-wrap').isVisible(), false, `${record.id}: completion hides the computation progress bar.`);
    const rows = await page.locator('#interval-rows tr').evaluateAll(nodes => nodes.map(node => [...node.querySelectorAll('td')].map(cell => cell.textContent.trim())));
    assert.equal(rows.length, 3);
    record.intervals = rows.map((cells, rowIndex) => {
      assert.equal(cells.length, 3);
      assert.ok(cells[0].includes('4,096 valid') && cells[0].includes('0 degenerate / 4,096 drawn'), `${record.id}: distribution ${rowIndex + 1} reports all 4,096 valid samples and zero degeneracies.`);
      const intervals = {};
      for (const [column, angle] of [[1, 'alpha'], [2, 'beta']]) {
        const numbers = cells[column].match(/\d+(?:\.\d+)?/g)?.map(Number);
        assert.equal(numbers?.length, 2, `${record.id}: the ${angle} interval has exactly two numeric bounds.`);
        for (const value of numbers) finiteAngle(value, `${record.id}: every ${angle} interval bound is finite and between 0 and 180 degrees.`);
        assert.ok(numbers[0] <= numbers[1], `${record.id}: ${angle} interval bounds are ordered.`);
        intervals[angle] = numbers;
      }
      return { distribution: cells[0], ...intervals, validSamples: 4096, degenerateSamples: 0, drawnSamples: 4096 };
    });
    assert.equal(await page.locator('#distribution-plot polyline').count(), 3);
    await page.locator('[data-angle="beta"]').click();
    assert.equal(await page.locator('[data-angle="beta"]').getAttribute('aria-pressed'), 'true');
    assert.equal(await page.locator(`button[data-example-id="${record.id}"]`).isDisabled(), false, 'Example controls are available again after completion.');
    await verifyPlainLayout(page, `${record.id}-desktop-prediction`);
    await page.screenshot({ path: path.join(output, `${record.id}-desktop-prediction.png`), fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    await verifyPlainLayout(page, `${record.id}-mobile-prediction`);
    await page.screenshot({ path: path.join(output, `${record.id}-mobile-prediction.png`), fullPage: true });
    await page.setViewportSize({ width: 1440, height: 1000 });
  }

  assert.equal(workerURLs.length, 1, 'All three examples reuse the single explicitly loaded inference worker.');
  assert.equal(requests.filter(request => isModelAsset(request.url)).length, modelRequestsAfterLoad, 'The model loads only once for all three examples.');
  assert.deepEqual(errors, [], 'There are no unhandled frontend JavaScript errors.');
  assert.deepEqual(blockedRequests, [], 'The page and inference worker attempt no external-origin, upload, or API request.');
  const networkRequests = requests.filter(request => /^https?:/.test(request.url));
  assert.equal(networkRequests.filter(request => !['GET', 'HEAD'].includes(request.method)).length, 0, 'Images are not uploaded.');
  for (const request of networkRequests) assert.equal(new URL(request.url).origin, allowedOrigin, 'Every network request stays on the hosting origin.');
  const imageURLs = new Set(examples.map(item => item.url));
  const publicImageRequests = requests.filter(request => imageURLs.has(request.url));
  assert.ok(publicImageRequests.length >= 3);
  for (const request of publicImageRequests) assert.equal(request.method, 'GET', 'Public example images are only retrieved with GET.');
  const proof = {
    status: 'passed', execution: 'GitHub Actions hosted Chromium, complete website interface and genuine browser CPU inference',
    runId: process.env.GITHUB_RUN_ID, url, input: 'Three public deidentified source images',
    authorComputerInference: false, sshHostInference: false, clinicalAccuracyAssessed: false,
    passesPerExample: 50, intervalsPerDistributionSamples: 4096, originalImageCoordinates: true,
    browserModelDownloadsExplicit: true, modelLoadCount: 1, modelIntegrityVerified: true,
    rawExampleIntegrityVerifiedWithBrowserAndNodeSHA256: true, rawExampleDimensionsVerified: true,
    publicExamplesUnannotatedBeforePrediction: true, sampleChangesClearPreviousPrediction: true,
    imageUploadRequests: 0, remoteInferenceRequests: 0, allRequestOriginsRestrictedTo: allowedOrigin,
    publicImageRequests, workerURLs, modelLoadBusyControls, modelDownloadAndLoadSeconds,
    modelLoadProgress, paperAndLicenseInformation,
    secondsIncludingInferenceAndInteractions: (Date.now() - started) / 1000,
    layoutChecks, examples, javascriptErrors: errors, externalRequestsBlocked: blockedRequests,
    requests, browserConsole: logs,
  };
  await writeFile(path.join(output, 'examples_validation.json'), JSON.stringify(proof, null, 2) + '\n');
  console.log(JSON.stringify(proof, null, 2));
} catch (error) {
  const operationProgress = page ? await page.evaluate(() => ({ records: window.__qaObservation?.records, timer: window.__qaTimers })).catch(() => null) : null;
  await writeFile(path.join(output, 'examples_failure.json'), JSON.stringify({ error: error.message, stack: error.stack, operationProgress, url, examples, layoutChecks, errors, blockedRequests, requests, logs }, null, 2) + '\n');
  throw error;
} finally {
  if (browser) await browser.close();
  if (server) await new Promise(resolve => server.close(resolve));
}
