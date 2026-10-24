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
let server, browser;
let url = process.env.SITE_URL;
const requests = [], blockedRequests = [], errors = [], logs = [];
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
  const page = await context.newPage();
  page.setDefaultTimeout(30 * 60 * 1000);
  page.on('pageerror', error => { errors.push(error.message); console.error('[site pageerror]', error.message); });
  page.on('console', message => { logs.push({ type: message.type(), text: message.text() }); if (message.type() === 'error') console.error('[site console]', message.text()); });
  await page.goto(url, { waitUntil: 'networkidle' });
  await page.waitForFunction(() => !document.getElementById('model-status-text').textContent.includes('Checking'));
  assert.equal(await page.locator('#predict').isDisabled(), true, 'AI inference is not automatic.');
  assert.equal(await page.locator('#load-model').isDisabled(), false, 'Published model must have passed its cloud release gate.');
  assert.equal(await page.locator('#image-source').textContent(), 'Demonstration image, not a patient scan');
  assert.equal(requests.filter(request => /\.onnx(?:\?|$)|\/runtime\//.test(request.url)).length, 0, 'No model download before a visitor explicitly requests it.');
  assert.equal(await page.locator('#uncertainty-results').isVisible(), false, 'No fabricated uncertainty result is prefilled.');
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
  const modelLoadStarted = Date.now();
  await page.locator('#load-model').click();
  console.log('Full website QA: explicit model download and integrity check started.');
  await page.waitForFunction(() => {
    const status = document.getElementById('model-status');
    return status.classList.contains('ready') || status.classList.contains('error');
  });
  assert.equal(await page.locator('#model-status').evaluate(node => node.classList.contains('ready')), true, await page.locator('#model-status-text').textContent());
  const modelDownloadAndLoadSeconds = (Date.now() - modelLoadStarted) / 1000;
  assert.equal(await page.locator('#predict').isDisabled(), false);
  const modelLink = await page.locator('#download-model').getAttribute('href');
  const decoderLink = await page.locator('#download-head').getAttribute('href');
  assert.ok(modelLink?.includes('backbone'), 'Backbone weights are downloadable.');
  assert.ok(decoderLink?.includes('stage2_decoder'), 'Decoder weights are downloadable.');
  assert.ok((await page.locator('#model-details').textContent()).includes('SHA-256'));
  await page.locator('#pass-count').selectOption('50');
  const started = Date.now();
  await page.locator('#predict').click();
  assert.equal(await page.locator('#cancel-predict').isVisible(), true);
  assert.equal(await page.locator('#progress-wrap').isVisible(), true);
  console.log('Full website QA: running 50 genuine stochastic passes on the GitHub cloud CPU.');
  await page.waitForFunction(() => !document.getElementById('uncertainty-results').hidden || document.getElementById('model-status').classList.contains('error'));
  const inferenceAndIntervalSamplingSeconds = (Date.now() - started) / 1000;
  assert.equal(await page.locator('#model-status').evaluate(node => node.classList.contains('error')), false, await page.locator('#model-status-text').textContent());
  const values = {};
  for (const key of ['alpha', 'beta']) {
    const label = await page.locator(`#${key}-value`).textContent();
    const value = Number.parseFloat(label);
    assert.ok(Number.isFinite(value) && value >= 0 && value <= 180, `${key} must be a finite unsigned angle.`);
    values[key] = value;
  }
  assert.equal(await page.locator('#measurement-mode').textContent(), 'Model mean coordinates');
  assert.ok((await page.locator('#uncertainty-badge').textContent()).includes('50 passes'));
  assert.ok((await page.locator('#sampling-note').textContent()).includes('4,096'));
  const intervalText = await page.locator('#interval-rows tr').allTextContents();
  assert.equal(intervalText.length, 3);
  for (const row of intervalText) assert.ok(!/Undefined|NaN|Infinity/.test(row) && row.match(/°/g)?.length === 4, 'All three distributions expose finite alpha and beta intervals.');
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
  assert.deepEqual(errors, [], 'No unhandled frontend JavaScript errors.');
  assert.deepEqual(blockedRequests, [], 'Neither the page nor its worker may contact an external origin.');
  assert.equal(requests.filter(request => !['GET', 'HEAD'].includes(request.method)).length, 0, 'No image upload or inference service request.');
  const proof = {
    status: 'passed', execution: 'GitHub Actions hosted Chromium, complete website interface', runId: process.env.GITHUB_RUN_ID,
    url, patientDataUsed: false, input: 'Generated synthetic canvas, not a patient scan', authorComputerInference: false,
    passes: 50, intervalsPerDistributionSamples: 4096, originalImageCoordinates: true, finiteAngles: values,
    browserModelDownloadsExplicit: true, modelIntegrityVerified: true, imageUploadRequests: 0,
    manualGeometryKeptSeparateFromInference: true, restoredModelLandmarks: true, mobileHorizontalOverflow: overflow,
    javascriptErrors: errors, externalRequestsBlocked: blockedRequests, allRequestOriginsRestrictedTo: allowedOrigin,
    modelDownloadAndLoadSeconds, inferenceAndIntervalSamplingSeconds,
    secondsIncludingInferenceAndInteractions: (Date.now() - started) / 1000,
    intervalRows: intervalText, requests, browserConsole: logs,
  };
  await writeFile(path.join(output, 'site_validation.json'), JSON.stringify(proof, null, 2) + '\n');
  console.log(JSON.stringify(proof, null, 2));
} catch (error) {
  await writeFile(path.join(output, 'site_failure.json'), JSON.stringify({ error: error.message, errors, blockedRequests, requests, logs }, null, 2) + '\n');
  throw error;
} finally {
  if (browser) await browser.close();
  if (server) await new Promise(resolve => server.close(resolve));
}
