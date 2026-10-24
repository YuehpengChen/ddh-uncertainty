/** Cloud-only UI verification. Does not load or evaluate neural models. */
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright';

if (process.env.GITHUB_ACTIONS !== 'true' || !process.env.GITHUB_RUN_ID) {
  throw new Error('Run this browser QA in GitHub Actions; no browser model runs are permitted on the author computer.');
}
const base = process.argv[2];
if (!base || !/^https?:\/\//.test(base)) throw new Error('Pass the HTTP URL of the staged frontend.');
const output = path.resolve(process.argv[3] || 'frontend-qa');
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true });
const requests = [], errors = [];
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
  page.on('request', request => requests.push({ method: request.method(), url: request.url() }));
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(base, { waitUntil: 'networkidle' });
  await page.locator('#model-status-text').waitFor();
  assert.equal(await page.locator('#predict').isDisabled(), true, 'Prediction requires explicit model load and a local image.');
  assert.equal(await page.locator('#image-source').textContent(), 'Demonstration image, not a patient scan');
  assert.equal(await page.locator('#uncertainty-results').isVisible(), false);
  assert.equal(await page.locator('.paper-header').evaluate(node => getComputedStyle(node).textAlign), 'center', 'Paper header follows the centered reference-demo layout.');
  assert.equal(await page.locator('#contour-mode').inputValue(), 'total');
  assert.equal(await page.locator('#contour-mode').isDisabled(), true, 'Contour selection requires a genuine completed prediction.');
  assert.deepEqual(await page.locator('#contour-mode option').allTextContents(), ['總不確定性', '多次預測中心的變動']);
  assert.ok(await page.locator('#canvas-wrap').evaluate(node => node.getBoundingClientRect().width <= 720), 'A single image remains bounded rather than filling an entire large screen.');
  assert.ok((await page.locator('#image-note').textContent()).includes('not model predictions'));
  await page.screenshot({ path: path.join(output, 'desktop.png'), fullPage: true });
  const originalAngles = [await page.locator('#alpha-value').textContent(), await page.locator('#beta-value').textContent()];
  await page.locator('.coordinates-details summary').click();
  await page.locator('#coordinate-3-0').fill('420');
  await page.locator('#coordinate-3-0').dispatchEvent('change');
  const changedAngles = [await page.locator('#alpha-value').textContent(), await page.locator('#beta-value').textContent()];
  assert.notEqual(changedAngles[0], originalAngles[0], 'P4 changes alpha.');
  assert.notEqual(changedAngles[1], originalAngles[1], 'P4 changes beta.');
  assert.equal(await page.locator('#measurement-mode').textContent(), 'Geometry exploration');
  // The uploaded fixture is this application's synthetic canvas, not a patient image.
  const png = await page.locator('#image-canvas').screenshot();
  await page.locator('#image-file').setInputFiles({ name: 'synthetic-ui-fixture.png', mimeType: 'image/png', buffer: png });
  await page.waitForFunction(() => document.getElementById('image-source').textContent === 'Your local image · not uploaded');
  assert.equal(await page.locator('#predict').isDisabled(), true, 'An image alone does not start or enable unloaded model inference.');
  assert.ok((await page.locator('#image-note').textContent()).includes('manual placeholders'));
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: path.join(output, 'mobile.png'), fullPage: true });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
  assert.equal(overflow, false, 'Mobile page must not overflow horizontally.');
  assert.deepEqual(errors, [], 'Frontend has no unhandled JavaScript errors.');
  assert.equal(requests.filter(request => !['GET', 'HEAD'].includes(request.method)).length, 0, 'No image uploads or remote inference requests.');
  assert.equal(requests.filter(request => /\.onnx(?:\?|$)|\/runtime\//.test(request.url)).length, 0, 'Neural models and runtimes are never loaded automatically.');
  const proof = { status: 'passed', execution: 'GitHub Actions hosted Chromium', neuralInferenceExecuted: false, patientDataUsed: false, javascriptErrors: errors, imageUploadRequests: 0, automaticModelDownloads: 0, mobileHorizontalOverflow: overflow, originalAngles, changedAngles, requests };
  await writeFile(path.join(output, 'frontend_validation.json'), JSON.stringify(proof, null, 2) + '\n');
  console.log(JSON.stringify(proof, null, 2));
} finally { await browser.close(); }
