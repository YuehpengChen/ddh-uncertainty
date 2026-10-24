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
const started = Date.now();
let server, browser;
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
    const selectors = ['html', 'body', 'main', '.intro', '.workspace', '.controls-panel', '.uncertainty-panel'];
    const surfaces = selectors.map(selector => {
      const style = getComputedStyle(document.querySelector(selector));
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
  assert.equal(await page.locator('#restore-prediction').isVisible(), false);
  assert.equal(await page.locator('#example-credit').isVisible(), true);
  const credit = await page.locator('#example-credit').textContent();
  assert.match(credit, /Apache[\s\S]*2\.0/i, `${item.id}: the displayed credit states the Apache 2.0 license.`);
  const creditLinks = await page.locator('#example-credit a[href]').evaluateAll(nodes => nodes.map(node => ({ text: node.textContent.trim(), href: node.href })));
  assert.ok(creditLinks.some(link => /^https:\/\//.test(link.href)), `${item.id}: the displayed credit links to its public source.`);

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
  const page = await context.newPage();
  page.setDefaultTimeout(30 * 60 * 1000);
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

  const modelLoadStarted = Date.now();
  await page.locator('#load-model').click();
  const modelLoadBusyControls = await verifyBusyControls(page, 'model load');
  console.log('Public-example QA: downloading and verifying the browser model once in GitHub cloud.');
  await page.waitForFunction(() => {
    const node = document.getElementById('model-status');
    return node.classList.contains('ready') || node.classList.contains('error');
  });
  assert.equal(await page.locator('#model-status').evaluate(node => node.classList.contains('ready')), true, await page.locator('#model-status-text').textContent());
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
    const inferenceStarted = Date.now();
    await page.locator('#predict').click();
    record.busyControls = await verifyBusyControls(page, record.id);
    assert.equal(await page.locator('#cancel-predict').isVisible(), true);
    assert.equal(await page.locator('#progress-wrap').isVisible(), true);
    console.log(`Public-example QA: ${record.id}, 50 genuine stochastic passes on the GitHub cloud CPU.`);
    await page.waitForFunction(() => document.getElementById('inference-progress').value >= 1 || document.getElementById('model-status').classList.contains('error'));
    assert.equal(await page.locator('#model-status').evaluate(node => node.classList.contains('error')), false, await page.locator('#model-status-text').textContent());
    record.actualPassProgressObserved = await page.locator('#inference-progress').evaluate(node => node.value);
    assert.ok(record.actualPassProgressObserved >= 1);
    await page.waitForFunction(() => !document.getElementById('uncertainty-results').hidden || document.getElementById('model-status').classList.contains('error'));
    assert.equal(await page.locator('#model-status').evaluate(node => node.classList.contains('error')), false, await page.locator('#model-status-text').textContent());
    record.inferenceAndIntervalSamplingSeconds = (Date.now() - inferenceStarted) / 1000;
    record.finiteAngles = {};
    for (const angle of ['alpha', 'beta']) record.finiteAngles[angle] = finiteAngle(Number.parseFloat(await page.locator(`#${angle}-value`).textContent()), `${record.id}: ${angle} is a finite unsigned angle.`);
    assert.equal(await page.locator('#measurement-mode').textContent(), 'Model mean coordinates');
    assert.match(await page.locator('#uncertainty-badge').textContent(), /50 passes/);
    assert.match(await page.locator('#sampling-note').textContent(), /4,096/);
    record.completedPasses = await page.locator('#inference-progress').evaluate(node => node.value);
    assert.equal(record.completedPasses, 50, `${record.id}: the run completes all 50 passes.`);
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
    secondsIncludingInferenceAndInteractions: (Date.now() - started) / 1000,
    layoutChecks, examples, javascriptErrors: errors, externalRequestsBlocked: blockedRequests,
    requests, browserConsole: logs,
  };
  await writeFile(path.join(output, 'examples_validation.json'), JSON.stringify(proof, null, 2) + '\n');
  console.log(JSON.stringify(proof, null, 2));
} catch (error) {
  await writeFile(path.join(output, 'examples_failure.json'), JSON.stringify({ error: error.message, stack: error.stack, url, examples, layoutChecks, errors, blockedRequests, requests, logs }, null, 2) + '\n');
  throw error;
} finally {
  if (browser) await browser.close();
  if (server) await new Promise(resolve => server.close(resolve));
}
