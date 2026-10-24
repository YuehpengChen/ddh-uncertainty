import { sampleAngleDistributions } from './geometry.mjs';
import { opaqueRgbaToRgb, preprocessRgbPillow, IMAGENET_MEAN, IMAGENET_STD } from './validation/resize_pillow.mjs';

// This worker evaluates models only after an explicit predict message.
let ort, manifest, backbone, head, baseURL, loaded = false, running = false;
const notify = message => self.postMessage(message);
const modelDownloads = { receivedBytes: 0, totalBytes: 0, totalKnown: false };

function expectedModelBytes(description) {
  return Number.isSafeInteger(description?.bytes) && description.bytes > 0 ? description.bytes : null;
}

function loadProgress(phase, model, message, computable = phase === 'download' && modelDownloads.totalKnown) {
  notify({ type: 'load-progress', phase, model, message, receivedBytes: modelDownloads.receivedBytes, totalBytes: modelDownloads.totalBytes, computable });
}

function modelURL(value) {
  const resolved = new URL(value, baseURL);
  if (!['https:', 'http:'].includes(resolved.protocol)) throw new Error('Unsupported model URL.');
  if (resolved.origin !== new URL(baseURL).origin || resolved.username || resolved.password) throw new Error('Model downloads must remain on this website’s origin.');
  return resolved.href;
}

async function verifiedModel(description, name) {
  if (!description?.url || !/^[a-f0-9]{64}$/i.test(description.sha256 || '')) throw new Error(`${name} has no valid integrity record.`);
  notify({ type: 'status', message: `Downloading ${name} weights…` });
  loadProgress('download', name, `Downloading ${name} weights…`);
  const response = await fetch(modelURL(description.url), { redirect: 'error' });
  if (!response.ok) throw new Error(`${name} download failed (HTTP ${response.status}).`);
  const expectedBytes = expectedModelBytes(description);
  const reader = response.body?.getReader?.();
  let bytes;
  if (reader) {
    // Manifest sizes describe the decoded model bytes. Content-Length can
    // describe compressed transfer bytes, so it is not a progress denominator.
    const target = expectedBytes === null ? null : new Uint8Array(expectedBytes);
    const chunks = target ? null : [];
    const receivedBeforeModel = modelDownloads.receivedBytes;
    let received = 0, lastReportedBytes = 0, lastReportedAt = performance.now();
    const report = force => {
      modelDownloads.receivedBytes = receivedBeforeModel + received;
      const now = performance.now();
      if (force || now - lastReportedAt >= 80 || received - lastReportedBytes >= 256 * 1024) {
        loadProgress('download', name, `Downloading ${name} weights…`);
        lastReportedAt = now; lastReportedBytes = received;
      }
    };
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (!value?.byteLength) continue;
        const nextReceived = received + value.byteLength;
        if (!Number.isSafeInteger(nextReceived) || (expectedBytes !== null && nextReceived > expectedBytes)) throw new Error(`${name} size does not match the published model.`);
        if (target) target.set(value, received);
        else chunks.push(value);
        received = nextReceived;
        report(false);
      }
      report(true);
      if (expectedBytes !== null && received !== expectedBytes) throw new Error(`${name} size does not match the published model.`);
      if (target) bytes = target.buffer;
      else {
        const combined = new Uint8Array(received);
        let offset = 0;
        for (const chunk of chunks) { combined.set(chunk, offset); offset += chunk.byteLength; }
        bytes = combined.buffer;
      }
    } catch (error) {
      try { await reader.cancel(); } catch { /* Preserve the original download error. */ }
      throw error;
    } finally { reader.releaseLock(); }
  } else {
    loadProgress('download', name, `Downloading ${name} weights; this browser does not expose byte progress…`, false);
    bytes = await response.arrayBuffer();
    modelDownloads.receivedBytes += bytes.byteLength;
    loadProgress('download', name, `Downloaded ${name} weights.`);
  }
  if (description.bytes && bytes.byteLength !== description.bytes) throw new Error(`${name} size does not match the published model.`);
  notify({ type: 'status', message: `Verifying ${name} SHA-256…` });
  loadProgress('verify', name, `Verifying ${name} SHA-256…`);
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  const hex = Array.from(new Uint8Array(digest), v => v.toString(16).padStart(2, '0')).join('');
  if (hex.toLowerCase() !== description.sha256.toLowerCase()) throw new Error(`${name} integrity verification failed. Prediction is disabled.`);
  return bytes;
}

function validateManifest(value) {
  if (value.schemaVersion !== 1 || value.status !== 'ready') throw new Error('This model has not passed the published cloud checks.');
  if (value.dropout?.stochastic !== true || value.dropout?.mechanism !== 'explicit_masks') throw new Error('The stochastic inference contract is unsupported. Native ONNX Dropout is not accepted.');
  if (value.input?.layout !== 'NCHW' || String(value.input.shape) !== '1,3,448,448') throw new Error('Unexpected model image input.');
  if (value.outputs?.mean?.coordinates !== 'normalized' || value.outputs?.variance?.parameterization !== 'variance') throw new Error('Unexpected coordinate or variance parameterization.');
  if (!value.features?.length || !value.models?.backbone || !value.models?.head) throw new Error('The model bundle is incomplete.');
  if (value.runtime?.provider !== 'wasm') throw new Error('This deployment does not specify the supported browser CPU runtime.');
  const normalization = value.input.normalization;
  if (normalization?.scale !== 255 || String(normalization.mean) !== String(IMAGENET_MEAN) || String(normalization.std) !== String(IMAGENET_STD)) throw new Error('The image normalization differs from the validated preprocessing pipeline.');
  const masks = value.dropout.inputs?.filter(input => input.name !== 'stochastic');
  if (!masks?.length || masks.length !== value.dropout.sites) throw new Error('The explicit dropout mask specification is incomplete.');
  for (const mask of masks) {
    if (mask.dtype !== 'uint8' || !Array.isArray(mask.shape) || !mask.shape.length || mask.shape.some(n => !Number.isSafeInteger(n) || n <= 0) || mask.shape.reduce((a, b) => a * b, 1) > 5000000 || !Number.isFinite(mask.probability) || mask.probability < 0 || mask.probability >= 1) throw new Error('An explicit dropout mask has invalid dimensions or probability.');
  }
}

async function initialize(manifestURL) {
  const manifestLocation = new URL(manifestURL);
  if (manifestLocation.origin !== self.location.origin) throw new Error('The model manifest must remain on this website’s origin.');
  baseURL = new URL('../', manifestLocation).href;
  const response = await fetch(manifestLocation, { cache: 'no-store', redirect: 'error' });
  if (!response.ok) throw new Error('The model manifest could not be loaded.');
  manifest = await response.json(); validateManifest(manifest);
  const runtimeURL = new URL(manifest.runtime.module, baseURL);
  if (runtimeURL.origin !== new URL(baseURL).origin) throw new Error('The runtime must be self-hosted with this application.');
  const modelSizes = [manifest.models.backbone, manifest.models.head].map(expectedModelBytes);
  modelDownloads.receivedBytes = 0;
  modelDownloads.totalBytes = modelSizes.reduce((sum, size) => sum + (size || 0), 0);
  modelDownloads.totalKnown = modelSizes.every(size => size !== null) && Number.isSafeInteger(modelDownloads.totalBytes);
  loadProgress('download', 'runtime', 'Loading browser CPU runtime before model downloads; byte counts cover model weights only…', false);
  const runtime = await import(runtimeURL.href); ort = runtime.default || runtime;
  ort.env.wasm.wasmPaths = new URL('runtime/', baseURL).href;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  const options = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
  const backboneBytes = await verifiedModel(manifest.models.backbone, 'backbone');
  notify({ type: 'status', message: 'Preparing browser CPU feature extractor…' });
  loadProgress('prepare', 'backbone', 'Preparing browser CPU feature extractor…');
  backbone = await ort.InferenceSession.create(backboneBytes, options);
  const headBytes = await verifiedModel(manifest.models.head, 'landmark decoder');
  notify({ type: 'status', message: 'Preparing browser CPU landmark decoder…' });
  loadProgress('prepare', 'landmark decoder', 'Preparing browser CPU landmark decoder…');
  head = await ort.InferenceSession.create(headBytes, options);
  if (!backbone.inputNames.includes(manifest.input.name) || !head.inputNames.includes('stochastic')) throw new Error('Loaded model input names do not match the manifest.');
  for (const mask of manifest.dropout.inputs.filter(input => input.name !== 'stochastic')) if (!head.inputNames.includes(mask.name)) throw new Error(`The decoder is missing dropout mask input ${mask.name}.`);
  loaded = true;
  loadProgress('ready', 'landmark decoder', 'Model weights verified and browser CPU sessions ready.');
  notify({ type: 'loaded', backend: 'wasm', stochastic: true });
}

function preprocess(image) {
  if (image.width <= 0 || image.height <= 0 || image.width * image.height > 16000000 || image.data.length !== image.width * image.height * 4) throw new Error('The decoded image exceeds the supported 16-megapixel limit.');
  const rgb = opaqueRgbaToRgb(image.data, image.width, image.height);
  const data = preprocessRgbPillow(rgb, image.width, image.height, 448);
  return new ort.Tensor('float32', data, [1, 3, 448, 448]);
}

function dropoutFeeds() {
  const feeds = { stochastic: new ort.Tensor('bool', new Uint8Array([1]), []) };
  for (const description of manifest.dropout.inputs) {
    if (description.name === 'stochastic') continue;
    const count = description.shape.reduce((a, b) => a * b, 1), mask = new Uint8Array(count);
    // One independent Bernoulli draw per exported mask entry, freshly generated
    // for every site and pass. Channel sharing is encoded by each mask's shape.
    for (let i = 0; i < count; i++) mask[i] = Math.random() >= description.probability ? 1 : 0;
    feeds[description.name] = new ort.Tensor('uint8', mask, description.shape);
  }
  return feeds;
}

function dispose(tensorMap) {
  for (const tensor of Object.values(tensorMap || {})) if (typeof tensor?.dispose === 'function') tensor.dispose();
}

async function predict(message) {
  if (!loaded) throw new Error('Load and verify the model before prediction.');
  if (running) throw new Error('A prediction is already running.');
  const { id, image, originalWidth: width, originalHeight: height, passes } = message;
  if (![10, 20, 50].includes(passes)) throw new Error('Choose 10, 20 or 50 stochastic passes.');
  if (!(width > 0 && height > 0 && width <= 8192 && height <= 8192)) throw new Error('Invalid original image dimensions.');
  if (image.width !== width || image.height !== height) throw new Error('The image must retain its original dimensions before validated resampling.');
  running = true;
  let features, input;
  try {
    notify({ type: 'progress', id, phase: 'preprocess', completed: 0, total: 1, computable: false, message: 'Preparing the image with validated resampling…' });
    input = preprocess(image);
    notify({ type: 'progress', id, phase: 'features', completed: 0, total: 1, computable: false, message: 'Extracting image features once…' });
    features = await backbone.run({ [manifest.input.name]: input });
    notify({ type: 'progress', id, phase: 'features', completed: 1, total: 1, computable: true, message: 'Image features extracted.' });
    const means = [], variances = [];
    notify({ type: 'progress', id, phase: 'predict', completed: 0, total: passes, computable: true, message: `0 / ${passes} stochastic passes completed` });
    for (let pass = 0; pass < passes; pass++) {
      const masks = dropoutFeeds();
      const feeds = { ...masks };
      for (const feature of manifest.features) {
        if (!features[feature.name]) throw new Error(`Missing image feature: ${feature.name}.`);
        feeds[feature.name] = features[feature.name];
      }
      let output;
      try {
        output = await head.run(feeds);
        const mean = output[manifest.outputs.mean.name]?.data;
        const variance = output[manifest.outputs.variance.name]?.data;
        if (mean?.length !== 10 || variance?.length !== 10) throw new Error('The decoder did not return five coordinate pairs and variances.');
        means.push(Array.from(mean, (v, i) => v * (i % 2 ? height : width)));
        variances.push(Array.from(variance, (v, i) => v * (i % 2 ? height * height : width * width)));
      } finally { dispose(output); dispose(masks); }
      notify({ type: 'progress', id, phase: 'predict', completed: pass + 1, total: passes, computable: true, message: `${pass + 1} / ${passes} stochastic passes completed` });
      // Allow message dispatch and visible progress; no simulated timing.
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    const variation = means.some(row => row.some((v, i) => Math.abs(v - means[0][i]) > 1e-7));
    if (!variation) throw new Error('Repeated predictions were identical. Stochastic uncertainty is unavailable; no interval was substituted.');
    notify({ type: 'progress', id, phase: 'intervals', completed: 0, total: 4096, computable: false, message: 'Sampling 4,096 joint configurations and angle intervals…' });
    const result = sampleAngleDistributions(means, variances, 4096);
    requireCompleteAngleSamples(result);
    notify({ type: 'progress', id, phase: 'intervals', completed: result.samples, total: 4096, computable: true, message: '4,096 joint configurations and angle intervals completed.' });
    result.backend = 'wasm';
    notify({ type: 'result', id, result });
  } finally { input?.dispose?.(); dispose(features); running = false; }
}

function requireCompleteAngleSamples(result) {
  if (!Number.isSafeInteger(result.samples) || result.samples <= 0) throw new Error('Angle intervals unavailable: no requested sample set.');
  const counts = Object.entries(result.distributions).map(([name, distribution]) => ({
    name, valid: Math.min(distribution.alpha.length, distribution.beta.length), degenerate: distribution.invalid,
  }));
  if (counts.some(row => row.degenerate !== 0 || row.valid !== result.samples)) {
    throw new Error(`Angle intervals unavailable: ${counts.map(row => `${row.name}: ${row.valid} valid, ${row.degenerate} degenerate out of ${result.samples} draws`).join('; ')}. No intervals are reported from an incomplete sample set.`);
  }
}

self.addEventListener('message', async event => {
  const message = event.data;
  try {
    if (message.type === 'init') await initialize(message.manifestURL);
    else if (message.type === 'predict') await predict(message);
    else throw new Error('Unknown worker request.');
  } catch (error) { notify({ type: 'error', id: message.id, phase: message.type === 'init' ? 'load' : 'predict', message: error.message || String(error) }); }
});
