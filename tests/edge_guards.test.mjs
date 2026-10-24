/** Pure guard tests: no model/runtime import, image inference or browser launch. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';

const worker = await readFile(new URL('../inference-worker.mjs', import.meta.url), 'utf8');
function isolatedFunction(name, context = {}) {
  const match = worker.match(new RegExp(`function ${name}\\([^)]*\\) \\{[\\s\\S]*?\\n\\}`));
  assert.ok(match, `${name} must remain present for verification.`);
  return vm.runInNewContext(`(${match[0]})`, { URL, ...context });
}
const modelURL = isolatedFunction('modelURL', { baseURL: 'https://example.test/ddh/' });
const requireComplete = isolatedFunction('requireCompleteAngleSamples');

test('model downloads are restricted to the website origin', () => {
  assert.equal(modelURL('models/backbone.onnx'), 'https://example.test/ddh/models/backbone.onnx');
  assert.equal(modelURL('https://example.test/ddh/models/backbone.onnx'), 'https://example.test/ddh/models/backbone.onnx');
  for (const value of ['https://other.test/model.onnx', '//other.test/model.onnx', 'http://example.test/model.onnx', 'file:///model.onnx', 'javascript:alert(1)', 'https://name:pass@example.test/model.onnx']) assert.throws(() => modelURL(value));
  assert.match(worker, /fetch\(modelURL\(description\.url\), \{ redirect: 'error' \}\)/, 'Cross-origin redirects must not be followed.');
});

const result = (valid, invalid = 0) => ({ samples: 4, distributions: Object.fromEntries(['mixture', 'full', 'diagonal'].map(name => [name, { alpha: Array(valid).fill(60), beta: Array(valid).fill(50), invalid }])) });

test('complete angle samples are accepted', () => assert.doesNotThrow(() => requireComplete(result(4))));
test('degenerate draws cause an explicit interval failure with counts', () => {
  assert.throws(() => requireComplete(result(3, 1)), /mixture: 3 valid, 1 degenerate out of 4 draws/);
  assert.throws(() => requireComplete(result(0, 4)), /0 valid, 4 degenerate/);
});
test('unexplained missing or zero-request samples are rejected', () => {
  assert.throws(() => requireComplete(result(3)), /incomplete sample set/);
  assert.throws(() => requireComplete({ ...result(0), samples: 0 }), /no requested sample set/);
});
