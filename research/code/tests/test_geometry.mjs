/** Synthetic numerical checks only; no images, annotations or model weights. */
import assert from 'node:assert/strict';
import test from 'node:test';
import {
  grafAngles, quantile, covarianceSummary, cholesky, seededRandom,
  sampleAngleDistributions,
} from '../js/geometry.mjs';
import {
  angles, angleJacobian, mixtureMoments, propagatedAngleCovariance,
  transformCoordinates, linearQuantile, sampleMixture,
  seededRandom as referenceRandom,
} from '../js/reference.mjs';

const known = [[0, 0], [10, 0], [25, 10 + 5 * Math.sqrt(3)], [20, 10], [20 + 5 * Math.sqrt(3), 5]];
const near = (a, b, tolerance = 1e-10) => assert.ok(Math.abs(a - b) <= tolerance, `${a} differs from ${b}`);
function nearArray(actual, expected, tolerance = 1e-10) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; i++) {
    if (Array.isArray(actual[i])) nearArray(actual[i], expected[i], tolerance);
    else near(actual[i], expected[i], tolerance);
  }
}
const translated = (points, [dx, dy]) => points.map(([x, y]) => [x + dx, y + dy]);
const multiplyLLT = lower => lower.map(a => lower.map(b => a.reduce((s, v, k) => s + v * b[k], 0)));
const flat = known.flat();
const delta = Array.from({length: 10}, (_, j) => j % 2 === 0 ? 1 : 2);
const componentMeans = [-1, 1].map(sign => flat.map((v, j) => v + sign * delta[j]));
const componentVariances = [1, 2].map(scale => Array.from({length: 10}, (_, j) => scale * (j + 1)));

test('Graf alpha/beta match known geometry in degrees', () => {
  const found = grafAngles(known);
  near(found.alpha, 60);
  near(found.beta, 30);
  nearArray(angles(known), [60, 30]);
});

test('common translation leaves both angles unchanged', () => {
  const found = grafAngles(translated(known, [123.4, -987.65]));
  near(found.alpha, 60);
  near(found.beta, 30);
  nearArray(angles(translated(known, [-4, 7])), angles(known));
});

test('analytical Jacobian agrees with central differences in degrees per pixel', () => {
  const J = angleJacobian(known), h = 1e-5;
  for (let j = 0; j < 10; j++) {
    const plus = known.map(p => [...p]), minus = known.map(p => [...p]);
    plus[Math.floor(j / 2)][j % 2] += h;
    minus[Math.floor(j / 2)][j % 2] -= h;
    const ap = angles(plus), am = angles(minus);
    for (let a = 0; a < 2; a++) near(J[a][j], (ap[a] - am[a]) / (2 * h), 1e-7);
  }
});

test('Jacobian cancels common x/y translations', () => {
  for (const row of angleJacobian(known)) {
    near(row.filter((_, j) => j % 2 === 0).reduce((s, v) => s + v, 0), 0);
    near(row.filter((_, j) => j % 2 === 1).reduce((s, v) => s + v, 0), 0);
  }
});

test('finite-mixture mean and covariance use denominator T', () => {
  const found = covarianceSummary(componentMeans, componentVariances);
  nearArray(found.mean, flat);
  nearArray(found.conditional, Array.from({length: 10}, (_, j) => 1.5 * (j + 1)));
  const expectedBetween = delta.map(a => delta.map(b => a * b));
  nearArray(found.epistemic, expectedBetween);
  // Two components at +/-delta have covariance delta*delta^T, not 2*delta*delta^T.
  near(found.epistemic[0][0], 1);
  near(found.epistemic[1][1], 4);
  assert.equal(found.passes, 2);
});

test('mixture moments agree with the separately implemented reference', () => {
  const found = covarianceSummary(componentMeans, componentVariances);
  const reference = mixtureMoments(componentMeans, componentVariances);
  nearArray(found.mean, reference.mean);
  nearArray(found.conditional, reference.within);
  nearArray(found.epistemic, reference.between);
  nearArray(found.total, reference.total);
});

test('diagonal control preserves marginal variances and removes cross terms', () => {
  const found = covarianceSummary(componentMeans, componentVariances);
  for (let j = 0; j < 10; j++) for (let k = 0; k < 10; k++) {
    if (j === k) near(found.diagonal[j][k], found.total[j][k]);
    else {
      near(found.diagonal[j][k], 0);
      near(found.total[j][k], found.epistemic[j][k]);
    }
  }
});

test('one retained component has no between-pass covariance', () => {
  const found = covarianceSummary([flat], [componentVariances[0]]);
  nearArray(found.mean, flat);
  nearArray(found.epistemic, Array.from({length: 10}, () => Array(10).fill(0)));
  nearArray(found.total, componentVariances[0].map((v, j) => Array.from({length: 10}, (_, k) => j === k ? v : 0)));
});

test('local propagation of common translation covariance is zero', () => {
  const covariance = delta.map(a => delta.map(b => a * b));
  nearArray(propagatedAngleCovariance(known, covariance), [[0, 0], [0, 0]], 1e-10);
});

test('Cholesky reconstructs positive-definite and semidefinite covariance', () => {
  for (const covariance of [[[4, 2], [2, 3]], [[1, 1], [1, 1]], [[0, 0], [0, 0]]]) {
    nearArray(multiplyLLT(cholesky(covariance)), covariance);
  }
});

test('invalid covariance and prediction inputs are rejected', () => {
  assert.throws(() => cholesky([[1, 2], [2, 1]]), /positive semidefinite/);
  assert.throws(() => cholesky([[1, 0.1], [0, 1]]), /symmetric/);
  assert.throws(() => cholesky([[NaN]]), /Invalid covariance/);
  assert.throws(() => covarianceSummary([], []), /Incomplete/);
  assert.throws(() => covarianceSummary([flat], [Array(10).fill(-1)]), /invalid coordinates or variances/);
  assert.throws(() => mixtureMoments([flat], [Array(9).fill(1)]), /ten-coordinate/);
  assert.throws(() => grafAngles([[0, 0]]), /Five finite/);
});

test('short segments and unsigned-angle branch boundaries have explicit behavior', () => {
  const zero = known.map(p => [...p]); zero[1] = [...zero[0]];
  assert.deepEqual(grafAngles(zero), {alpha: null, beta: null});
  assert.throws(() => angles(zero), /too short/);
  assert.throws(() => angleJacobian(zero), /too short/);
  const branch = known.map(p => [...p]); branch[2] = [branch[3][0] + 10, branch[3][1]];
  near(grafAngles(branch).alpha, 0);
  assert.throws(() => angleJacobian(branch), /branch boundary/);
});

test('linear quantiles match known interpolation and independent reference', () => {
  const sample = [30, 0, 20, 10];
  for (const [q, expected] of [[0, 0], [0.25, 7.5], [0.5, 15], [1, 30]]) {
    near(quantile(sample, q), expected);
    near(linearQuantile(sample, q), expected);
  }
  assert.throws(() => quantile([], 0.5), /Invalid/);
  assert.throws(() => quantile(sample, 1.1), /Invalid/);
  assert.throws(() => quantile([NaN], 0.5), /Invalid/);
});

test('normalized coordinates transform each axis using its image dimension', () => {
  const normalized = known.map(([x, y]) => [x / 617, y / 563]);
  nearArray(transformCoordinates(normalized, 617, 563), known);
  assert.throws(() => transformCoordinates(normalized, 0, 563), /Invalid image dimensions/);
});

test('deterministic sampling preserves common-translation angles', () => {
  const base = [[0, 0], [40, 0], [30, 40], [0, 10], [20, 0]];
  const means = [-1, 1].map(s => translated(base, [s, 2 * s]).flat());
  const variances = means.map(() => Array(10).fill(0));
  const a = sampleAngleDistributions(means, variances, 1024, seededRandom(42));
  const b = sampleAngleDistributions(means, variances, 1024, seededRandom(42));
  assert.deepEqual(a, b);
  const expected = grafAngles(base);
  for (const key of ['mixture', 'full']) {
    assert.equal(a.distributions[key].invalid, 0);
    for (const name of ['alpha', 'beta']) {
      for (const value of a.distributions[key][name]) near(value, expected[name]);
    }
  }
  for (const distribution of Object.values(a.distributions)) {
    assert.equal(distribution.alpha.length, 1024);
    for (const name of ['alpha', 'beta']) {
      const [lo, hi] = distribution.intervals[name];
      assert.ok(Number.isFinite(lo) && Number.isFinite(hi) && 0 <= lo && lo <= hi && hi <= 180);
    }
  }
  assert.ok(a.distributions.diagonal.intervals.alpha[1] - a.distributions.diagonal.intervals.alpha[0] > 0.01);
  assert.throws(() => sampleAngleDistributions(means, variances, 0), /positive integer/);
});

test('reference mixture sampler chooses one common component for all coordinates', () => {
  const means = [Array(10).fill(0), Array(10).fill(10)];
  const variances = means.map(() => Array(10).fill(0));
  const first = sampleMixture(means, variances, 128, referenceRandom(9));
  const second = sampleMixture(means, variances, 128, referenceRandom(9));
  assert.deepEqual(first, second);
  assert.equal(new Set(first.map(row => row[0])).size, 2);
  for (const row of first) assert.ok(row.every(v => v === row[0]) && [0, 10].includes(row[0]));
});
