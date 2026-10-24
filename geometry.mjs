/** Geometry and uncertainty utilities. No model is evaluated in this module. */
export const LANDMARKS = ['P1', 'P2', 'P3', 'P4', 'P5'];
const DEG = 180 / Math.PI;

export function grafAngles(points) {
  if (points.length !== 5 || points.some(p => p.length !== 2 || p.some(v => !Number.isFinite(v)))) {
    throw new Error('Five finite two-dimensional landmarks are required.');
  }
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
  const angle = (a, b) => {
    if (Math.hypot(...a) <= 1e-6 || Math.hypot(...b) <= 1e-6) return null;
    return DEG * Math.atan2(Math.abs(a[0] * b[1] - a[1] * b[0]), a[0] * b[0] + a[1] * b[1]);
  };
  const u = sub(points[1], points[0]);
  const v = sub(points[2], points[3]);
  const w = sub(points[4], points[3]);
  return { alpha: angle(u, v), beta: angle(w, u) };
}

export function quantile(values, q) {
  if (!values.length || !Number.isFinite(q) || q < 0 || q > 1 || values.some(v => !Number.isFinite(v))) throw new Error('Invalid quantile input.');
  const ordered = [...values].sort((a, b) => a - b);
  const p = (ordered.length - 1) * q;
  const lo = Math.floor(p), hi = Math.ceil(p);
  return ordered[lo] + (ordered[hi] - ordered[lo]) * (p - lo);
}

export function covarianceSummary(means, variances) {
  if (!means.length || means.length !== variances.length) throw new Error('Incomplete prediction passes.');
  const n = means.length, d = 10;
  if (means.some(v => v.length !== d || v.some(x => !Number.isFinite(x))) ||
      variances.some(v => v.length !== d || v.some(x => !Number.isFinite(x) || x < 0))) {
    throw new Error('Model returned invalid coordinates or variances.');
  }
  const mean = Array(d).fill(0);
  for (const row of means) row.forEach((v, i) => { mean[i] += v / n; });
  const epistemic = Array.from({ length: d }, () => Array(d).fill(0));
  const conditional = Array(d).fill(0);
  for (let t = 0; t < n; t++) {
    for (let i = 0; i < d; i++) {
      conditional[i] += variances[t][i] / n;
      for (let j = 0; j < d; j++) epistemic[i][j] += (means[t][i] - mean[i]) * (means[t][j] - mean[j]) / n;
    }
  }
  const total = epistemic.map((row, i) => row.map((x, j) => x + (i === j ? conditional[i] : 0)));
  const diagonal = total.map((row, i) => row.map((x, j) => i === j ? x : 0));
  return { mean, conditional, epistemic, total, diagonal, passes: n };
}

export function cholesky(matrix) {
  const n = matrix.length;
  if (!n || matrix.some(row => row.length !== n || row.some(v => !Number.isFinite(v)))) throw new Error('Invalid covariance matrix.');
  for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) {
    if (Math.abs(matrix[i][j] - matrix[j][i]) > 1e-8 * Math.max(1, Math.abs(matrix[i][j]), Math.abs(matrix[j][i]))) throw new Error('Covariance is not symmetric.');
  }
  const out = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let v = matrix[i][j];
      for (let k = 0; k < j; k++) v -= out[i][k] * out[j][k];
      if (i === j) {
        if (v < -1e-6 * Math.max(1, Math.abs(matrix[i][i]))) throw new Error('Covariance is not positive semidefinite.');
        out[i][j] = Math.sqrt(Math.max(0, v));
      } else {
        if (out[j][j] <= 1e-12 && Math.abs(v) > 1e-8) throw new Error('Covariance is not positive semidefinite.');
        out[i][j] = out[j][j] > 1e-12 ? v / out[j][j] : 0;
      }
    }
  }
  return out;
}

export function seededRandom(seed = 20261024) {
  let state = seed >>> 0;
  return () => {
    state += 0x6D2B79F5;
    let v = Math.imul(state ^ (state >>> 15), 1 | state);
    v ^= v + Math.imul(v ^ (v >>> 7), 61 | v);
    return ((v ^ (v >>> 14)) >>> 0) / 4294967296;
  };
}

export function normalSource(random = Math.random) {
  let cached;
  return () => {
    if (cached !== undefined) { const result = cached; cached = undefined; return result; }
    const radius = Math.sqrt(-2 * Math.log(Math.max(Number.EPSILON, random())));
    const angle = 2 * Math.PI * random();
    cached = radius * Math.sin(angle);
    return radius * Math.cos(angle);
  };
}

export function sampleAngleDistributions(means, variances, count = 4096, random = seededRandom()) {
  if (!Number.isInteger(count) || count <= 0) throw new Error('A positive integer sample count is required.');
  const summary = covarianceSummary(means, variances);
  const lower = cholesky(summary.total);
  const normal = normalSource(random);
  const distributions = Object.fromEntries(['mixture', 'full', 'diagonal'].map(name => [name, { alpha: [], beta: [], invalid: 0 }]));
  const add = (name, flat) => {
    const points = Array.from({ length: 5 }, (_, i) => flat.slice(i * 2, i * 2 + 2));
    const angles = grafAngles(points);
    if (angles.alpha === null || angles.beta === null) { distributions[name].invalid++; return; }
    distributions[name].alpha.push(angles.alpha);
    distributions[name].beta.push(angles.beta);
  };
  for (let s = 0; s < count; s++) {
    const z = Array.from({ length: 10 }, normal);
    const t = Math.min(means.length - 1, Math.floor(random() * means.length));
    add('mixture', z.map((v, i) => means[t][i] + Math.sqrt(variances[t][i]) * v));
    add('full', z.map((_, i) => summary.mean[i] + lower[i].reduce((sum, c, j) => sum + c * z[j], 0)));
    add('diagonal', z.map((v, i) => summary.mean[i] + Math.sqrt(summary.total[i][i]) * v));
  }
  for (const distribution of Object.values(distributions)) {
    distribution.intervals = {};
    for (const key of ['alpha', 'beta']) {
      distribution.intervals[key] = distribution[key].length ? [quantile(distribution[key], .05), quantile(distribution[key], .95)] : null;
    }
  }
  return { ...summary, points: Array.from({ length: 5 }, (_, i) => summary.mean.slice(i * 2, i * 2 + 2)), distributions, samples: count };
}

export function landmarkContour(covariance, index, radius = 2, steps = 48) {
  const i = index * 2, xx = covariance[i][i], xy = covariance[i][i + 1], yy = covariance[i + 1][i + 1];
  const trace = xx + yy, root = Math.hypot(xx - yy, 2 * xy);
  const a = radius * Math.sqrt(Math.max(0, (trace + root) / 2));
  const b = radius * Math.sqrt(Math.max(0, (trace - root) / 2));
  const phi = .5 * Math.atan2(2 * xy, xx - yy);
  return Array.from({ length: steps + 1 }, (_, k) => {
    const angle = 2 * Math.PI * k / steps, x = a * Math.cos(angle), y = b * Math.sin(angle);
    return [x * Math.cos(phi) - y * Math.sin(phi), x * Math.sin(phi) + y * Math.cos(phi)];
  });
}
