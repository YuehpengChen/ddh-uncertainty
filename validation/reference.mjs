/** Independent numerical references. These routines never load a neural model. */
const DEG = 180 / Math.PI;
export const MIN_SEGMENT_PX = 1e-6;

export function flatCoordinates(points) {
  if (!Array.isArray(points) || points.length !== 5 || points.some(p => !Array.isArray(p) || p.length !== 2)) {
    throw new RangeError('Exactly five [x,y] landmarks are required.');
  }
  const flat = points.flat();
  if (flat.some(v => !Number.isFinite(v))) throw new RangeError('Coordinates must be finite.');
  return flat;
}

export function unsignedAngle(a, b) {
  if (Math.hypot(...a) <= MIN_SEGMENT_PX || Math.hypot(...b) <= MIN_SEGMENT_PX) {
    throw new RangeError('An angle-defining segment is too short.');
  }
  return DEG * Math.atan2(Math.abs(a[0] * b[1] - a[1] * b[0]), a[0] * b[0] + a[1] * b[1]);
}

export function angles(points) {
  flatCoordinates(points);
  const difference = (i, j) => points[i].map((v, k) => v - points[j][k]);
  const u = difference(1, 0), v = difference(2, 3), w = difference(4, 3);
  return [unsignedAngle(u, v), unsignedAngle(w, u)];
}

export function angleJacobian(points) {
  flatCoordinates(points);
  const difference = (i, j) => points[i].map((v, k) => v - points[j][k]);
  const u = difference(1, 0), v = difference(2, 3), w = difference(4, 3);
  angles(points);
  const derivative = a => [-a[1] / (a[0] ** 2 + a[1] ** 2), a[0] / (a[0] ** 2 + a[1] ** 2)];
  const cross = (a, b) => a[0] * b[1] - a[1] * b[0];
  if (Math.abs(cross(u, v)) < 1e-10 || Math.abs(cross(w, u)) < 1e-10) {
    throw new RangeError('Unsigned-angle derivative is undefined at a branch boundary.');
  }
  const gu = derivative(u), gv = derivative(v), gw = derivative(w);
  const sa = Math.sign(cross(u, v)), sb = Math.sign(cross(w, u));
  const neg = a => a.map(x => -x);
  return [
    [...gu, ...neg(gu), ...gv, ...neg(gv), 0, 0].map(x => x * DEG * sa),
    [...neg(gu), ...gu, 0, 0, ...gw, ...neg(gw)].map(x => x * DEG * sb),
  ];
}

export function transformCoordinates(normalized, width, height) {
  if (![width, height].every(v => Number.isFinite(v) && v > 0)) throw new RangeError('Invalid image dimensions.');
  flatCoordinates(normalized);
  return normalized.map(([x, y]) => [x * width, y * height]);
}

export function mixtureMoments(means, variances) {
  const T = means.length;
  if (!T || T !== variances.length || means.some(v => v.length !== 10) || variances.some(v => v.length !== 10)) {
    throw new RangeError('Matching nonempty arrays of ten-coordinate components are required.');
  }
  if (means.flat().some(v => !Number.isFinite(v)) || variances.flat().some(v => !Number.isFinite(v) || v < 0)) {
    throw new RangeError('Finite means and nonnegative variances are required.');
  }
  const mean = Array.from({length: 10}, (_, j) => means.reduce((sum, m) => sum + m[j], 0) / T);
  const within = Array.from({length: 10}, (_, j) => variances.reduce((sum, d) => sum + d[j], 0) / T);
  const between = Array.from({length: 10}, (_, j) => Array.from({length: 10}, (_, k) =>
    means.reduce((sum, m) => sum + (m[j] - mean[j]) * (m[k] - mean[k]), 0) / T));
  const total = between.map((row, j) => row.map((value, k) => value + (j === k ? within[j] : 0)));
  return {mean, within, between, total};
}

export function propagatedAngleCovariance(points, covariance) {
  const J = angleJacobian(points);
  return J.map(a => J.map(b => a.reduce((sum, aj, j) =>
    sum + b.reduce((inner, bk, k) => inner + aj * covariance[j][k] * bk, 0), 0)));
}

export function linearQuantile(values, probability) {
  if (!values.length || !Number.isFinite(probability) || probability < 0 || probability > 1 || values.some(v => !Number.isFinite(v))) {
    throw new RangeError('Finite samples and a probability in [0,1] are required.');
  }
  const sorted = [...values].sort((a, b) => a - b);
  const index = (sorted.length - 1) * probability;
  const lo = Math.floor(index), hi = Math.ceil(index);
  return sorted[lo] + (index - lo) * (sorted[hi] - sorted[lo]);
}

export function sampleMixture(means, variances, count, random = Math.random) {
  mixtureMoments(means, variances);
  if (!Number.isInteger(count) || count < 1) throw new RangeError('A positive sample count is required.');
  const normal = () => {
    const first = Math.max(Number.MIN_VALUE, random());
    return Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * random());
  };
  return Array.from({length: count}, () => {
    const t = Math.min(means.length - 1, Math.floor(random() * means.length));
    return means[t].map((mean, j) => mean + Math.sqrt(variances[t][j]) * normal());
  });
}

export function seededRandom(value = 1927) {
  let state = value >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return (state + 0.5) / 4294967296;
  };
}
