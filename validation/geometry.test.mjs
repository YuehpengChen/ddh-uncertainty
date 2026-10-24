/** Run with: node --test validation/geometry.test.mjs
 * Synthetic geometry/statistics only. This file never loads a neural model.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as browser from '../geometry.mjs';
import * as reference from './reference.mjs';

const BASE = [[100,100],[100,400],[280,180],[100,180],[280,360]];
const FLAT = BASE.flat();
const close = (actual, expected, tolerance = 1e-10) => {
  assert.ok(Number.isFinite(actual), `Nonfinite actual value: ${actual}`);
  assert.ok(Math.abs(actual - expected) <= tolerance, `Expected ${actual} ≈ ${expected} (tolerance ${tolerance})`);
};
const deepClose = (actual, expected, tolerance = 1e-10) => {
  assert.equal(actual.length, expected.length);
  actual.forEach((v, i) => Array.isArray(v) ? deepClose(v, expected[i], tolerance) : close(v, expected[i], tolerance));
};
const pair = ({alpha,beta}) => [alpha,beta];
const variance = values => {
  const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
  return values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length;
};

test('exact synthetic alpha/beta geometry uses all five points in the stated order', () => {
  deepClose(pair(browser.grafAngles(BASE)), [90,45]);
  deepClose(pair(browser.grafAngles(BASE)), reference.angles(BASE));
});

test('shared translation, rotation and isotropic scale preserve angles', () => {
  const theta = 0.37;
  for (const transform of [
    ([x,y]) => [x + 126, y - 92],
    ([x,y]) => [x * 2.5, y * 2.5],
    ([x,y]) => [Math.cos(theta) * x - Math.sin(theta) * y, Math.sin(theta) * x + Math.cos(theta) * y],
  ]) deepClose(pair(browser.grafAngles(BASE.map(transform))), [90,45]);
});

test('non-square resize requires restoring x by width and y by height', () => {
  const width = 617, height = 563;
  const normalized = BASE.map(([x,y]) => [x / width, y / height]);
  const restored = reference.transformCoordinates(normalized, width, height);
  deepClose(pair(browser.grafAngles(restored)), [90,45]);
  assert.ok(Math.abs(browser.grafAngles(normalized).beta - 45) > 1);
  // Multiplying by W-1/H-1 is not the study's normalization convention.
  const incorrect = normalized.map(([x,y]) => [x * (width - 1), y * (height - 1)]);
  assert.ok(Math.abs(browser.grafAngles(incorrect).beta - 45) > 0.001);
});

test('obtuse angles remain above 90 degrees; P4 movement changes both angles', () => {
  const obtuse = BASE.map(p => [...p]);
  obtuse[2] = [0,80];
  obtuse[4] = [200,80];
  deepClose(pair(browser.grafAngles(obtuse)), [135,135]);
  const moved = BASE.map(p => [...p]);
  moved[3] = [108,187];
  const changed = pair(browser.grafAngles(moved));
  deepClose(changed, reference.angles(moved));
  assert.ok(Math.abs(changed[0] - 90) > 0.1);
  assert.ok(Math.abs(changed[1] - 45) > 0.1);
});

test('degenerate geometry is explicitly unavailable and nonfinite points are rejected', () => {
  const zeroBaseline = BASE.map(p => [...p]);
  zeroBaseline[1] = [...zeroBaseline[0]];
  assert.deepEqual(browser.grafAngles(zeroBaseline), {alpha:null,beta:null});
  const invalid = BASE.map(p => [...p]);
  invalid[3][0] = NaN;
  assert.throws(() => browser.grafAngles(invalid));
  assert.throws(() => browser.grafAngles(BASE.slice(0,4)));
});

test('linear quantiles match the manuscript definition', () => {
  const values = [30,0,20,10];
  close(browser.quantile(values,.05), 1.5);
  close(browser.quantile(values,.95), 28.5);
  close(browser.quantile(values,.5), 15);
  assert.deepEqual(values,[30,0,20,10]);
  close(browser.quantile([7],.05),7);
  assert.throws(() => browser.quantile([],0.5));
  assert.throws(() => browser.quantile([1,2],NaN));
  assert.throws(() => browser.quantile([1,NaN],.5));
});

test('finite-mixture moments divide by T and retain signed coordinate covariance', () => {
  const shift = [2,-3,-4,3,6,-4,1,5,-7,9];
  const means = [-1,1].map(sign => FLAT.map((v,j) => v + sign * shift[j]));
  const variances = [[1,4,2,3,4,5,6,7,8,9],[3,2,4,5,6,7,8,9,10,11]];
  const actual = browser.covarianceSummary(means,variances);
  const expected = reference.mixtureMoments(means,variances);
  deepClose(actual.mean,FLAT);
  deepClose(actual.conditional,expected.within);
  deepClose(actual.epistemic,expected.between);
  deepClose(actual.total,expected.total);
  close(actual.epistemic[0][0],4); // T-1 would incorrectly give eight.
  close(actual.epistemic[0][1],-6);
  for (let j=0;j<10;j++) for (let k=0;k<10;k++) {
    close(actual.diagonal[j][k], j === k ? actual.total[j][j] : 0);
  }
  assert.equal(actual.passes,2);
});

test('one-pass conditional variance is not mislabeled as between-pass variability', () => {
  const actual = browser.covarianceSummary([FLAT],[Array(10).fill(4)]);
  deepClose(actual.epistemic,Array.from({length:10},()=>Array(10).fill(0)));
  deepClose(actual.conditional,Array(10).fill(4));
  assert.throws(() => browser.covarianceSummary([FLAT],[]));
  assert.throws(() => browser.covarianceSummary([FLAT],[Array(10).fill(-1)]));
});

test('analytic five-point Jacobian agrees with independent centered finite differences', () => {
  const J = reference.angleJacobian(BASE);
  const h = 1e-3;
  for (let j=0;j<10;j++) {
    const plus = FLAT.map((v,k)=>v+(j===k?h:0));
    const minus = FLAT.map((v,k)=>v-(j===k?h:0));
    const unpack = flat => Array.from({length:5},(_,i)=>flat.slice(i*2,i*2+2));
    const a = pair(browser.grafAngles(unpack(plus)));
    const b = pair(browser.grafAngles(unpack(minus)));
    a.forEach((v,k)=>close((v-b[k])/(2*h),J[k][j],1e-8));
  }
  assert.ok(Math.hypot(...J[0].slice(6,8))>0);
  assert.ok(Math.hypot(...J[1].slice(6,8))>0);
});

test('shared translation cancels angle uncertainty only when covariance is retained', () => {
  const shift = Array.from({length:5},()=>[2,-3]).flat();
  const means = [-1,1].map(sign => FLAT.map((v,j)=>v+sign*shift[j]));
  const variances = means.map(()=>Array(10).fill(0));
  const actual = browser.sampleAngleDistributions(means,variances,512,browser.seededRandom(914));
  const fullLocal = reference.propagatedAngleCovariance(BASE,actual.total);
  const diagonalLocal = reference.propagatedAngleCovariance(BASE,actual.diagonal);
  close(fullLocal[0][0],0,1e-10);
  close(fullLocal[1][1],0,1e-10);
  assert.ok(diagonalLocal[0][0]>0);
  assert.ok(diagonalLocal[1][1]>0);
  for (const key of ['alpha','beta']) {
    assert.ok(variance(actual.distributions.mixture[key])<1e-18);
    assert.ok(variance(actual.distributions.full[key])<1e-12);
    assert.ok(variance(actual.distributions.diagonal[key])>1e-3);
  }
});

test('one mixture draw uses a single pass for every landmark', () => {
  const shift = [2,-3,-4,3,6,-4,1,5,-7,9];
  const means = [-1,1].map(sign => FLAT.map((v,j)=>v+sign*shift[j]));
  const variances = means.map(()=>Array(10).fill(0));
  const allowed = means.map(flat=>reference.angles(Array.from({length:5},(_,i)=>flat.slice(i*2,i*2+2))));
  const result = browser.sampleAngleDistributions(means,variances,512,browser.seededRandom(819));
  const {alpha,beta,invalid} = result.distributions.mixture;
  assert.equal(invalid,0);
  for (let i=0;i<alpha.length;i++) assert.ok(allowed.some(([a,b])=>Math.abs(a-alpha[i])<1e-10 && Math.abs(b-beta[i])<1e-10));
});

test('matched full/diagonal Gaussians share noise and agree when off-diagonal covariance is zero', () => {
  const result = browser.sampleAngleDistributions([FLAT],[Array(10).fill(3)],512,browser.seededRandom(295));
  assert.equal(result.samples,512);
  for (const key of ['alpha','beta']) {
    deepClose(result.distributions.full[key],result.distributions.diagonal[key]);
    const interval = result.distributions.full.intervals[key];
    close(interval[0],reference.linearQuantile(result.distributions.full[key],.05));
    close(interval[1],reference.linearQuantile(result.distributions.full[key],.95));
  }
});

test('Cholesky reconstructs covariance without silently accepting invalid covariance', () => {
  const matrix = [[9,3,0],[3,4,1],[0,1,2]];
  const lower = browser.cholesky(matrix);
  const reconstruction = lower.map(a=>lower.map(b=>a.reduce((sum,v,i)=>sum+v*b[i],0)));
  deepClose(reconstruction,matrix);
  assert.throws(()=>browser.cholesky([[1,2],[2,1]]));
  assert.throws(()=>browser.cholesky([[0,1],[1,1]]));
});

test('sampling rejects nonpositive or noninteger sample counts', () => {
  for (const count of [0,-1,.5,NaN]) assert.throws(()=>browser.sampleAngleDistributions([FLAT],[Array(10).fill(1)],count));
});

test('P4 contour uses its own complete 2D covariance block', () => {
  const covariance = Array.from({length:10},()=>Array(10).fill(0));
  covariance[6][6]=9; covariance[6][7]=3; covariance[7][6]=3; covariance[7][7]=4;
  const contour = browser.landmarkContour(covariance,3,1,100);
  assert.equal(contour.length,101);
  deepClose(contour[0],contour[100]);
  const determinant = 9*4-3*3;
  for (const [x,y] of contour) close((4*x*x-6*x*y+9*y*y)/determinant,1,1e-9);
});
