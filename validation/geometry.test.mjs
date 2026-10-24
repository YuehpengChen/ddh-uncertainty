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

// These fixtures have known eigenvectors and eigenvalues. The checks project
// the returned curve onto those vectors rather than reproducing the renderer's
// eigenvalue/angle calculation. The inverse quadratic form independently checks
// that every contour point has the requested Mahalanobis radius.
const blockCovariance = (index, [[xx,xy],[yx,yy]]) => {
  const matrix = Array.from({length:10},()=>Array(10).fill(0));
  const i = index * 2;
  matrix[i][i]=xx; matrix[i][i+1]=xy;
  matrix[i+1][i]=yx; matrix[i+1][i+1]=yy;
  return matrix;
};

const checkKnownEllipse = (covariance, index, {block,major,minor,majorVariance,minorVariance,radius=2}) => {
  const contour = browser.landmarkContour(covariance,index,radius,120);
  const points = contour.slice(0,-1); // Do not count the closing point twice.
  deepClose(contour[0],contour.at(-1),1e-10);
  const projection = axis => points.map(([x,y])=>x*axis[0]+y*axis[1]);
  const majorProjection = projection(major), minorProjection = projection(minor);
  const majorRadius = Math.max(...majorProjection.map(Math.abs));
  const minorRadius = Math.max(...minorProjection.map(Math.abs));
  close(majorRadius,radius*Math.sqrt(majorVariance),1e-10);
  close(minorRadius,radius*Math.sqrt(minorVariance),1e-10);
  close(majorRadius/minorRadius,Math.sqrt(majorVariance/minorVariance),1e-10);
  close(majorProjection.reduce((s,v)=>s+v,0)/points.length,0,1e-10);
  close(minorProjection.reduce((s,v)=>s+v,0)/points.length,0,1e-10);
  close(majorProjection.reduce((s,v,i)=>s+v*minorProjection[i],0)/points.length,0,1e-10);
  const [[xx,xy],[,yy]]=block, determinant=xx*yy-xy*xy;
  for (const [x,y] of points) {
    close((yy*x*x-2*xy*x*y+xx*y*y)/determinant,radius*radius,1e-9);
  }
  return contour;
};

test('positive within-point covariance produces a positively tilted ellipse', () => {
  const block = [[5,3],[3,5]], s=Math.SQRT1_2;
  const covariance=blockCovariance(3,block);
  // Other landmarks have different spreads; P4 must use coordinates 6 and 7.
  covariance[4][4]=400; covariance[5][5]=900;
  covariance[8][8]=1600; covariance[9][9]=2500;
  checkKnownEllipse(covariance,3,{block,major:[s,s],minor:[-s,s],majorVariance:8,minorVariance:2});
});

test('negative within-point covariance reverses the ellipse tilt', () => {
  const block = [[5,-3],[-3,5]], s=Math.SQRT1_2;
  checkKnownEllipse(blockCovariance(3,block),3,{block,major:[s,-s],minor:[s,s],majorVariance:8,minorVariance:2});
});

test('zero within-point covariance preserves the correct axis-aligned spread', () => {
  const block = [[9,0],[0,4]];
  checkKnownEllipse(blockCovariance(0,block),0,{block,major:[1,0],minor:[0,1],majorVariance:9,minorVariance:4});
  const verticalBlock = [[4,0],[0,9]];
  checkKnownEllipse(blockCovariance(4,verticalBlock),4,{block:verticalBlock,major:[0,1],minor:[1,0],majorVariance:9,minorVariance:4});
});

test('isotropic covariance gives a circle without assigning a meaningful orientation', () => {
  const contour=browser.landmarkContour(blockCovariance(2,[[4,0],[0,4]]),2,2,120);
  for (const [x,y] of contour) close(Math.hypot(x,y),4,1e-10);
  // A circle is invariant under rotation; no particular angle is required.
  const theta=.63;
  for (const [x,y] of contour) {
    const xr=x*Math.cos(theta)-y*Math.sin(theta), yr=x*Math.sin(theta)+y*Math.cos(theta);
    close(xr*xr/4+yr*yr/4,4,1e-10);
  }
});

test('diagonal conditional variance reduces correlation without erasing signed xy dependence', () => {
  const offsets=Array(10).fill(0); offsets[6]=2; offsets[7]=1;
  const means=[-1,1].map(sign=>FLAT.map((v,j)=>v+sign*offsets[j]));
  const conditional=Array(10).fill(0); conditional[6]=3; conditional[7]=3;
  const result=browser.covarianceSummary(means,means.map(()=>conditional));
  close(result.epistemic[6][6],4); close(result.epistemic[7][7],1);
  close(result.epistemic[6][7],2); close(result.total[6][7],2);
  close(result.total[6][6],7); close(result.total[7][7],4);
  const epistemicCorrelation=result.epistemic[6][7]/Math.sqrt(result.epistemic[6][6]*result.epistemic[7][7]);
  const totalCorrelation=result.total[6][7]/Math.sqrt(result.total[6][6]*result.total[7][7]);
  close(epistemicCorrelation,1); close(totalCorrelation,1/Math.sqrt(7));
  assert.ok(totalCorrelation>0 && totalCorrelation<epistemicCorrelation);
  close(result.diagonal[6][7],0); // This diagnostic is intentionally different.
  const s=Math.sqrt(5), block=[[7,2],[2,4]];
  checkKnownEllipse(result.total,3,{block,major:[2/s,1/s],minor:[-1/s,2/s],majorVariance:8,minorVariance:3});
});

test('non-square original-pixel conversion scales variances quadratically and xy by width times height', () => {
  const width=800, height=200;
  const normalizedCenter=Array.from({length:5},()=>[.5,.5]).flat();
  const normalizedOffset=Array(10).fill(0); normalizedOffset[6]=.01; normalizedOffset[7]=.02;
  const normalizedMeans=[-1,1].map(sign=>normalizedCenter.map((v,j)=>v+sign*normalizedOffset[j]));
  const normalizedConditional=Array(10).fill(0); normalizedConditional[6]=.0001; normalizedConditional[7]=.0004;
  const normalized=browser.covarianceSummary(normalizedMeans,normalizedMeans.map(()=>normalizedConditional));
  const scales=Array.from({length:5},()=>[width,height]).flat();
  const pixelMeans=normalizedMeans.map(row=>row.map((v,j)=>v*scales[j]));
  const pixelVariances=normalizedMeans.map(()=>normalizedConditional.map((v,j)=>v*scales[j]**2));
  const pixels=browser.covarianceSummary(pixelMeans,pixelVariances);
  close(pixels.mean[6],400); close(pixels.mean[7],100);
  close(pixels.conditional[6],64); close(pixels.conditional[7],16);
  close(pixels.total[6][6],128); close(pixels.total[7][7],32);
  close(pixels.total[6][7],32); close(pixels.total[7][6],32);
  for (let j=0;j<10;j++) for (let k=0;k<10;k++) {
    close(pixels.total[j][k],normalized.total[j][k]*scales[j]*scales[k],1e-9);
  }
});
