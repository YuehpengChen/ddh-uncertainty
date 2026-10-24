/** Validate RGB8 resize and float32 normalization on GitHub-hosted fixtures.
 * node validation/verify_resize_pillow.mjs validation/fixtures/resize_pillow.json validation/resize_pillow_validation.json
 * No model is imported or evaluated.
 */
import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {resizeRgbBilinearPillow, normalizeImageNetPlanar, opaqueRgbaToRgb} from './resize_pillow.mjs';

if (process.env.GITHUB_ACTIONS !== 'true' || !process.env.GITHUB_RUN_ID) {
  throw new Error('Refused: Pillow compatibility proof must run on GitHub Actions cloud.');
}
const fixturePath = process.argv[2] ?? 'validation/fixtures/resize_pillow.json';
const reportPath = process.argv[3] ?? 'validation/resize_pillow_validation.json';
const rawFixture = await readFile(fixturePath);
const fixture = JSON.parse(rawFixture);
assert.equal(fixture.synthetic,true);
assert.equal(fixture.patientDataUsed,false);
assert.equal(fixture.pillowVersion,'11.3.0');
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const results = [];
for (const entry of fixture.cases) {
  const source = new Uint8Array(Buffer.from(entry.sourceRGBBase64,'base64'));
  const expected = new Uint8Array(Buffer.from(entry.expectedRGBBase64,'base64'));
  assert.equal(sha256(source),entry.sourceSHA256);
  assert.equal(sha256(expected),entry.expectedSHA256);
  const resized = resizeRgbBilinearPillow(source,entry.width,entry.height,entry.outputWidth,entry.outputHeight);
  assert.equal(resized.length,expected.length);
  let maxByteError=0, mismatches=0, totalByteError=0;
  for(let i=0;i<resized.length;i++) {
    const error=Math.abs(resized[i]-expected[i]);
    maxByteError=Math.max(maxByteError,error);
    totalByteError+=error;
    if(error) mismatches++;
  }
  // The initial engineering gate allows a single RGB8 intensity level. Report
  // exact-byte equality separately; never turn this tolerance into an exact claim.
  assert.ok(maxByteError<=1,`${entry.name}: RGB8 resize error ${maxByteError} exceeds one byte level`);
  const normalized = normalizeImageNetPlanar(resized,entry.outputWidth,entry.outputHeight);
  const area=entry.outputWidth*entry.outputHeight;
  let maxNormalizationError=0, maxTensorErrorAgainstPillow=0;
  for(let c=0;c<3;c++) for(let i=0;i<area;i++) {
    const expectedValue=fixture.normalizationLookup[c][resized[i*3+c]];
    maxNormalizationError=Math.max(maxNormalizationError,Math.abs(normalized[c*area+i]-expectedValue));
    const pillowTensorValue=fixture.normalizationLookup[c][expected[i*3+c]];
    maxTensorErrorAgainstPillow=Math.max(maxTensorErrorAgainstPillow,Math.abs(normalized[c*area+i]-pillowTensorValue));
  }
  assert.equal(maxNormalizationError,0,`${entry.name}: float32 normalization differs from reference`);
  results.push({name:entry.name,width:entry.width,height:entry.height,outputWidth:entry.outputWidth,
    outputHeight:entry.outputHeight,comparedBytes:resized.length,maxByteError,mismatchingBytes:mismatches,
    meanAbsoluteByteError:totalByteError/resized.length,exactRGB8Parity:mismatches===0,
    normalizedRGBErrorBound:maxByteError/255,maxNormalizationError,maxTensorErrorAgainstPillow});
}
const rgba=new Uint8ClampedArray([17,128,247,255,1,2,3,255]);
assert.deepEqual(Array.from(opaqueRgbaToRgb(rgba,2,1)),[17,128,247,1,2,3]);
assert.throws(()=>opaqueRgbaToRgb(new Uint8Array([17,128,247,254]),1,1));
const levels=new Uint8Array(256*3);
for(let i=0;i<256;i++) levels.fill(i,i*3,i*3+3);
const normalizedLevels=normalizeImageNetPlanar(levels,256,1);
for(let c=0;c<3;c++) for(let i=0;i<256;i++) assert.equal(normalizedLevels[c*256+i],fixture.normalizationLookup[c][i]);
assert.throws(()=>resizeRgbBilinearPillow(new Uint8Array(3),0,1));
assert.throws(()=>resizeRgbBilinearPillow(new Uint8Array(2),1,1));
const implementationPath=fileURLToPath(new URL('./resize_pillow.mjs',import.meta.url));
const report={status:'passed',execution:'GitHub Actions synthetic RGB8 preprocessing; no neural inference',
  runId:process.env.GITHUB_RUN_ID,commit:process.env.GITHUB_SHA ?? null,patientDataUsed:false,
  completedAt:new Date().toISOString(),pillowVersion:fixture.pillowVersion,caseCount:results.length,
  fixtureSHA256:sha256(rawFixture),implementationSHA256:sha256(await readFile(implementationPath)),
  exactRGB8Parity:results.every(row=>row.exactRGB8Parity),
  maxByteError:Math.max(...results.map(row=>row.maxByteError)),
  maxNormalizationError:Math.max(...results.map(row=>row.maxNormalizationError)),
  maxTensorErrorAgainstPillow:Math.max(...results.map(row=>row.maxTensorErrorAgainstPillow)),
  allByteLevelsAllChannelsNormalizationExact:true,rgbaOpaqueConversionVerified:true,
  transparentInputsRejected:true,
  scope:'Decoded opaque RGB8 bytes, whole-image bilinear resize with antialias support, ImageNet float32 normalization',
  notEstablished:['Image decoder equivalence','EXIF orientation equivalence','ICC/color-profile conversion equivalence','Clinical or model performance'],
  cases:results};
await writeFile(reportPath,JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({report:path.resolve(reportPath),status:report.status,caseCount:report.caseCount,
  exactRGB8Parity:report.exactRGB8Parity,maxByteError:report.maxByteError,maxNormalizationError:report.maxNormalizationError}));
