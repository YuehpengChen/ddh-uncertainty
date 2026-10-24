/** Read-only file-integrity checks; no neural runtime is imported or executed. */
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import path from 'node:path';

const root=path.resolve(process.argv[2] || '_site');
const readJSON=async name=>JSON.parse(await readFile(path.join(root,name),'utf8'));
const sha256=bytes=>createHash('sha256').update(bytes).digest('hex');
const manifest=await readJSON('models/manifest.json');
const proof=await readJSON('models/cloud_validation.json');
const preprocessing=await readJSON('models/preprocessing_validation.json');
assert.equal(manifest.status,'ready');
assert.equal(manifest.browserValidation.status,'passed');
assert.equal(proof.patientDataUsed,false);
assert.equal(proof.strictWeightLoading,true);
assert.equal(proof.frozenBatchNorm,true);
assert.equal(proof.browserWASM.status,'passed');
assert.equal(proof.browserWASM.passes,50);
assert.equal(proof.browserWASM.uniquePasses,50);
assert.equal(proof.browserWASM.frozenStateAfterMC,true);
assert.equal(manifest.dropout.mechanism,'explicit_masks');
assert.equal(manifest.dropout.inputs.length,26);
assert.equal(preprocessing.exactRGB8Parity,true);
assert.equal(preprocessing.implementationSHA256,sha256(await readFile(path.join(root,'validation/resize_pillow.mjs'))));
for(const [name,description] of Object.entries(manifest.models)) {
  const bytes=await readFile(path.join(root,description.url));
  assert.equal(bytes.length,description.bytes,`${name} byte count`);
  assert.equal(sha256(bytes),description.sha256,`${name} checksum`);
}
console.log(JSON.stringify({status:'passed',checks:'Released model and preprocessing file integrity',modelBytes:Object.values(manifest.models).reduce((sum,value)=>sum+value.bytes,0),cloudRunId:proof.runId}));
