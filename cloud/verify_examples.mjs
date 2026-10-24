/** Static image/licensing integrity checks only: no image or neural inference. */
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import path from 'node:path';

const root=path.resolve(process.argv[2] || '.');
const manifest=JSON.parse(await readFile(path.join(root,'examples/manifest.json'),'utf8'));
assert.equal(manifest.schemaVersion,1);
assert.equal(manifest.items.length,3);
assert.equal(manifest.license.label,'Apache-2.0');
assert.equal(manifest.sourceCommit,'de5fb68c9094c668205fde3ce1225d337c3e3200');
const license=await readFile(path.join(root,manifest.license.url));
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
assert.equal(hash(license),'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4');
const unique=new Set();
for(const item of manifest.items) {
  assert.match(item.url,/^examples\/hip-0[123]\.png$/);
  const bytes=await readFile(path.join(root,item.url));
  assert.equal(bytes.length,item.bytes);
  assert.equal(hash(bytes),item.sha256);
  assert.equal(item.clinicalDiagnosis,null);
  assert.ok(!unique.has(item.sha256),'Example files must be distinct'); unique.add(item.sha256);
  assert.equal(bytes.subarray(0,8).toString('hex'),'89504e470d0a1a0a');
  let offset=8,ended=false;
  while(offset<bytes.length) {
    const length=bytes.readUInt32BE(offset),type=bytes.toString('ascii',offset+4,offset+8);
    assert.ok(['IHDR','IDAT','IEND'].includes(type),`Unexpected PNG chunk ${type}`);
    assert.ok(offset+length+12<=bytes.length);
    if(type==='IHDR') {
      assert.equal(bytes.readUInt32BE(offset+8),item.width);
      assert.equal(bytes.readUInt32BE(offset+12),item.height);
      assert.equal(bytes[offset+16],8); assert.equal(bytes[offset+17],2);
    }
    if(type==='IEND') { ended=true; assert.equal(offset+12,bytes.length); }
    offset+=length+12;
  }
  assert.ok(ended);
}
console.log(JSON.stringify({status:'passed',examples:3,unchangedSourceBytes:true,PNGMetadataChunks:false,licensePreserved:true,scope:'Static file integrity; not clinical validation'}));
