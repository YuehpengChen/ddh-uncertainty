/** Pure synthetic pixel arithmetic; no file decoder or neural model is loaded. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {resizeRgbBilinearPillow, normalizeImageNetPlanar, opaqueRgbaToRgb} from './resize_pillow.mjs';

test('one opaque RGB pixel retains channel order when enlarged',()=>{
  const resized=resizeRgbBilinearPillow(new Uint8Array([17,128,247]),1,1,11,7);
  assert.equal(resized.length,11*7*3);
  for(let i=0;i<resized.length;i+=3) assert.deepEqual(Array.from(resized.slice(i,i+3)),[17,128,247]);
});

test('identity resize is byte-exact and does not alias the source buffer',()=>{
  const source=new Uint8Array([1,2,3,4,5,6,7,8,9,10,11,12]);
  const resized=resizeRgbBilinearPillow(source,2,2,2,2);
  assert.deepEqual(resized,source);
  resized[0]=255;
  assert.equal(source[0],1);
});

test('two equal RGB weights round to nearest byte',()=>{
  const resized=resizeRgbBilinearPillow(new Uint8Array([0,20,200,255,40,100]),2,1,1,1);
  assert.deepEqual(Array.from(resized),[128,30,150]);
});

test('separable RGB8 resize rounds between horizontal and vertical passes',()=>{
  const source=Uint8Array.from([0,0,0,0,0,0,0,0,0,1,1,1]);
  assert.deepEqual(Array.from(resizeRgbBilinearPillow(source,2,2,1,1)),[1,1,1]);
});

test('ImageNet normalization is channel-planar float32 arithmetic',()=>{
  const rgb=new Uint8Array([0,128,255,255,64,0]);
  const result=normalizeImageNetPlanar(rgb,2,1);
  const mean=[.485,.456,.406],sd=[.229,.224,.225];
  for(let c=0;c<3;c++) for(let i=0;i<2;i++) {
    const expected=Math.fround(Math.fround(Math.fround(rgb[i*3+c]/255)-Math.fround(mean[c]))/Math.fround(sd[c]));
    assert.equal(result[c*2+i],expected);
  }
});

test('transparent RGBA is rejected rather than silently composited',()=>{
  assert.deepEqual(Array.from(opaqueRgbaToRgb(new Uint8Array([1,2,3,255]),1,1)),[1,2,3]);
  assert.throws(()=>opaqueRgbaToRgb(new Uint8Array([1,2,3,254]),1,1));
});

test('invalid dimensions and buffer lengths are rejected',()=>{
  for(const [w,h] of [[0,1],[1,-1],[NaN,1],[1.5,2]]) assert.throws(()=>resizeRgbBilinearPillow(new Uint8Array(3),w,h));
  assert.throws(()=>resizeRgbBilinearPillow(new Uint8Array(2),1,1));
  assert.throws(()=>normalizeImageNetPlanar(new Uint8Array(4),1,1));
});
