/** Actual ONNX Runtime Web WASM verification, GitHub cloud only.
 * Usage: node cloud/verify_browser.mjs public
 * Requires public/models/{graphs,manifest,synthetic fixtures,cloud_validation.json}
 * and public/runtime/ (the bundled onnxruntime-web dist files).
 */
import http from 'node:http';
import {readFile, writeFile, stat} from 'node:fs/promises';
import path from 'node:path';
import {chromium} from 'playwright';

if (process.env.GITHUB_ACTIONS !== 'true' || !process.env.GITHUB_RUN_ID) {
  throw new Error('Refused: browser model inference must run in GitHub Actions cloud, not on the author computer.');
}
const root = path.resolve(process.argv[2] ?? 'public');
const mime = {'.html':'text/html','.js':'text/javascript','.mjs':'text/javascript',
  '.wasm':'application/wasm','.json':'application/json','.onnx':'application/octet-stream',
  '.f32':'application/octet-stream'};
const pageHTML = `<!doctype html><meta charset="utf-8"><title>Cloud-only synthetic WASM validation</title>
<main id="status">Verifying model in GitHub-hosted Chromium</main><script type="module">
import * as ort from '/runtime/ort.min.mjs';
window.runValidation = async () => {
  ort.env.wasm.wasmPaths = new URL('/runtime/', location.href).href;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  const manifest = await (await fetch('/models/manifest.json')).json();
  const expected = await (await fetch('/models/synthetic_expected.json')).json();
  const raw = await (await fetch('/models/synthetic_input.f32')).arrayBuffer();
  const input = new ort.Tensor('float32', new Float32Array(raw), [1,3,448,448]);
  const encoder = await ort.InferenceSession.create('/models/backbone.onnx', {executionProviders:['wasm']});
  const head = await ort.InferenceSession.create('/models/stage2_decoder.onnx', {executionProviders:['wasm']});
  const features = await encoder.run({image:input});
  const feeds = {...features, stochastic:new ort.Tensor('bool', new Uint8Array([0]), [])};
  const deterministic = await head.run(feeds);
  const mean = Array.from(deterministic.mean.data), variance = Array.from(deterministic.variance.data);
  const maxMeanError = Math.max(...mean.map((v,i)=>Math.abs(v-expected.mean[i])));
  const maxVarianceError = Math.max(...variance.map((v,i)=>Math.abs(v-expected.variance[i])));
  if (maxMeanError>0.0001 || maxVarianceError>0.000001) {
    throw new Error('WASM/native deterministic parity failed: '+JSON.stringify({maxMeanError,maxVarianceError}));
  }
  const repeat = await head.run(feeds);
  if (Array.from(repeat.mean.data).some((v,i)=>v!==mean[i]) || Array.from(repeat.variance.data).some((v,i)=>v!==variance[i])) {
    throw new Error('Deterministic WASM outputs changed between identical calls.');
  }
  feeds.stochastic = new ort.Tensor('bool', new Uint8Array([1]), []);
  const samples = [], conditionalVariances = [];
  const started = performance.now();
  for(let p=0;p<50;p++) {
    const out = await head.run(feeds);
    const mu = Array.from(out.mean.data), vv = Array.from(out.variance.data);
    if(!mu.every(Number.isFinite)||!vv.every(v=>Number.isFinite(v)&&v>0)) throw new Error('Invalid stochastic output.');
    samples.push(mu); conditionalVariances.push(vv);
    if((p+1)%10===0) console.log('Cloud WASM MC pass '+(p+1)+'/50');
  }
  const average = mean.map((_,j)=>samples.reduce((s,r)=>s+r[j],0)/samples.length);
  const spreads = average.map((v,j)=>samples.reduce((s,r)=>s+(r[j]-v)**2,0)/samples.length);
  const unique = new Set(samples.map(r=>JSON.stringify(r))).size;
  if(unique<25 || Math.max(...spreads)<1e-12) throw new Error('WASM Dropout was removed, unsupported, or repeated.');
  feeds.stochastic = new ort.Tensor('bool', new Uint8Array([0]), []);
  const afterMC = await head.run(feeds);
  if(Array.from(afterMC.mean.data).some((v,i)=>v!==mean[i]) || Array.from(afterMC.variance.data).some((v,i)=>v!==variance[i])) {
    throw new Error('Deterministic outputs changed after MC; frozen inference state was modified.');
  }
  const within = mean.map((_,j)=>conditionalVariances.reduce((s,r)=>s+r[j],0)/50);
  const covariance = average.map((v,j)=>average.map((w,k)=>samples.reduce((s,r)=>s+(r[j]-v)*(r[k]-w),0)/50));
  const totalDiagonal = spreads.map((v,j)=>v+within[j]);
  if(!totalDiagonal.every(v=>Number.isFinite(v)&&v>0) || covariance.some((r,j)=>Math.abs(r[j]-spreads[j])>1e-12)) {
    throw new Error('Full covariance/conditional-variance decomposition failed.');
  }
  await encoder.release(); await head.release();
  return {status:'passed',execution:'GitHub Actions hosted Chromium, ONNX Runtime Web WASM',
    patientDataUsed:false,passes:50,finite:true,uniquePasses:unique,
    meanMaxAbsError:maxMeanError,varianceMaxAbsError:maxVarianceError,
    maxCoordinateVariance:Math.max(...spreads),frozenStateAfterMC:true,
    covarianceDenominator:50,P4VarianceNormalized:totalDiagonal.slice(6,8),
    totalVarianceFinite:true,seconds:(performance.now()-started)/1000,
    userAgent:navigator.userAgent,threads:1,externalInferenceRequests:0};
};
</script>`;

const server = http.createServer(async (request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
    if (pathname === '/__verification') {
      response.writeHead(200, {'Content-Type':'text/html','Cache-Control':'no-store'});
      response.end(pageHTML); return;
    }
    const target = path.resolve(root, '.' + pathname);
    if (!target.startsWith(root + path.sep)) throw new Error('Path outside verification root');
    const contents = await readFile(target);
    response.writeHead(200, {'Content-Type':mime[path.extname(target)] ?? 'application/octet-stream',
      'Cache-Control':'no-store','Content-Length':contents.byteLength});
    response.end(contents);
  } catch(error) {response.writeHead(404);response.end(String(error.message));}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const base = `http://127.0.0.1:${server.address().port}`;
let browser;
try {
  browser = await chromium.launch({headless:true});
  const page = await browser.newPage();
  page.setDefaultTimeout(30*60*1000);
  const blocked = [];
  await page.route('**/*', route=>{
    const url = route.request().url();
    if(url.startsWith(base+'/')||url.startsWith('blob:')||url.startsWith('data:')) return route.continue();
    blocked.push(url); return route.abort();
  });
  page.on('console',message=>console.log('[WASM]',message.text()));
  page.on('pageerror',error=>console.error('[Browser error]',error.message));
  await page.goto(base+'/__verification');
  await page.waitForFunction(()=>typeof window.runValidation==='function');
  const result = await page.evaluate(()=>window.runValidation());
  if(blocked.length) throw new Error('Unexpected external requests: '+JSON.stringify(blocked));
  const validationPath = path.join(root,'models/cloud_validation.json');
  const proof = JSON.parse(await readFile(validationPath,'utf8'));
  proof.browserWASM = result;
  await writeFile(validationPath,JSON.stringify(proof,null,2)+'\n');
  const manifestPath = path.join(root,'models/manifest.json');
  const manifest = JSON.parse(await readFile(manifestPath,'utf8'));
  manifest.status = 'ready';
  manifest.browserValidation = {status:'passed',passes:50,patientDataUsed:false,
    execution:result.execution,runId:process.env.GITHUB_RUN_ID};
  await writeFile(manifestPath,JSON.stringify(manifest,null,2)+'\n');
  console.log(JSON.stringify(result,null,2));
} finally {
  if(browser) await browser.close();
  await new Promise(resolve=>server.close(resolve));
}
