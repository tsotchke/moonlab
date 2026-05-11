import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import vm from 'node:vm';

const demoRoot = resolve(new URL('..', import.meta.url).pathname);
const repoRoot = resolve(demoRoot, '../../..');

const assert = (condition, message) => {
  if (!condition) {
    throw new Error(message);
  }
};

const assertMoonlabModuleProbeSurface = (path) => {
  const source = readFileSync(path, 'utf8');
  assert(source.includes('MoonlabModule.lastBackendRuntimeProbe'), `${path}: missing last probe`);
  assert(source.includes('MoonlabModule.recordBackendRuntimeProbe'), `${path}: missing factory trace recorder`);
  assert(source.includes('MoonlabModule.getLastBackendRuntimeProbe'), `${path}: missing probe getter`);
  assert(source.includes('MoonlabModule.probeBackendRuntime'), `${path}: missing runtime probe`);
  assert(source.includes('factory-invocation'), `${path}: missing factory invocation trace`);
  assert(source.includes('factory-ready'), `${path}: missing factory ready trace`);
  assert(source.includes('backendName'), `${path}: missing backendName field`);
  assert(source.includes('backend_name'), `${path}: missing backend_name field`);
  assert(source.includes('backendAvailable'), `${path}: missing backend availability field`);
  assert(source.includes('fallbackIntentional'), `${path}: missing fallback field`);
};

const assertWorkerGpuGateTrace = (path) => {
  const source = readFileSync(path, 'utf8');
  const context = {
    console,
    performance: { now: () => 0 },
    URL,
    Float64Array,
    Int32Array,
    Uint8Array,
    ArrayBuffer,
    BigInt,
    setTimeout,
    clearTimeout,
  };
  context.self = {
    location: new URL('http://localhost/moonlab-worker.js'),
    postMessage: () => {},
    onmessage: null,
  };
  context.importScripts = () => {};
  context.globalThis = context;
  vm.createContext(context);

  vm.runInContext(
    `${source}
const fullModule = {
  _gpu_compute_init: () => 1,
  _gpu_compute_free: () => 0,
  _gpu_get_backend_type: () => GPU_BACKEND_WEBGPU,
  _gpu_buffer_create_from_data: () => 1,
  _gpu_buffer_create: () => 1,
  _gpu_buffer_read: () => 0,
  _gpu_buffer_free: () => 0,
  _gpu_compute_probabilities_u32: () => 0,
};
globalThis.__missingProbe = probeUnifiedGpuApi({});
globalThis.__missingHas = hasUnifiedGpuApi({}, globalThis.__missingProbe);
globalThis.__missingTrace = lastGpuBackendTrace;
globalThis.__fullProbe = probeUnifiedGpuApi(fullModule);
globalThis.__fullHas = hasUnifiedGpuApi(fullModule, globalThis.__fullProbe);
globalThis.__fullTrace = lastGpuBackendTrace;
`,
    context,
    { filename: path }
  );

  assert(context.__missingProbe.available === false, `${path}: missing probe should be unavailable`);
  assert(context.__missingHas === false, `${path}: missing gate should be false`);
  assert(context.__missingTrace.owner === 'hasUnifiedGpuApi', `${path}: missing trace owner`);
  assert(context.__missingTrace.backend_name === context.__missingTrace.backendName, `${path}: missing backend alias`);
  assert(context.__missingTrace.backendAvailable === false, `${path}: missing backend availability`);
  assert(context.__missingTrace.fallbackIntentional === true, `${path}: missing fallback flag`);
  assert(
    context.__missingTrace.reason === 'unified-gpu-api-unavailable',
    `${path}: missing fallback reason`
  );

  assert(context.__fullProbe.available === true, `${path}: full probe should be available`);
  assert(context.__fullHas === true, `${path}: full gate should be true`);
  assert(context.__fullTrace.owner === 'hasUnifiedGpuApi', `${path}: full trace owner`);
  assert(context.__fullTrace.backend_name === context.__fullTrace.backendName, `${path}: full backend alias`);
  assert(context.__fullTrace.backendAvailable === true, `${path}: full backend availability`);
  assert(context.__fullTrace.fallbackIntentional === false, `${path}: full fallback flag`);
  assert(context.__fullTrace.reason === 'ok', `${path}: full reason`);
};

assertMoonlabModuleProbeSurface(resolve(demoRoot, 'public/moonlab.js'));
assertMoonlabModuleProbeSurface(resolve(repoRoot, 'docs/moonlab.js'));
assertWorkerGpuGateTrace(resolve(demoRoot, 'public/moonlab-worker.js'));
assertWorkerGpuGateTrace(resolve(repoRoot, 'docs/moonlab-worker.js'));

console.log('worker backend trace smoke passed');
