import { readFileSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const probeSurface = `MoonlabModule.lastBackendRuntimeProbe = {
  owner: 'MoonlabModule',
  operation: 'factory-definition',
  backendName: 'wasm-factory',
  backend_name: 'wasm-factory',
  backendAvailable: false,
  moduleReady: false,
  missingUnifiedGpuApi: [],
  fallbackIntentional: false,
  reason: 'not-probed',
};

MoonlabModule.recordBackendRuntimeProbe = function(details = {}) {
  MoonlabModule.lastBackendRuntimeProbe = {
    owner: 'MoonlabModule',
    operation: details.operation || 'factory-event',
    backendName: 'wasm-factory',
    backend_name: 'wasm-factory',
    backendAvailable: Boolean(details.backendAvailable),
    moduleReady: Boolean(details.moduleReady),
    missingUnifiedGpuApi: details.missingUnifiedGpuApi || [],
    fallbackIntentional: Boolean(details.fallbackIntentional),
    reason: details.reason || 'factory-event',
  };
  return MoonlabModule.getLastBackendRuntimeProbe();
};

MoonlabModule.getLastBackendRuntimeProbe = function() {
  return JSON.parse(JSON.stringify(MoonlabModule.lastBackendRuntimeProbe));
};

MoonlabModule.probeBackendRuntime = async function(moduleArg = {}) {
  const instance = await MoonlabModule(moduleArg);
  if (instance && instance.ready) {
    await instance.ready;
  }

  const trace_runtime_name_for_type = (backendType) => {
    switch (backendType) {
      case 0: return 'none';
      case 1: return 'metal';
      case 2: return 'webgpu';
      case 3: return 'opencl';
      case 4: return 'vulkan';
      case 5: return 'cuda';
      case 6: return 'cuquantum';
      case 7: return 'auto';
      default: return 'unknown';
    }
  };

  const requiredUnifiedGpuApi = [
    '_gpu_compute_init',
    '_gpu_compute_free',
    '_gpu_get_backend_type',
    '_gpu_buffer_create_from_data',
    '_gpu_buffer_create',
    '_gpu_buffer_read',
    '_gpu_buffer_free',
    '_gpu_compute_probabilities',
  ];
  const missingUnifiedGpuApi = requiredUnifiedGpuApi.filter((name) => {
    if (name === '_gpu_compute_probabilities') {
      return (
        typeof instance._gpu_compute_probabilities_u32 !== 'function' &&
        typeof instance._gpu_compute_probabilities !== 'function'
      );
    }
    return typeof instance[name] !== 'function';
  });

  const trace_runtime_backend_selection = (preferredBackendType) => {
    if (missingUnifiedGpuApi.length > 0) {
      return {
        preferredBackendType,
        ctxCreated: false,
        backendType: 0,
        backendName: 'none',
        backend_name: 'none',
        nativeAccelerated: false,
        fallbackIntentional: true,
        reason: 'unified-gpu-api-unavailable',
      };
    }

    const ctxPtr = instance._gpu_compute_init(preferredBackendType);
    if (!ctxPtr) {
      return {
        preferredBackendType,
        ctxCreated: false,
        backendType: 0,
        backendName: 'none',
        backend_name: 'none',
        nativeAccelerated: false,
        fallbackIntentional: true,
        reason: 'gpu-context-unavailable',
      };
    }

    const backendType = instance._gpu_get_backend_type(ctxPtr);
    const nativeAccelerated =
      typeof instance._gpu_is_native_accelerated === 'function' &&
      instance._gpu_is_native_accelerated(ctxPtr) !== 0;
    instance._gpu_compute_free(ctxPtr);
    return {
      preferredBackendType,
      ctxCreated: true,
      backendType,
      backendName: trace_runtime_name_for_type(backendType),
      backend_name: trace_runtime_name_for_type(backendType),
      nativeAccelerated,
      fallbackIntentional: backendType !== preferredBackendType,
      reason: backendType === preferredBackendType ? 'ok' : \`selected-backend-\${backendType}\`,
    };
  };

  const trace = {
    owner: 'MoonlabModule',
    operation: 'probeBackendRuntime',
    backendName: 'wasm-unified-gpu',
    backend_name: 'wasm-unified-gpu',
    backendAvailable: missingUnifiedGpuApi.length === 0,
    moduleReady: true,
    missingUnifiedGpuApi,
    webgpu: trace_runtime_backend_selection(2),
    auto: trace_runtime_backend_selection(7),
  };
  trace.fallbackIntentional =
    missingUnifiedGpuApi.length > 0 || trace.webgpu.fallbackIntentional;
  trace.reason =
    missingUnifiedGpuApi.length > 0 ? 'unified-gpu-api-unavailable' : trace.webgpu.reason;
  MoonlabModule.lastBackendRuntimeProbe = trace;
  return trace;
};`;

const invocationTrace = `    if (typeof MoonlabModule !== 'undefined' &&
        typeof MoonlabModule.recordBackendRuntimeProbe === 'function') {
      MoonlabModule.recordBackendRuntimeProbe({
        operation: 'factory-invocation',
        moduleReady: false,
        backendAvailable: false,
        reason: 'factory-loading',
      });
    }
`;

const readyTrace = `    if (typeof MoonlabModule !== 'undefined' &&
        typeof MoonlabModule.recordBackendRuntimeProbe === 'function') {
      MoonlabModule.recordBackendRuntimeProbe({
        operation: 'factory-ready',
        moduleReady: true,
        backendAvailable: true,
        reason: 'wasm-runtime-ready',
      });
    }
`;

export function postprocessMoonlabFactorySource(source) {
  if (
    source.includes('function build_moonlab_module_factory()') &&
    source.includes('MoonlabModule.probeBackendRuntime')
  ) {
    return source;
  }

  let output = source;
  const factoryStart = /var\s+MoonlabModule\s*=\s*\(\(\)\s*=>\s*\{/;
  if (!factoryStart.test(output)) {
    throw new Error('moonlab.js does not contain the expected MODULARIZE factory start');
  }
  output = output.replace(factoryStart, 'function build_moonlab_module_factory() {\n  return (() => {');

  const moduleReturn = /(^|\n)([ \t]*)return\s+moduleRtn\s*;/g;
  let moduleReturnMatch = null;
  for (const match of output.matchAll(moduleReturn)) {
    moduleReturnMatch = match;
  }
  if (!moduleReturnMatch) {
    throw new Error('moonlab.js does not contain the expected factory return');
  }
  const moduleReturnIndex = moduleReturnMatch.index + moduleReturnMatch[1].length;
  output = `${output.slice(0, moduleReturnIndex)}${readyTrace}${output.slice(moduleReturnIndex)}`;

  const invocationNeedle = /(^|\n)([ \t]*)return\s*\(?\s*(?:async\s+)?function\s*\([^)]*\)\s*\{\s*\n?/;
  if (!invocationNeedle.test(output)) {
    throw new Error('moonlab.js does not contain the expected factory entry');
  }
  output = output.replace(invocationNeedle, (match) => `${match}${invocationTrace}`);

  const exportMarker = '// Export using a UMD style export, or ES6 exports if selected';
  const exportMarkerIndex = output.indexOf(exportMarker);
  const closeIndex = exportMarkerIndex >= 0 ? output.lastIndexOf('})();', exportMarkerIndex) : -1;
  if (exportMarkerIndex < 0 || closeIndex < 0) {
    throw new Error('moonlab.js does not contain the expected factory close before exports');
  }
  output =
    `${output.slice(0, closeIndex)}  })();\n}\n\n` +
    `var MoonlabModule = build_moonlab_module_factory();\n\n${probeSurface}\n` +
    output.slice(exportMarkerIndex);

  return output;
}

export function postprocessMoonlabFactoryFile(inputPath, outputPath) {
  const source = readFileSync(inputPath, 'utf8');
  writeFileSync(outputPath, `${postprocessMoonlabFactorySource(source).trimEnd()}\n`);
}

const scriptPath = fileURLToPath(import.meta.url);
const isCli = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];

if (isCli) {
  const [, , inputPath, outputPath] = process.argv;
  if (!inputPath || !outputPath) {
    const scriptName = scriptPath.slice(dirname(scriptPath).length + 1);
    throw new Error(`usage: node ${scriptName} <input moonlab.js> <output moonlab.js>`);
  }
  postprocessMoonlabFactoryFile(inputPath, outputPath);
}
