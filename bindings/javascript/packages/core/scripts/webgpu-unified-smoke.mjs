#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const packageRoot = path.resolve(__dirname, '..');
const artifactDir = path.resolve(packageRoot, 'artifacts/webgpu_smoke');
const resultsPath = path.resolve(artifactDir, 'results.json');

const {
  GPUBackendSession,
  GPU_BACKEND_WEBGPU,
  backendTypeName,
} = await import(path.resolve(packageRoot, 'dist/index.mjs'));

function getEnv(name, fallback = '') {
  if (typeof process !== 'undefined' && process.env && typeof process.env[name] !== 'undefined') {
    return process.env[name];
  }
  if (typeof Deno !== 'undefined' && Deno.env) {
    try {
      const value = Deno.env.get(name);
      if (typeof value !== 'undefined') {
        return value;
      }
    } catch (_err) {
      // ignore when Deno env access is disallowed
    }
  }
  return fallback;
}

function approxEqual(a, b, tol = 1e-6) {
  return Math.abs(a - b) <= tol;
}

async function main() {
  const requireWebGPU = getEnv('MOONLAB_WEBGPU_SMOKE_REQUIRE_BACKEND', '0') === '1';
  const requireNativeWebGPU = getEnv('MOONLAB_WEBGPU_SMOKE_REQUIRE_NATIVE', '0') === '1';
  const report = {
    generatedAt: new Date().toISOString(),
    requireWebGPU,
    requireNativeWebGPU,
    sessionCreated: false,
    backend: 'none',
    nativeAccelerated: false,
    hadamardRc: null,
    pauliXRc: null,
    pauliZRc: null,
    cnotRc1: null,
    cnotRc2: null,
    probabilities: null,
    passed: false,
    reason: '',
  };

  const session = await GPUBackendSession.create();
  if (!session) {
    if (requireWebGPU) {
      report.reason = 'GPUBackendSession.create returned null';
      report.passed = false;
    } else {
      report.reason = 'No GPU backend available in this runtime';
      report.passed = true;
    }
    await fs.mkdir(artifactDir, { recursive: true });
    await fs.writeFile(resultsPath, JSON.stringify(report, null, 2), 'utf8');
    console.log(`WebGPU unified smoke: ${report.passed ? 'PASS' : 'FAIL'} (${report.reason})`);
    if (!report.passed) {
      throw new Error(report.reason);
    }
    return;
  }

  report.sessionCreated = true;
  report.backend = backendTypeName(session.backendType);
  report.nativeAccelerated = session.nativeAccelerated;
  const hadamardStateDim = 2;
  const hadamardBuffer = session.createBufferFromInterleaved(new Float64Array([
    1, 0,
    0, 0,
  ]));
  const pauliStateDim = 2;
  const pauliBuffer = session.createBufferFromInterleaved(new Float64Array([
    1, 0,
    0, 0,
  ]));
  const cnotStateDim = 4;
  const cnotInitial = new Float64Array([
    0.5, 0.0,
    0.2, -0.3,
    0.1, 0.4,
    -0.6, 0.2,
  ]);
  const cnotBuffer = session.createBufferFromInterleaved(cnotInitial);

  try {
    report.hadamardRc = session.hadamard(hadamardBuffer, 0, hadamardStateDim);
    report.pauliXRc = session.pauliX(pauliBuffer, 0, pauliStateDim);
    report.pauliZRc = session.pauliZ(pauliBuffer, 0, pauliStateDim);
    report.cnotRc1 = session.cnot(cnotBuffer, 0, 1, cnotStateDim);
    report.cnotRc2 = session.cnot(cnotBuffer, 0, 1, cnotStateDim);
    const isWebGPU = session.backendType === GPU_BACKEND_WEBGPU;

    if (requireWebGPU && !isWebGPU) {
      report.reason = `Expected WebGPU backend, got ${report.backend}`;
      report.passed = false;
      throw new Error(report.reason);
    }

    if (requireNativeWebGPU && !report.nativeAccelerated) {
      report.reason = 'Expected native WebGPU acceleration, but backend is in fallback mode';
      report.passed = false;
      throw new Error(report.reason);
    }

    if (isWebGPU && report.hadamardRc !== 0) {
      report.reason = `Hadamard failed on WebGPU backend: rc=${report.hadamardRc}`;
      report.passed = false;
      throw new Error(report.reason);
    }

    if (isWebGPU && (report.pauliXRc !== 0 || report.pauliZRc !== 0)) {
      report.reason = `Pauli gate failed on WebGPU backend: x=${report.pauliXRc}, z=${report.pauliZRc}`;
      report.passed = false;
      throw new Error(report.reason);
    }

    if (isWebGPU && (report.cnotRc1 !== 0 || report.cnotRc2 !== 0)) {
      report.reason = `CNOT failed on WebGPU backend: first=${report.cnotRc1}, second=${report.cnotRc2}`;
      report.passed = false;
      throw new Error(report.reason);
    }

    if (report.hadamardRc === 0) {
      const probs = session.computeProbabilities(hadamardBuffer, hadamardStateDim);
      report.probabilities = [probs[0], probs[1]];
      if (!approxEqual(probs[0], 0.5) || !approxEqual(probs[1], 0.5)) {
        report.reason = `Unexpected probabilities: [${probs[0]}, ${probs[1]}]`;
        report.passed = false;
        throw new Error(report.reason);
      }
    }

    if (report.pauliXRc === 0 && report.pauliZRc === 0) {
      const pauliState = session.readInterleavedBuffer(pauliBuffer, pauliStateDim);
      if (!approxEqual(pauliState[0], 0) ||
          !approxEqual(pauliState[1], 0) ||
          !approxEqual(pauliState[2], -1) ||
          !approxEqual(pauliState[3], 0)) {
        report.reason = `Unexpected pauli state: [${pauliState.join(', ')}]`;
        report.passed = false;
        throw new Error(report.reason);
      }
    }

    if (report.cnotRc1 === 0 && report.cnotRc2 === 0) {
      const cnotState = session.readInterleavedBuffer(cnotBuffer, cnotStateDim);
      for (let i = 0; i < cnotInitial.length; i++) {
        if (!approxEqual(cnotState[i], cnotInitial[i], 1e-5)) {
          report.reason = `CNOT involution mismatch at index ${i}: got=${cnotState[i]}, expected=${cnotInitial[i]}`;
          report.passed = false;
          throw new Error(report.reason);
        }
      }
    }

    report.passed = true;
    report.reason = 'PASS';
  } finally {
    session.freeBuffer(hadamardBuffer);
    session.freeBuffer(pauliBuffer);
    session.freeBuffer(cnotBuffer);
    session.dispose();
    await fs.mkdir(artifactDir, { recursive: true });
    await fs.writeFile(resultsPath, JSON.stringify(report, null, 2), 'utf8');
  }

  if (!report.passed) {
    console.error(`WebGPU unified smoke: FAIL (${report.reason})`);
    throw new Error(report.reason || 'webgpu unified smoke failed');
  }

  console.log(`WebGPU unified smoke: PASS (backend=${report.backend}, native=${report.nativeAccelerated})`);
}

await main().catch((err) => {
  const message = err instanceof Error ? err.message : String(err);
  console.error(`WebGPU unified smoke: FAIL (${message})`);
  throw err;
});
