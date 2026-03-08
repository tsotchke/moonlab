#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const packageRoot = path.resolve(__dirname, '..');
const artifactDir = path.resolve(packageRoot, 'artifacts/webgpu_unified_eval');
const resultsPath = path.resolve(artifactDir, 'results.json');
const summaryPath = path.resolve(artifactDir, 'summary.txt');

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

function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let x = Math.imul(t ^ (t >>> 15), t | 1);
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function randomInt(rand, min, maxInclusive) {
  return Math.floor(rand() * (maxInclusive - min + 1)) + min;
}

function randomAngle(rand) {
  return (rand() * 2 - 1) * Math.PI;
}

function applyHadamardInterleaved(state, qubit, stateDim) {
  const stride = 1 << qubit;
  const invSqrt2 = Math.SQRT1_2;
  const pairs = stateDim >>> 1;
  for (let idx = 0; idx < pairs; idx++) {
    const i0 = ((idx / stride) | 0) * (2 * stride) + (idx % stride);
    const i1 = i0 + stride;

    const r0 = state[i0 * 2];
    const i0im = state[i0 * 2 + 1];
    const r1 = state[i1 * 2];
    const i1im = state[i1 * 2 + 1];

    state[i0 * 2] = (r0 + r1) * invSqrt2;
    state[i0 * 2 + 1] = (i0im + i1im) * invSqrt2;
    state[i1 * 2] = (r0 - r1) * invSqrt2;
    state[i1 * 2 + 1] = (i0im - i1im) * invSqrt2;
  }
}

function applyPauliXInterleaved(state, qubit, stateDim) {
  const stride = 1 << qubit;
  const pairs = stateDim >>> 1;
  for (let idx = 0; idx < pairs; idx++) {
    const i0 = ((idx / stride) | 0) * (2 * stride) + (idx % stride);
    const i1 = i0 + stride;

    const r0 = state[i0 * 2];
    const i0im = state[i0 * 2 + 1];
    state[i0 * 2] = state[i1 * 2];
    state[i0 * 2 + 1] = state[i1 * 2 + 1];
    state[i1 * 2] = r0;
    state[i1 * 2 + 1] = i0im;
  }
}

function applyPauliZInterleaved(state, qubit, stateDim) {
  const mask = 1 << qubit;
  for (let i = 0; i < stateDim; i++) {
    if ((i & mask) !== 0) {
      state[i * 2] = -state[i * 2];
      state[i * 2 + 1] = -state[i * 2 + 1];
    }
  }
}

function applyPhaseInterleaved(state, qubit, theta, stateDim) {
  const mask = 1 << qubit;
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  for (let i = 0; i < stateDim; i++) {
    if ((i & mask) !== 0) {
      const re = state[i * 2];
      const im = state[i * 2 + 1];
      state[i * 2] = re * c - im * s;
      state[i * 2 + 1] = re * s + im * c;
    }
  }
}

function applyCnotInterleaved(state, control, target, stateDim) {
  const targetStride = 1 << target;
  const controlMask = 1 << control;
  const pairs = stateDim >>> 1;
  for (let idx = 0; idx < pairs; idx++) {
    const i0 = ((idx / targetStride) | 0) * (2 * targetStride) + (idx % targetStride);
    if ((i0 & controlMask) === 0) {
      continue;
    }
    const i1 = i0 + targetStride;

    const r0 = state[i0 * 2];
    const i0im = state[i0 * 2 + 1];
    state[i0 * 2] = state[i1 * 2];
    state[i0 * 2 + 1] = state[i1 * 2 + 1];
    state[i1 * 2] = r0;
    state[i1 * 2 + 1] = i0im;
  }
}

function probabilitiesFromInterleaved(state, stateDim) {
  const probs = new Float64Array(stateDim);
  for (let i = 0; i < stateDim; i++) {
    const re = state[i * 2];
    const im = state[i * 2 + 1];
    probs[i] = re * re + im * im;
  }
  return probs;
}

function maxAbsDiff(a, b) {
  let max = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = Math.abs(a[i] - b[i]);
    if (diff > max) {
      max = diff;
    }
  }
  return max;
}

function sampleOp(rand, numQubits) {
  const oneQubitOps = ['h', 'x', 'z', 'phase'];
  if (numQubits > 1 && rand() < 0.3) {
    let control = randomInt(rand, 0, numQubits - 1);
    let target = randomInt(rand, 0, numQubits - 1);
    while (target === control) {
      target = randomInt(rand, 0, numQubits - 1);
    }
    return { type: 'cnot', control, target };
  }

  const type = oneQubitOps[randomInt(rand, 0, oneQubitOps.length - 1)];
  const qubit = randomInt(rand, 0, numQubits - 1);
  if (type === 'phase') {
    return { type, qubit, theta: randomAngle(rand) };
  }
  return { type, qubit };
}

function applyCpuOp(state, op, stateDim) {
  switch (op.type) {
    case 'h':
      applyHadamardInterleaved(state, op.qubit, stateDim);
      break;
    case 'x':
      applyPauliXInterleaved(state, op.qubit, stateDim);
      break;
    case 'z':
      applyPauliZInterleaved(state, op.qubit, stateDim);
      break;
    case 'phase':
      applyPhaseInterleaved(state, op.qubit, op.theta, stateDim);
      break;
    case 'cnot':
      applyCnotInterleaved(state, op.control, op.target, stateDim);
      break;
    default:
      throw new Error(`Unsupported op: ${op.type}`);
  }
}

function applyGpuOp(session, buffer, op, stateDim) {
  switch (op.type) {
    case 'h':
      return session.hadamard(buffer, op.qubit, stateDim);
    case 'x':
      return session.pauliX(buffer, op.qubit, stateDim);
    case 'z':
      return session.pauliZ(buffer, op.qubit, stateDim);
    case 'phase':
      return session.phase(buffer, op.qubit, op.theta, stateDim);
    case 'cnot':
      return session.cnot(buffer, op.control, op.target, stateDim);
    default:
      return -7;
  }
}

async function main() {
  const seed = Number(getEnv('MOONLAB_WEBGPU_UNIFIED_EVAL_SEED', '20260307'));
  const numCases = Number(getEnv('MOONLAB_WEBGPU_UNIFIED_EVAL_CASES', '24'));
  const maxAbsTolerance = Number(getEnv('MOONLAB_WEBGPU_UNIFIED_EVAL_MAX_ABS', '1e-5'));
  const requireWebGPU = getEnv('MOONLAB_WEBGPU_UNIFIED_EVAL_REQUIRE_BACKEND', '0') === '1';
  const requireNative = getEnv('MOONLAB_WEBGPU_UNIFIED_EVAL_REQUIRE_NATIVE', '0') === '1';

  const rand = mulberry32(seed);
  const report = {
    generatedAt: new Date().toISOString(),
    seed,
    numCases,
    maxAbsTolerance,
    requireWebGPU,
    requireNative,
    sessionCreated: false,
    backend: 'none',
    nativeAccelerated: false,
    failures: 0,
    maxObservedAbsDiff: 0,
    cases: [],
    passed: false,
    reason: '',
  };

  const session = await GPUBackendSession.create();
  if (!session) {
    report.reason = 'GPUBackendSession.create returned null';
    report.passed = !requireWebGPU;
    if (!report.passed) {
      report.failures += 1;
    }

    await fs.mkdir(artifactDir, { recursive: true });
    await fs.writeFile(resultsPath, JSON.stringify(report, null, 2), 'utf8');

    const summary = [
      'Moonlab WebGPU Unified Eval',
      `Date: ${report.generatedAt}`,
      `Seed: ${seed}`,
      `Cases: ${numCases}`,
      'Session: unavailable',
      `Require WebGPU backend: ${requireWebGPU}`,
      `Result: ${report.passed ? 'PASS' : 'FAIL'}`,
      `Reason: ${report.reason}`,
    ].join('\n');

    await fs.writeFile(summaryPath, `${summary}\n`, 'utf8');
    if (!report.passed) {
      console.error(summary);
      process.exitCode = 1;
    } else {
      console.log(summary);
    }
    return;
  }

  report.sessionCreated = true;
  report.backend = backendTypeName(session.backendType);
  report.nativeAccelerated = session.nativeAccelerated;

  try {
    if (requireWebGPU && session.backendType !== GPU_BACKEND_WEBGPU) {
      report.failures += 1;
      report.reason = `Expected webgpu backend, got ${report.backend}`;
    }

    if (session.backendType === 0) {
      if (!report.reason) {
        report.reason = 'No GPU backend available in this runtime';
      }
      report.passed = report.failures === 0;
      await fs.mkdir(artifactDir, { recursive: true });
      await fs.writeFile(resultsPath, JSON.stringify(report, null, 2), 'utf8');

      const summary = [
        'Moonlab WebGPU Unified Eval',
        `Date: ${report.generatedAt}`,
        `Seed: ${seed}`,
        `Cases: ${numCases}`,
        `Backend: ${report.backend}`,
        `Native accelerated: ${report.nativeAccelerated}`,
        `Require WebGPU backend: ${requireWebGPU}`,
        `Require native acceleration: ${requireNative}`,
        `Tolerance (max abs diff): ${maxAbsTolerance}`,
        `Observed max abs diff: ${report.maxObservedAbsDiff}`,
        `Failures: ${report.failures}`,
        `Result: ${report.passed ? 'PASS' : 'FAIL'}`,
        `Reason: ${report.reason}`,
      ].join('\n');

      await fs.writeFile(summaryPath, `${summary}\n`, 'utf8');
      if (!report.passed) {
        console.error(summary);
        process.exitCode = 1;
      } else {
        console.log(summary);
      }
      return;
    }

    for (let caseIndex = 0; caseIndex < numCases; caseIndex++) {
      const numQubits = randomInt(rand, 1, 6);
      const stateDim = 1 << numQubits;
      const depth = randomInt(rand, 6, 40);

      const cpuState = new Float64Array(stateDim * 2);
      cpuState[0] = 1;
      const gpuInit = new Float64Array(cpuState);
      const buffer = session.createBufferFromInterleaved(gpuInit);

      let opFailure = null;
      try {
        const ops = [];
        for (let d = 0; d < depth; d++) {
          const op = sampleOp(rand, numQubits);
          ops.push(op);
          applyCpuOp(cpuState, op, stateDim);
          const rc = applyGpuOp(session, buffer, op, stateDim);
          if (rc !== 0) {
            opFailure = {
              depth: d,
              op,
              rc,
            };
            break;
          }
        }

        let casePassed = false;
        let caseReason = '';
        let caseDiff = Number.NaN;

        if (!opFailure) {
          const expectedProbs = probabilitiesFromInterleaved(cpuState, stateDim);
          const observedProbs = session.computeProbabilities(buffer, stateDim);
          caseDiff = maxAbsDiff(expectedProbs, observedProbs);
          report.maxObservedAbsDiff = Math.max(report.maxObservedAbsDiff, caseDiff);
          casePassed = caseDiff <= maxAbsTolerance;
          if (!casePassed) {
            caseReason = `maxAbsDiff=${caseDiff}`;
          }
        } else {
          caseReason = `op failed at depth=${opFailure.depth} rc=${opFailure.rc}`;
        }

        if (!casePassed) {
          report.failures += 1;
        }

        report.cases.push({
          caseId: caseIndex + 1,
          numQubits,
          stateDim,
          depth,
          opFailure,
          maxAbsDiff: Number.isNaN(caseDiff) ? null : caseDiff,
          passed: casePassed,
          reason: caseReason,
        });
      } finally {
        session.freeBuffer(buffer);
      }
    }

    report.nativeAccelerated = session.nativeAccelerated;
    if (requireNative && !report.nativeAccelerated) {
      report.failures += 1;
      if (!report.reason) {
        report.reason = 'Native acceleration was required but not active';
      }
    }
  } finally {
    session.dispose();
  }

  report.passed = report.failures === 0;
  if (!report.reason) {
    report.reason = report.passed ? 'PASS' : 'One or more eval cases failed';
  }

  await fs.mkdir(artifactDir, { recursive: true });
  await fs.writeFile(resultsPath, JSON.stringify(report, null, 2), 'utf8');

  const summary = [
    'Moonlab WebGPU Unified Eval',
    `Date: ${report.generatedAt}`,
    `Seed: ${seed}`,
    `Cases: ${numCases}`,
    `Backend: ${report.backend}`,
    `Native accelerated: ${report.nativeAccelerated}`,
    `Require WebGPU backend: ${requireWebGPU}`,
    `Require native acceleration: ${requireNative}`,
    `Tolerance (max abs diff): ${maxAbsTolerance}`,
    `Observed max abs diff: ${report.maxObservedAbsDiff}`,
    `Failures: ${report.failures}`,
    `Result: ${report.passed ? 'PASS' : 'FAIL'}`,
    `Reason: ${report.reason}`,
  ].join('\n');

  await fs.writeFile(summaryPath, `${summary}\n`, 'utf8');

  if (!report.passed) {
    console.error(summary);
    process.exitCode = 1;
    return;
  }

  console.log(summary);
}

await main();
