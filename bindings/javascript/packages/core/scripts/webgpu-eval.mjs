#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const packageRoot = path.resolve(__dirname, '..');

const {
  QuantumState,
  getActiveTensorGPUBackend,
  initializeWebGPUBackend,
  isWebGPUAvailable,
} = await import(path.resolve(packageRoot, 'dist/index.mjs'));

const ARTIFACT_DIR = path.resolve(packageRoot, 'artifacts/webgpu_eval');
const RESULTS_PATH = path.resolve(ARTIFACT_DIR, 'results.json');
const SUMMARY_PATH = path.resolve(ARTIFACT_DIR, 'summary.txt');

const NUM_CASES = Number(process.env.MOONLAB_WEBGPU_EVAL_CASES || 20);
const SEED = Number(process.env.MOONLAB_WEBGPU_EVAL_SEED || 1337);
const MAX_ABS_TOL = Number(process.env.MOONLAB_WEBGPU_EVAL_MAX_ABS || 1e-8);
const REQUIRE_WEBGPU_BACKEND = process.env.MOONLAB_WEBGPU_EVAL_REQUIRE_BACKEND === '1';

const ONE_QUBIT_GATES = ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz'];
const TWO_QUBIT_GATES = ['cnot', 'cz', 'swap'];

function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6D2B79F5;
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

function sampleGate(rand, numQubits) {
  if (numQubits > 1 && rand() < 0.35) {
    const gate = TWO_QUBIT_GATES[randomInt(rand, 0, TWO_QUBIT_GATES.length - 1)];
    let q1 = randomInt(rand, 0, numQubits - 1);
    let q2 = randomInt(rand, 0, numQubits - 1);
    while (q2 === q1) {
      q2 = randomInt(rand, 0, numQubits - 1);
    }
    return { gate, q1, q2 };
  }

  const gate = ONE_QUBIT_GATES[randomInt(rand, 0, ONE_QUBIT_GATES.length - 1)];
  const q = randomInt(rand, 0, numQubits - 1);
  const angle = gate === 'rx' || gate === 'ry' || gate === 'rz' ? randomAngle(rand) : undefined;
  return { gate, q, angle };
}

function applyGate(state, op) {
  switch (op.gate) {
    case 'h':
      state.h(op.q);
      break;
    case 'x':
      state.x(op.q);
      break;
    case 'y':
      state.y(op.q);
      break;
    case 'z':
      state.z(op.q);
      break;
    case 'rx':
      state.rx(op.q, op.angle);
      break;
    case 'ry':
      state.ry(op.q, op.angle);
      break;
    case 'rz':
      state.rz(op.q, op.angle);
      break;
    case 'cnot':
      state.cnot(op.q1, op.q2);
      break;
    case 'cz':
      state.cz(op.q1, op.q2);
      break;
    case 'swap':
      state.swap(op.q1, op.q2);
      break;
    default:
      throw new Error(`Unknown gate ${op.gate}`);
  }
}

function compareProbabilities(a, b) {
  let maxAbsError = 0;
  for (let i = 0; i < a.length; i++) {
    const absErr = Math.abs(a[i] - b[i]);
    if (absErr > maxAbsError) {
      maxAbsError = absErr;
    }
  }
  return { maxAbsError };
}

async function runCase(rand, caseId) {
  const numQubits = randomInt(rand, 2, 6);
  const depth = randomInt(rand, 4, 36);
  const ops = [];
  for (let i = 0; i < depth; i++) {
    ops.push(sampleGate(rand, numQubits));
  }

  const ref = await QuantumState.create({ numQubits });
  const test = await QuantumState.create({ numQubits });

  try {
    for (const op of ops) {
      applyGate(ref, op);
      applyGate(test, op);
    }

    const refProbs = Array.from(ref.getProbabilities());
    const testProbs = Array.from(test.getProbabilities());
    const metrics = compareProbabilities(refProbs, testProbs);

    return {
      caseId,
      numQubits,
      depth,
      maxAbsError: metrics.maxAbsError,
      passed: metrics.maxAbsError <= MAX_ABS_TOL,
    };
  } finally {
    ref.dispose();
    test.dispose();
  }
}

async function main() {
  const rand = mulberry32(SEED);
  const webgpuAvailable = await isWebGPUAvailable();
  const backendInit = await initializeWebGPUBackend();
  const activeBackend = await getActiveTensorGPUBackend();

  const cases = [];
  let failures = 0;

  for (let i = 0; i < NUM_CASES; i++) {
    const result = await runCase(rand, i + 1);
    cases.push(result);
    if (!result.passed) {
      failures += 1;
    }
  }

  const maxObservedError = cases.reduce((acc, item) => Math.max(acc, item.maxAbsError), 0);
  const backendRequirementFailed = REQUIRE_WEBGPU_BACKEND && activeBackend !== 'webgpu';
  if (backendRequirementFailed) {
    failures += 1;
  }

  const report = {
    generatedAt: new Date().toISOString(),
    seed: SEED,
    numCases: NUM_CASES,
    thresholds: {
      maxAbsError: MAX_ABS_TOL,
    },
    runtime: {
      webgpuAvailable,
      backendInit,
      activeBackend,
    },
    summary: {
      failures,
      maxObservedError,
      backendRequirementFailed,
      passed: failures === 0,
    },
    cases,
  };

  await fs.mkdir(ARTIFACT_DIR, { recursive: true });
  await fs.writeFile(RESULTS_PATH, JSON.stringify(report, null, 2), 'utf8');

  const summary = [
    `Moonlab WebGPU Eval`,
    `Date: ${report.generatedAt}`,
    `Seed: ${SEED}`,
    `Cases: ${NUM_CASES}`,
    `WebGPU runtime available: ${webgpuAvailable}`,
    `Backend initialized: ${backendInit}`,
    `Active tensor GPU backend: ${activeBackend}`,
    `Require active WebGPU backend: ${REQUIRE_WEBGPU_BACKEND}`,
    `Tolerance (max abs error): ${MAX_ABS_TOL}`,
    `Observed max abs error: ${maxObservedError}`,
    `Backend requirement failed: ${backendRequirementFailed}`,
    `Failures: ${failures}`,
    `Result: ${failures === 0 ? 'PASS' : 'FAIL'}`,
  ].join('\n');

  await fs.writeFile(SUMMARY_PATH, summary + '\n', 'utf8');

  if (failures > 0) {
    console.error(summary);
    process.exitCode = 1;
    return;
  }

  console.log(summary);
}

await main();
