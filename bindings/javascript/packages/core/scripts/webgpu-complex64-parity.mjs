#!/usr/bin/env node
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import {
  buildMoonlabWebGpuComplex64ParityScope,
  canonicalJson,
  validateMoonlabWebGpuComplex64ParityScope,
} from '../dist/index.mjs';

const args = parseArgs(process.argv.slice(2));
const requireBackend =
  args.requireBackend || process.env.MOONLAB_WEBGPU_PARITY_REQUIRE_BACKEND === '1';
const backendDetection = await detectBrowserWebGpuBackend();
const artifact = buildMoonlabWebGpuComplex64ParityScope({
  generatedAt: args.generatedAt,
  backendDetection,
  requireBackend,
});
const validation = validateMoonlabWebGpuComplex64ParityScope(artifact);
const outputPath = args.out ? resolve(args.out) : undefined;

writeJson({
  ...artifact,
  contractValidation: validation,
}, outputPath, { canonical: args.canonical });

if (!validation.valid || (requireBackend && !artifact.webgpuParity.executed)) {
  process.exitCode = 1;
}

function parseArgs(argv) {
  const parsed = {};
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--') {
      continue;
    } else if (arg === '--out') {
      parsed.out = argv[++index];
    } else if (arg === '--canonical') {
      parsed.canonical = true;
    } else if (arg === '--generated-at') {
      parsed.generatedAt = argv[++index];
    } else if (arg === '--require-backend') {
      parsed.requireBackend = true;
    } else if (arg === '--help' || arg === '-h') {
      process.stdout.write(
        'Usage: node scripts/webgpu-complex64-parity.mjs [--out path] [--canonical] [--generated-at iso] [--require-backend]\n'
      );
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return parsed;
}

async function detectBrowserWebGpuBackend() {
  const nav = globalThis.navigator;
  const runtime = detectRuntime();
  const gpu = nav && typeof nav === 'object' && 'gpu' in nav ? nav.gpu : undefined;
  if (!gpu || typeof gpu.requestAdapter !== 'function') {
    return {
      available: false,
      runtime,
      reason: 'navigator.gpu.requestAdapter is unavailable in this JavaScript runtime',
    };
  }

  try {
    const adapter = await gpu.requestAdapter();
    if (!adapter) {
      return {
        available: false,
        runtime,
        reason: 'navigator.gpu.requestAdapter returned no adapter',
      };
    }

    return {
      available: true,
      runtime,
      reason:
        'browser WebGPU adapter detected; native MoonLab WebGPU kernels are not wired on this branch',
      adapterInfo: await readAdapterInfo(adapter),
    };
  } catch (error) {
    return {
      available: false,
      runtime,
      reason: `navigator.gpu.requestAdapter failed: ${
        error instanceof Error ? error.message : String(error)
      }`,
    };
  }
}

async function readAdapterInfo(adapter) {
  if (typeof adapter.requestAdapterInfo === 'function') {
    try {
      const info = await adapter.requestAdapterInfo();
      return plainObject(info);
    } catch {
      return { available: false, reason: 'adapter.requestAdapterInfo failed' };
    }
  }
  return {
    available: false,
    reason: 'adapter.requestAdapterInfo unavailable',
  };
}

function plainObject(value) {
  if (!value || typeof value !== 'object') {
    return {};
  }
  return Object.fromEntries(
    Object.entries(value).filter(([, entryValue]) => entryValue !== undefined)
  );
}

function detectRuntime() {
  if (typeof window !== 'undefined' && typeof document !== 'undefined') {
    return 'browser';
  }
  if (typeof process !== 'undefined' && process.versions?.node) {
    return `node-${process.versions.node}`;
  }
  return 'unknown-js-runtime';
}

function writeJson(value, outputPath, options = {}) {
  const jsonBody = options.canonical ? canonicalJson(value) : JSON.stringify(value, null, 2);
  const json = `${jsonBody}\n`;
  if (outputPath) {
    mkdirSync(dirname(outputPath), { recursive: true });
    writeFileSync(outputPath, json);
  } else {
    process.stdout.write(json);
  }
}
