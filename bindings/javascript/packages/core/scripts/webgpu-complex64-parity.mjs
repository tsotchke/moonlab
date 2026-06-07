#!/usr/bin/env node
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import {
  buildMoonlabWebGpuComplex64ParityScopeWithBrowserProbe,
  canonicalJson,
  summarizeMoonlabWebGpuComplex64ParityScope,
  validateMoonlabWebGpuComplex64ParityScope,
} from '../dist/index.mjs';

const args = parseArgs(process.argv.slice(2));
const requireBackend =
  args.requireBackend || process.env.MOONLAB_WEBGPU_PARITY_REQUIRE_BACKEND === '1';
const artifact = await buildMoonlabWebGpuComplex64ParityScopeWithBrowserProbe({
  generatedAt: args.generatedAt,
  requireBackend,
});
const validation = validateMoonlabWebGpuComplex64ParityScope(artifact);
const artifactWithValidation = {
  ...artifact,
  contractValidation: validation,
};
const outputValue = args.summary
  ? summarizeMoonlabWebGpuComplex64ParityScope(artifactWithValidation)
  : artifactWithValidation;
const outputPath = args.out ? resolve(args.out) : undefined;

writeJson(outputValue, outputPath, { canonical: args.canonical });

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
    } else if (arg === '--summary') {
      parsed.summary = true;
    } else if (arg === '--generated-at') {
      parsed.generatedAt = argv[++index];
    } else if (arg === '--require-backend') {
      parsed.requireBackend = true;
    } else if (arg === '--help' || arg === '-h') {
      process.stdout.write(
        'Usage: node scripts/webgpu-complex64-parity.mjs [--out path] [--canonical] [--summary] [--generated-at iso] [--require-backend]\n'
      );
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return parsed;
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
