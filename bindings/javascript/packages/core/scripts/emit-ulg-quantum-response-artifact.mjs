#!/usr/bin/env node
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  buildUlgBellStateArtifact,
  validateUlgQuantumResponseArtifact,
} from '../dist/index.mjs';

const scriptPath = fileURLToPath(import.meta.url);
const coreRoot = resolve(dirname(scriptPath), '..');
const moonlabRoot = resolve(coreRoot, '../../../..');
const defaultSchemaPath = resolve(
  moonlabRoot,
  '..',
  'ulg/ulg-gpu-abi/src/schemas/quantum_response_artifact.schema.json'
);

const args = parseArgs(process.argv.slice(2));
const schemaPath = resolve(args.schema ?? process.env.ULG_QUANTUM_RESPONSE_SCHEMA ?? defaultSchemaPath);
const outputPath = args.out ? resolve(args.out) : undefined;

const artifact = await buildUlgBellStateArtifact({
  schemaPath,
  provenance: {
    cli: relative(moonlabRoot, scriptPath),
    dist: {
      moonlabJs: inspectFile(resolve(coreRoot, 'dist/moonlab.js')),
      moonlabWasm: inspectFile(resolve(coreRoot, 'dist/moonlab.wasm')),
    },
  },
});

if (existsSync(schemaPath)) {
  const schema = JSON.parse(readFileSync(schemaPath, 'utf8'));
  const validation = validateUlgQuantumResponseArtifact(artifact, schema);
  artifact.validation.schemaCompatible = validation.valid;
  artifact.validation.schemaPath = schemaPath;
  artifact.validation.errors = validation.errors;
  artifact.validation.checks = [
    ...artifact.validation.checks.filter((check) => check.name !== 'ulg-schema-file'),
    {
      name: 'ulg-schema-file',
      passed: validation.valid,
      details: {
        schemaPath,
        errors: validation.errors,
      },
    },
  ];
  artifact.provenance.validationSchemaPath = schemaPath;
} else {
  artifact.validation.checks = [
    ...artifact.validation.checks,
    {
      name: 'ulg-schema-file',
      passed: false,
      details: {
        schemaPath,
        errors: ['schema file not found'],
      },
    },
  ];
}

const json = `${JSON.stringify(artifact, null, 2)}\n`;
if (outputPath) {
  mkdirSync(dirname(outputPath), { recursive: true });
  writeFileSync(outputPath, json);
} else {
  process.stdout.write(json);
}

if (!artifact.validation.schemaCompatible || artifact.parity.passed !== true) {
  process.exitCode = 1;
}

function parseArgs(argv) {
  const parsed = {};
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--') {
      continue;
    } else if (arg === '--schema') {
      parsed.schema = argv[++index];
    } else if (arg === '--out') {
      parsed.out = argv[++index];
    } else if (arg === '--help' || arg === '-h') {
      process.stdout.write(
        'Usage: node scripts/emit-ulg-quantum-response-artifact.mjs [--schema path] [--out path]\n'
      );
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return parsed;
}

function inspectFile(path) {
  if (!existsSync(path)) {
    return { path, present: false };
  }
  const stat = statSync(path);
  return {
    path,
    present: true,
    bytes: stat.size,
    modifiedAt: stat.mtime.toISOString(),
  };
}
