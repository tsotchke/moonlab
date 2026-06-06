#!/usr/bin/env node
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  buildUlgBellStateArtifact,
  buildUlgMagnetarDipoleIsingArtifact,
  normalizeMagnetarReferenceContractSuite,
  validateMagnetarReferenceContracts,
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
const validationReferencesPath = args.validateReferences
  ? resolve(args.validateReferences)
  : undefined;
if (validationReferencesPath) {
  const report = validateMagnetarReferenceContracts(
    JSON.parse(readFileSync(validationReferencesPath, 'utf8'))
  );
  writeJson(report, args.out ? resolve(args.out) : undefined);
  if (args.strict && !report.ready) {
    process.exitCode = 1;
  }
  process.exit();
}
const normalizeReferencesPath = args.normalizeReferences
  ? resolve(args.normalizeReferences)
  : undefined;
if (normalizeReferencesPath) {
  const suite = normalizeMagnetarReferenceContractSuite(
    JSON.parse(readFileSync(normalizeReferencesPath, 'utf8'))
  );
  writeJson(suite, args.out ? resolve(args.out) : undefined);
  if (args.strict && !suite.ready) {
    process.exitCode = 1;
  }
  process.exit();
}

const probe = args.probe ?? 'bell-state';
if (!['bell-state', 'magnetar-dipole-ising'].includes(probe)) {
  throw new Error(`Unknown probe: ${probe}`);
}
const schemaPath = resolve(args.schema ?? process.env.ULG_QUANTUM_RESPONSE_SCHEMA ?? defaultSchemaPath);
const outputPath = args.out ? resolve(args.out) : undefined;
const referencesPath = args.references ? resolve(args.references) : undefined;
const references = referencesPath ? readReferenceContracts(referencesPath) : undefined;

const provenance = {
  cli: relative(moonlabRoot, scriptPath),
  dist: {
    moonlabJs: inspectFile(resolve(coreRoot, 'dist/moonlab.js')),
    moonlabWasm: inspectFile(resolve(coreRoot, 'dist/moonlab.wasm')),
  },
  referenceContracts: referencesPath ? inspectFile(referencesPath) : undefined,
};

const artifact = probe === 'magnetar-dipole-ising'
  ? await buildUlgMagnetarDipoleIsingArtifact({
      schemaPath,
      provenance,
      references,
    })
  : await buildUlgBellStateArtifact({
      schemaPath,
      provenance,
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

writeJson(artifact, outputPath);

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
    } else if (arg === '--probe') {
      parsed.probe = argv[++index];
    } else if (arg === '--references') {
      parsed.references = argv[++index];
    } else if (arg === '--validate-references') {
      parsed.validateReferences = argv[++index];
    } else if (arg === '--normalize-references') {
      parsed.normalizeReferences = argv[++index];
    } else if (arg === '--strict') {
      parsed.strict = true;
    } else if (arg === '--help' || arg === '-h') {
      process.stdout.write(
        'Usage: node scripts/emit-ulg-quantum-response-artifact.mjs [--probe bell-state|magnetar-dipole-ising] [--schema path] [--out path] [--references path] [--validate-references path] [--normalize-references path] [--strict]\n'
      );
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return parsed;
}

function readReferenceContracts(path) {
  const parsed = JSON.parse(readFileSync(path, 'utf8'));
  if (Array.isArray(parsed)) return parsed;
  if (Array.isArray(parsed?.references)) return parsed.references;
  if (Array.isArray(parsed?.outputs?.references)) return parsed.outputs.references;
  throw new Error(`Reference contract file must be an array or contain references[]: ${path}`);
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

function writeJson(value, outputPath) {
  const json = `${JSON.stringify(value, null, 2)}\n`;
  if (outputPath) {
    mkdirSync(dirname(outputPath), { recursive: true });
    writeFileSync(outputPath, json);
  } else {
    process.stdout.write(json);
  }
}
