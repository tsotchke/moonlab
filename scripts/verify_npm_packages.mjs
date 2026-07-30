#!/usr/bin/env node
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(fileURLToPath(new URL('..', import.meta.url)));
const packDir = resolve(process.argv[2] ?? join(root, 'dist/npm'));
const version = readFileSync(join(root, 'VERSION.txt'), 'utf8').trim();
// Workspace directory -> published npm name. The core package is the
// unsuffixed @tsotchkecorp/moonlab; the other four carry their role as the
// suffix. `npm pack` flattens the scope separator to a dash, so
// @tsotchkecorp/moonlab-viz packs as tsotchkecorp-moonlab-viz-<version>.tgz.
const expected = new Map([
  ['core', '@tsotchkecorp/moonlab'],
  ['algorithms', '@tsotchkecorp/moonlab-algorithms'],
  ['viz', '@tsotchkecorp/moonlab-viz'],
  ['react', '@tsotchkecorp/moonlab-react'],
  ['vue', '@tsotchkecorp/moonlab-vue'],
]);
const tarballs = readdirSync(packDir)
  .filter((name) => name.endsWith('.tgz'))
  .map((name) => join(packDir, name));

assert.equal(tarballs.length, expected.size, 'expected exactly five npm tarballs');

for (const [packageName, registryName] of expected) {
  const suffix = `${registryName.slice(1).replace('/', '-')}-${version}.tgz`;
  const tarball = tarballs.find((candidate) => basename(candidate) === suffix);
  assert.ok(tarball, `missing ${suffix}`);

  const manifest = JSON.parse(execFileSync(
    'tar', ['-xOf', tarball, 'package/package.json'], {
      encoding: 'utf8', env: { ...process.env, LC_ALL: 'C' },
    },
  ));
  assert.equal(manifest.name, registryName);
  assert.equal(manifest.version, version);
  assert.equal(manifest.license, 'MIT');
  assert.equal(manifest.publishConfig?.access, 'public');
  assert.ok(manifest.exports?.['.']?.types, `${manifest.name} lacks typed exports`);

  for (const [dependency, dependencyVersion] of Object.entries(manifest.dependencies ?? {})) {
    if (dependency.startsWith('@tsotchkecorp/')) {
      assert.equal(dependencyVersion, version, `${manifest.name} has a drifting dependency`);
    }
  }

  const contents = execFileSync('tar', ['-tf', tarball], {
    encoding: 'utf8', env: { ...process.env, LC_ALL: 'C' },
  });
  assert.match(contents, /package\/LICENSE(?:\n|$)/);
  assert.match(contents, /package\/dist\/index\.js(?:\n|$)/);
  assert.match(contents, /package\/dist\/index\.mjs(?:\n|$)/);
  assert.match(contents, /package\/dist\/index\.d\.ts(?:\n|$)/);
  if (packageName === 'core') {
    assert.match(contents, /package\/dist\/moonlab\.wasm(?:\n|$)/);
    assert.match(contents, /package\/dist\/moonlab\.js(?:\n|$)/);
  }
}

const consumer = mkdtempSync(join(tmpdir(), 'moonlab-npm-consumer-'));
try {
  writeFileSync(join(consumer, 'package.json'), JSON.stringify({
    name: 'moonlab-release-consumer',
    private: true,
    type: 'module',
  }, null, 2));

  execFileSync('npm', [
    'install', '--no-audit', '--no-fund', '--ignore-scripts', '--legacy-peer-deps',
    '--cache', join(consumer, '.npm-cache'),
    ...tarballs, 'react@18', 'react-dom@18', 'vue@3', 'typescript@5',
    '@types/node@20', '@types/react@18', '@types/react-dom@18',
  ], { cwd: consumer, stdio: 'inherit' });

  writeFileSync(join(consumer, 'esm-smoke.mjs'), `
import assert from 'node:assert/strict';
import * as core from '@tsotchkecorp/moonlab';
import * as algorithms from '@tsotchkecorp/moonlab-algorithms';
import * as viz from '@tsotchkecorp/moonlab-viz';
import * as react from '@tsotchkecorp/moonlab-react';
import * as vue from '@tsotchkecorp/moonlab-vue';
for (const api of [core, algorithms, viz, react, vue]) assert.equal(api.VERSION, '${version}');
assert.equal(viz.indexToBitString(3, 2), '11');
const state = await core.QuantumState.create({ numQubits: 2 });
try {
  state.h(0).cnot(0, 1);
  const probabilities = Array.from(state.getProbabilities());
  assert.ok(probabilities[0] > 0.49 && probabilities[3] > 0.49);
} finally {
  state.dispose();
}
`);

  writeFileSync(join(consumer, 'cjs-smoke.cjs'), `
const assert = require('node:assert/strict');
const core = require('@tsotchkecorp/moonlab');
const algorithms = require('@tsotchkecorp/moonlab-algorithms');
const viz = require('@tsotchkecorp/moonlab-viz');
const react = require('@tsotchkecorp/moonlab-react');
const vue = require('@tsotchkecorp/moonlab-vue');
(async () => {
  for (const api of [core, algorithms, viz, react, vue]) assert.equal(api.VERSION, '${version}');
  const state = await core.QuantumState.create({ numQubits: 1 });
  try {
    state.x(0);
    assert.ok(state.getProbabilities()[1] > 0.99);
  } finally {
    state.dispose();
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
`);

  writeFileSync(join(consumer, 'types-smoke.ts'), `
import { QuantumState, type Complex } from '@tsotchkecorp/moonlab';
import { Grover } from '@tsotchkecorp/moonlab-algorithms';
import { amplitudesToBlochState } from '@tsotchkecorp/moonlab-viz';
import { BlochSphere as ReactBlochSphere } from '@tsotchkecorp/moonlab-react';
import { BlochSphere as VueBlochSphere } from '@tsotchkecorp/moonlab-vue';
const zero: Complex = { real: 1, imag: 0 };
void QuantumState; void Grover; void ReactBlochSphere; void VueBlochSphere;
amplitudesToBlochState([zero, { real: 0, imag: 0 }]);
`);

  execFileSync('node', ['esm-smoke.mjs'], { cwd: consumer, stdio: 'inherit' });
  execFileSync('node', ['cjs-smoke.cjs'], { cwd: consumer, stdio: 'inherit' });
  execFileSync(join(consumer, 'node_modules/.bin/tsc'), [
    '--noEmit', '--strict', '--target', 'ES2020', '--module', 'NodeNext',
    '--moduleResolution', 'NodeNext', 'types-smoke.ts',
  ], { cwd: consumer, stdio: 'inherit' });
} finally {
  rmSync(consumer, { recursive: true, force: true });
}

console.log(`Verified five npm tarballs for Moonlab ${version}`);
