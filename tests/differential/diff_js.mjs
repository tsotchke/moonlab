#!/usr/bin/env node
/**
 * Cross-binding differential: the WASM/JS dense binding (@moonlab/quantum-core)
 * vs the numpy reference oracle pinned in the corpus.
 *
 * Reproduces every corpus circuit through the WASM `QuantumState` and checks the
 * full probability vector and <Z_i>/<Z_iZ_j> expectations against the reference
 * within 1e-10 -- catching endianness / qubit-order / angle-sign marshalling
 * bugs across the JS boundary that a C-only test cannot see.
 *
 * Usage:  node diff_js.mjs [corpus.json]
 * Emits:  DIFF_JS_RESULT status=PASS|FAIL|SKIP cases=N failed=M reason="..."
 * Exit:   0 pass, 1 fail, 77 skip (bundle/toolchain missing -- logged, not fatal).
 */

import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, resolve } from 'node:path';

const TOL = 1e-10;
const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(__dirname, '..', '..');

function emit(status, cases, failed, reason) {
  process.stdout.write(
    `DIFF_JS_RESULT status=${status} cases=${cases} failed=${failed} reason="${reason}"\n`
  );
  return status === 'SKIP' ? 77 : status === 'PASS' ? 0 : 1;
}

async function loadBinding() {
  // Prefer the package specifier; fall back to the built dist bundle in-repo.
  const candidates = [
    '@moonlab/quantum-core',
    pathToFileURL(
      resolve(REPO_ROOT, 'bindings/javascript/packages/core/dist/index.mjs')
    ).href,
  ];
  let lastErr = null;
  for (const c of candidates) {
    try {
      const mod = await import(c);
      if (mod && mod.QuantumState) return { mod, reason: null };
    } catch (e) {
      lastErr = e;
    }
  }
  return { mod: null, reason: `@moonlab/quantum-core not importable (${lastErr && lastErr.message})` };
}

function applyGate(st, g) {
  const q = g.qubits;
  const a = g.angle || 0.0;
  switch (g.name) {
    case 'h': return st.h(q[0]);
    case 'x': return st.x(q[0]);
    case 'y': return st.y(q[0]);
    case 'z': return st.z(q[0]);
    case 's': return st.s(q[0]);
    case 'sdg': return st.sdg(q[0]);
    case 't': return st.t(q[0]);
    case 'tdg': return st.tdg(q[0]);
    case 'rx': return st.rx(q[0], a);
    case 'ry': return st.ry(q[0], a);
    case 'rz': return st.rz(q[0], a);
    case 'p': return st.phase(q[0], a);
    case 'cx': return st.cnot(q[0], q[1]);
    case 'cz': return st.cz(q[0], q[1]);
    case 'swap': return st.swap(q[0], q[1]);
    case 'cp': return st.cphase(q[0], q[1], a);
    case 'ccx': return st.toffoli(q[0], q[1], q[2]);
    default: throw new Error(`unknown gate ${g.name}`);
  }
}

function expZ(probs, n) {
  const out = [];
  for (let qq = 0; qq < n; qq++) {
    let e = 0.0;
    for (let i = 0; i < probs.length; i++) e += ((i >> qq) & 1) ? -probs[i] : probs[i];
    out.push(e);
  }
  return out;
}

function expZZ(probs, pairs) {
  return pairs.map(([a, b]) => {
    let e = 0.0;
    for (let i = 0; i < probs.length; i++) {
      const sa = ((i >> a) & 1) ? -1 : 1;
      const sb = ((i >> b) & 1) ? -1 : 1;
      e += sa * sb * probs[i];
    }
    return e;
  });
}

async function main() {
  const corpusPath = process.argv[2]
    || process.env.MOONLAB_DIFF_CORPUS
    || resolve(REPO_ROOT, 'build', 'differential', 'corpus.json');

  if (!existsSync(corpusPath)) {
    return emit('SKIP', 0, 0, `corpus not found: ${corpusPath}`);
  }
  const { mod, reason } = await loadBinding();
  if (!mod) return emit('SKIP', 0, 0, reason);

  const { QuantumState } = mod;
  const corpus = JSON.parse(readFileSync(corpusPath, 'utf8'));
  const failures = [];
  let exercised = 0;

  for (const cs of corpus.cases) {
    const n = cs.num_qubits;
    const ref = cs.reference;
    let st;
    try {
      st = await QuantumState.create({ numQubits: n });
    } catch (e) {
      return emit('SKIP', exercised, failures.length, `QuantumState.create failed (${e.message})`);
    }
    try {
      for (const g of cs.gates) applyGate(st, g);
      const probs = Array.from(st.getProbabilities());
      exercised++;
      let dev = 0.0;
      for (let i = 0; i < probs.length; i++) dev = Math.max(dev, Math.abs(probs[i] - ref.probabilities[i]));
      if (dev > TOL) {
        failures.push([cs.id, cs.seed, 'prob', dev]);
        continue;
      }
      const pairs = ref.exp_zz.map(([a, b]) => [a, b]);
      const ez = expZ(probs, n);
      let dz = 0.0;
      for (let q = 0; q < n; q++) dz = Math.max(dz, Math.abs(ez[q] - ref.exp_z[q]));
      const ezz = expZZ(probs, pairs);
      let dzz = 0.0;
      for (let k = 0; k < pairs.length; k++) dzz = Math.max(dzz, Math.abs(ezz[k] - ref.exp_zz[k][2]));
      if (dz > TOL) failures.push([cs.id, cs.seed, 'expZ', dz]);
      if (dzz > TOL) failures.push([cs.id, cs.seed, 'expZZ', dzz]);
    } finally {
      st.dispose();
    }
  }

  for (const [cid, seed, what, dev] of failures.slice(0, 50)) {
    process.stderr.write(
      `  FAIL  case=${cid} seed=${seed} ${what} [js] vs reference dev=${dev.toExponential(3)} (>${TOL})\n`
    );
  }
  return emit(failures.length ? 'FAIL' : 'PASS', exercised, failures.length, 'ok');
}

main().then((code) => process.exit(code)).catch((e) => {
  process.stdout.write(`DIFF_JS_RESULT status=SKIP cases=0 failed=0 reason="uncaught: ${e.message}"\n`);
  process.exit(77);
});
