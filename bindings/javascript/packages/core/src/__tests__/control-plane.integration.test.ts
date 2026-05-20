/**
 * Cross-language integration: JS client <-> real moonlab C control
 * plane.  Spawns a Python subprocess that runs
 * ``moonlab.control_plane.ControlPlaneServer`` (which talks to
 * libquantumsim), reads the bound port from its stdout, and drives
 * the JS client against it.
 *
 * Skipped automatically when:
 *   - libquantumsim isn't built (no ./build-mpi/libquantumsim.dylib),
 *   - or python3 + the bindings/python tree isn't reachable.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { spawn, ChildProcessWithoutNullStreams } from 'node:child_process';
import path from 'node:path';
import fs from 'node:fs';
import {
  submitCircuit,
  submitHealth,
  submitMetrics,
} from '../control-plane';

// ---------------------------------------------------------------------------
// Locate the repo root + the libquantumsim build dir.
// ---------------------------------------------------------------------------
function findRepoRoot(): string | null {
  let dir = path.resolve(__dirname);
  for (let i = 0; i < 10; i++) {
    if (fs.existsSync(path.join(dir, 'VERSION.txt')) &&
        fs.existsSync(path.join(dir, 'bindings'))) {
      return dir;
    }
    const up = path.dirname(dir);
    if (up === dir) return null;
    dir = up;
  }
  return null;
}

const REPO = findRepoRoot();
const LIB_DIR = REPO ? path.join(REPO, 'build-mpi') : null;
const LIB_PATH = LIB_DIR ? path.join(LIB_DIR, 'libquantumsim.dylib') : null;
const PYTHON_BINDINGS = REPO ? path.join(REPO, 'bindings', 'python') : null;

const CAN_RUN =
  REPO !== null &&
  LIB_PATH !== null && fs.existsSync(LIB_PATH) &&
  PYTHON_BINDINGS !== null && fs.existsSync(PYTHON_BINDINGS);

const HARNESS_PY = `
import sys, time, os
sys.path.insert(0, ${JSON.stringify(PYTHON_BINDINGS)})
from moonlab.control_plane import ControlPlaneServer
srv = ControlPlaneServer(host="127.0.0.1", port=0)
srv.__enter__()
print(srv.port, flush=True)
try:
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    pass
finally:
    srv.__exit__(None, None, None)
`;

(CAN_RUN ? describe : describe.skip)('control-plane JS <-> real C server', () => {
  let proc: ChildProcessWithoutNullStreams;
  let port = 0;

  beforeAll(async () => {
    proc = spawn('python3', ['-c', HARNESS_PY], {
      env: {
        ...process.env,
        DYLD_LIBRARY_PATH: LIB_DIR!,
        MOONLAB_LIB: LIB_PATH!,
      },
    });
    port = await new Promise<number>((resolve, reject) => {
      let buf = '';
      const timer = setTimeout(() => reject(
        new Error('python harness did not print a port within 5s')), 5000);
      proc.stdout.on('data', (chunk: Buffer) => {
        buf += chunk.toString('utf-8');
        const nl = buf.indexOf('\n');
        if (nl >= 0) {
          const n = parseInt(buf.slice(0, nl).trim(), 10);
          if (Number.isFinite(n) && n > 0) {
            clearTimeout(timer);
            resolve(n);
          }
        }
      });
      proc.stderr.on('data', (c) => { /* surface for debugging */
        process.stderr.write(`[harness] ${c.toString('utf-8')}`);
      });
      proc.once('exit', (code) => {
        clearTimeout(timer);
        reject(new Error(`python harness exited early (code=${code})`));
      });
    });
    // Give the server thread a moment to settle into accept().
    await new Promise((r) => setTimeout(r, 50));
  }, 10_000);

  afterAll(async () => {
    if (proc && !proc.killed) {
      proc.kill('SIGINT');
      await new Promise((res) => proc.once('exit', res));
    }
  });

  it('HEALTH succeeds against the real server', async () => {
    await submitHealth({ host: '127.0.0.1', port });
  });

  it('METRICS scrape contains v0.9.0 counters', async () => {
    const body = await submitMetrics({ host: '127.0.0.1', port });
    expect(body).toContain('moonlab_control_max_concurrent_rejected_total');
    expect(body).toContain('moonlab_control_tls_handshake_failed_total');
  });

  it('CIRCUIT bell-pair returns the right probabilities', async () => {
    const text = [
      '# moonlab-circuit v1',
      'NUM_QUBITS 2',
      'H 0',
      'CNOT 1 0',
      '',
    ].join('\n');
    const probs = await submitCircuit({
      host: '127.0.0.1', port, circuitText: text,
    });
    expect(probs).toHaveLength(4);
    expect(probs[0]).toBeCloseTo(0.5, 10);
    expect(probs[1]).toBeCloseTo(0.0, 10);
    expect(probs[2]).toBeCloseTo(0.0, 10);
    expect(probs[3]).toBeCloseTo(0.5, 10);
  });
});
