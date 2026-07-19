/**
 * Tests for the Node-side control-plane client (v0.9.4).
 *
 * Uses an in-process TypeScript fake server that mimics the
 * moonlab line protocol.  The real C control plane is exercised by
 * the C `test_control_plane_*` ctest suite and the Python
 * `test_control_plane.py` suite.
 */

import { describe, it, expect } from 'vitest';
import net from 'node:net';
import tls from 'node:tls';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execSync, spawnSync } from 'node:child_process';
import {
  submitCircuit,
  submitShots,
  submitHealth,
  submitMetrics,
  ControlPlaneError,
  MOONLAB_CONTROL_REJECTED,
  MOONLAB_CONTROL_SERVER_BUSY,
} from '../control-plane';

/** Spin up a fake control plane on an OS-chosen port.  Returns a
 *  bound port + close handle. */
async function spinFakeServer(
  onConn: (sock: net.Socket) => void | Promise<void>,
  opts: { tls?: tls.SecureContextOptions } = {},
): Promise<{ port: number; close: () => Promise<void> }> {
  let srv: net.Server;
  const sockets = new Set<net.Socket>();
  const socketClosures = new Map<net.Socket, Promise<void>>();
  const handlers = new Set<Promise<void>>();
  const failures: unknown[] = [];
  const handleConnection = (sock: net.Socket) => {
    sockets.add(sock);
    socketClosures.set(sock, new Promise<void>((resolve) => {
      sock.once('close', () => {
        sockets.delete(sock);
        resolve();
      });
    }));
    // EventEmitter does not await an async connection listener.  Own the
    // promise and every socket error so close() can deterministically surface
    // failures instead of leaking a late ECONNRESET into Vitest.
    sock.on('error', (error) => failures.push(error));
    const handler = Promise.resolve()
      .then(() => onConn(sock))
      .catch((error) => {
        failures.push(error);
        sock.destroy();
      })
      .finally(() => handlers.delete(handler));
    handlers.add(handler);
  };
  if (opts.tls) {
    srv = tls.createServer(opts.tls, (sock) => handleConnection(sock as unknown as net.Socket));
  } else {
    srv = net.createServer(handleConnection);
  }
  await new Promise<void>((res) => srv.listen(0, '127.0.0.1', () => res()));
  const port = (srv.address() as net.AddressInfo).port;
  return {
    port,
    close: async () => {
      const closed = new Promise<void>((resolve, reject) => {
        srv.close((error) => error ? reject(error) : resolve());
      });
      while (handlers.size > 0) {
        await Promise.all([...handlers]);
      }
      await closed;
      await Promise.all(socketClosures.values());
      expect(sockets.size).toBe(0);
      if (failures.length > 0) {
        throw failures[0];
      }
    },
  };
}

function readUntilNewline(sock: net.Socket, cap = 4096): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    let buf = Buffer.alloc(0);
    const onData = (chunk: Buffer) => {
      buf = Buffer.concat([buf, chunk]);
      const nl = buf.indexOf(0x0a);
      if (nl >= 0) {
        sock.removeListener('data', onData);
        sock.removeListener('error', onErr);
        resolve(buf);
        return;
      }
      if (buf.length > cap) {
        sock.removeListener('data', onData);
        sock.removeListener('error', onErr);
        reject(new Error('header overflow'));
      }
    };
    const onErr = (e: Error) => {
      sock.removeListener('data', onData);
      reject(e);
    };
    sock.on('data', onData);
    sock.once('error', onErr);
  });
}

describe('control-plane Node client', () => {
  it('submitCircuit -> probability vector', async () => {
    const server = await spinFakeServer(async (sock) => {
      await readUntilNewline(sock);   // consume CIRCUIT header (+ any inlined body)
      // For the Bell test we don't need to parse the body, just reply
      // with 4 doubles = 32 bytes.  Reply immediately -- the server
      // discards body bytes that arrived in the same chunk as the
      // framing newline.
      const buf = Buffer.alloc(32);
      buf.writeDoubleLE(0.5, 0);
      buf.writeDoubleLE(0.0, 8);
      buf.writeDoubleLE(0.0, 16);
      buf.writeDoubleLE(0.5, 24);
      // CIRCUIT reply framing is "OK <num_doubles>", not byte length.
      sock.write('OK 4\n');
      sock.write(buf);
      sock.end();
    });
    try {
      const probs = await submitCircuit({
        host: '127.0.0.1', port: server.port,
        circuitText: 'MOONLAB_CIRCUIT_V1\nN_QUBITS 2\nGATES 0\nEND\n',
      });
      expect(probs).toHaveLength(4);
      expect(probs[0]).toBeCloseTo(0.5, 10);
      expect(probs[3]).toBeCloseTo(0.5, 10);
    } finally {
      await server.close();
    }
  });

  it('submitMetrics -> Prometheus text body', async () => {
    const body =
      '# HELP moonlab_control_requests_total Requests.\n' +
      '# TYPE moonlab_control_requests_total counter\n' +
      'moonlab_control_requests_total{verb="CIRCUIT"} 0\n' +
      'moonlab_control_max_concurrent_rejected_total 0\n';

    const server = await spinFakeServer(async (sock) => {
      await readUntilNewline(sock);
      const bodyBuf = Buffer.from(body, 'utf-8');
      sock.write(`METRICS ${bodyBuf.length}\n`);
      sock.write(bodyBuf);
      sock.end();
    });
    try {
      const out = await submitMetrics({
        host: '127.0.0.1', port: server.port,
      });
      expect(out).toContain('moonlab_control_requests_total');
      expect(out).toContain('moonlab_control_max_concurrent_rejected_total');
    } finally {
      await server.close();
    }
  });

  it('submitHealth -> resolves on OK', async () => {
    const server = await spinFakeServer(async (sock) => {
      await readUntilNewline(sock);
      sock.write('OK\n');
      sock.end();
    });
    try {
      await submitHealth({ host: '127.0.0.1', port: server.port });
    } finally {
      await server.close();
    }
  });

  it('parses ERR -409 server busy as MOONLAB_CONTROL_SERVER_BUSY', async () => {
    const server = await spinFakeServer(async (sock) => {
      await readUntilNewline(sock);
      sock.write('ERR -409 server busy\n');
      sock.end();
    });
    try {
      await expect(
        submitCircuit({
          host: '127.0.0.1', port: server.port,
          circuitText: 'whatever',
        }),
      ).rejects.toMatchObject({
        name: 'ControlPlaneError',
        statusCode: MOONLAB_CONTROL_SERVER_BUSY,
      });
    } finally {
      await server.close();
    }
  });

  it('parses ERR -405 rejected as MOONLAB_CONTROL_REJECTED', async () => {
    const server = await spinFakeServer(async (sock) => {
      await readUntilNewline(sock);
      sock.write('ERR -405 bad circuit\n');
      sock.end();
    });
    try {
      const err = await submitCircuit({
        host: '127.0.0.1', port: server.port,
        circuitText: 'bad',
      }).catch(e => e);
      expect(err).toBeInstanceOf(ControlPlaneError);
      expect(err.statusCode).toBe(MOONLAB_CONTROL_REJECTED);
    } finally {
      await server.close();
    }
  });

  it('submitShots -> outcome counts', async () => {
    const server = await spinFakeServer(async (sock) => {
      await readUntilNewline(sock);
      const buf = Buffer.alloc(32);
      buf.writeBigUInt64LE(500n,   0);
      buf.writeBigUInt64LE(0n,     8);
      buf.writeBigUInt64LE(0n,    16);
      buf.writeBigUInt64LE(500n,  24);
      // SHOTS reply framing is "SAMPLES <num_outcomes>".
      sock.write('SAMPLES 4\n');
      sock.write(buf);
      sock.end();
    });
    try {
      const counts = await submitShots({
        host: '127.0.0.1', port: server.port,
        circuitText: 'whatever', numShots: 1000,
      });
      expect(counts).toEqual([500, 0, 0, 500]);
    } finally {
      await server.close();
    }
  });

  it('AUTH prelude HMAC matches the verb line including \\n', async () => {
    // The fake server captures the first two lines (AUTH + verb) and
    // verifies them by recomputing the HMAC against the verb line.
    let observedAuth: string | null = null;
    let observedVerb: string | null = null;
    const SECRET = Buffer.from('topsecret');
    const server = await spinFakeServer(async (sock) => {
      const line1 = await readUntilNewline(sock);
      // Strip up to first newline -- may include verb-line bytes too
      const nl = line1.indexOf(0x0a);
      observedAuth = line1.subarray(0, nl).toString('ascii');
      const after = line1.subarray(nl + 1);
      // The verb line might already be in `after`; if not, read more.
      let rest = after;
      const nl2 = rest.indexOf(0x0a);
      if (nl2 < 0) {
        const more = await readUntilNewline(sock);
        rest = Buffer.concat([rest, more]);
      }
      const nl3 = rest.indexOf(0x0a);
      observedVerb = rest.subarray(0, nl3).toString('ascii') + '\n';
      const buf = Buffer.alloc(32);
      buf.writeDoubleLE(0.5, 0); buf.writeDoubleLE(0.5, 24);
      sock.write('OK 4\n');
      sock.write(buf);
      sock.end();
    });
    try {
      await submitCircuit({
        host: '127.0.0.1', port: server.port,
        circuitText: 'whatever',
        secret: SECRET,
      });
    } finally {
      await server.close();
    }
    expect(observedAuth).toMatch(/^AUTH [0-9a-f]{64}$/);
    expect(observedVerb).toBe('CIRCUIT 8\n');

    // Cross-check the hex against an independent HMAC computation.
    const h = (await import('node:crypto'))
      .createHmac('sha3-256', SECRET);
    h.update(Buffer.from(observedVerb!, 'ascii'));
    const expected = h.digest('hex');
    expect(observedAuth).toBe(`AUTH ${expected}`);
  });

  it('connection refused surfaces as a rejected promise', async () => {
    await expect(
      submitHealth({ host: '127.0.0.1', port: 1, timeoutMs: 500 }),
    ).rejects.toBeTruthy();
  });
});

// TLS path: needs an openssl-generated cert + key, which is in
// system PATH on macOS/Linux dev machines.
const HAS_OPENSSL = (() => {
  const r = spawnSync('which', ['openssl']);
  return r.status === 0;
})();

(HAS_OPENSSL ? describe : describe.skip)('control-plane TLS', () => {
  it('TLS scrapes close cleanly without late socket errors', async () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'cp-tls-'));
    const certPath = path.join(tmp, 'cert.pem');
    const keyPath  = path.join(tmp, 'key.pem');
    execSync(
      `openssl req -x509 -newkey rsa:2048 -nodes -keyout ${keyPath} ` +
      `-out ${certPath} -days 1 ` +
      `-subj /CN=127.0.0.1 ` +
      `-addext subjectAltName=IP:127.0.0.1`,
      { stdio: 'ignore' });

    const body = 'moonlab_control_requests_total{verb="HEALTH"} 1\n';
    const server = await spinFakeServer(
      async (sock) => {
        await readUntilNewline(sock);
        const b = Buffer.from(body, 'utf-8');
        sock.write(`METRICS ${b.length}\n`);
        sock.write(b);
        sock.end();
      },
      {
        tls: {
          cert: fs.readFileSync(certPath),
          key: fs.readFileSync(keyPath),
        },
      },
    );
    try {
      // Repetition makes the former destroy-vs-close_notify race deterministic
      // enough to guard: every connection and async server handler is awaited.
      for (let attempt = 0; attempt < 20; attempt++) {
        const out = await submitMetrics({
          host: '127.0.0.1', port: server.port,
          tls: { insecure: true },
        });
        expect(out).toBe(body);
      }
    } finally {
      await server.close();
      fs.rmSync(tmp, { recursive: true });
    }
  });
});
