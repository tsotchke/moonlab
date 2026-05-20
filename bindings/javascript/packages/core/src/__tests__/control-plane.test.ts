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
  onConn: (sock: net.Socket) => void,
  opts: { tls?: tls.SecureContextOptions } = {},
): Promise<{ port: number; close: () => Promise<void> }> {
  let srv: net.Server;
  if (opts.tls) {
    srv = tls.createServer(opts.tls, (sock) => onConn(sock as unknown as net.Socket));
  } else {
    srv = net.createServer((sock) => onConn(sock));
  }
  await new Promise<void>((res) => srv.listen(0, '127.0.0.1', () => res()));
  const port = (srv.address() as net.AddressInfo).port;
  return {
    port,
    close: () =>
      new Promise<void>((res) => srv.close(() => res())),
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
  it('TLS scrape with insecure skip', async () => {
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
      const out = await submitMetrics({
        host: '127.0.0.1', port: server.port,
        tls: { insecure: true },
      });
      expect(out).toBe(body);
    } finally {
      await server.close();
      fs.rmSync(tmp, { recursive: true });
    }
  });
});
