/**
 * Control-plane client -- since v0.9.4.
 *
 * Node.js client for the Moonlab control plane line protocol.  This
 * binding is Node-only: browsers cannot open raw TCP sockets, so a
 * browser-side moonlab client requires a separate WebSocket gateway
 * (out of scope for this module).
 *
 * Mirrors the Python (`moonlab.control_plane`) and Rust
 * (`moonlab::control_plane`) client surfaces: `submitCircuit`,
 * `submitShots`, `submitHealth`, `submitMetrics`, with both plain TCP
 * and TLS / mTLS transports.
 *
 * @example
 * ```typescript
 * import { submitCircuit } from '@moonlab/quantum-core/control-plane';
 *
 * const circuit = "MOONLAB_CIRCUIT_V1\nN_QUBITS 2\nGATES 2\n" +
 *                 "GATE H 0 -1\nGATE CNOT 1 0\nEND\n";
 * const probs = await submitCircuit({
 *   host: '127.0.0.1', port: 7070, circuitText: circuit,
 * });
 * console.log(probs);   // [0.5, 0, 0, 0.5]
 * ```
 */

import net from 'node:net';
import tls from 'node:tls';
import fs from 'node:fs';
import type { Socket } from 'node:net';

/** Status codes mirror the C control_plane.h enum.  Negative on
 *  failure; matched 1:1 with `MOONLAB_CONTROL_*`. */
export const MOONLAB_CONTROL_OK            =    0;
export const MOONLAB_CONTROL_BAD_ARG       = -400;
export const MOONLAB_CONTROL_AUTH_REQUIRED = -401;
export const MOONLAB_CONTROL_AUTH_BAD      = -402;
export const MOONLAB_CONTROL_IO_ERROR      = -403;
export const MOONLAB_CONTROL_REJECTED      = -405;
export const MOONLAB_CONTROL_OOM           = -407;
export const MOONLAB_CONTROL_RATE_LIMITED  = -408;
export const MOONLAB_CONTROL_SERVER_BUSY   = -409;

/** Carries the rejection reason from the server's `ERR <code> <msg>`
 *  line, parsed into `.statusCode` and `.message`. */
export class ControlPlaneError extends Error {
  readonly statusCode: number;
  constructor(message: string, statusCode: number) {
    super(message);
    this.name = 'ControlPlaneError';
    this.statusCode = statusCode;
  }
}

/** Optional TLS settings.  `caPath` verifies the server cert;
 *  `clientCertPath`+`clientKeyPath` enable mTLS; `insecure` skips
 *  server-cert verification (development only). */
export interface TlsOptions {
  caPath?: string;
  clientCertPath?: string;
  clientKeyPath?: string;
  insecure?: boolean;
  serverName?: string;
}

export interface SubmitCircuitArgs {
  host: string;
  port: number;
  circuitText: string;
  /** Per-request socket timeout in milliseconds.  Default 30s. */
  timeoutMs?: number;
  tls?: TlsOptions;
}

export interface SubmitShotsArgs extends SubmitCircuitArgs {
  numShots: number;
}

export interface SubmitMetricsArgs {
  host: string;
  port: number;
  timeoutMs?: number;
  tls?: TlsOptions;
}

// ---------------------------------------------------------------------------
// Transport helpers
// ---------------------------------------------------------------------------

function openSocket(args: { host: string; port: number; timeoutMs: number;
                            tls?: TlsOptions }): Promise<Socket> {
  return new Promise<Socket>((resolve, reject) => {
    const { host, port, timeoutMs } = args;
    let sock: Socket;
    if (args.tls) {
      const opts: tls.ConnectionOptions = {
        host, port,
        timeout: timeoutMs,
        servername: args.tls.serverName ?? host,
      };
      if (args.tls.caPath) {
        opts.ca = fs.readFileSync(args.tls.caPath);
      }
      if (args.tls.clientCertPath && args.tls.clientKeyPath) {
        opts.cert = fs.readFileSync(args.tls.clientCertPath);
        opts.key = fs.readFileSync(args.tls.clientKeyPath);
      }
      if (args.tls.insecure) {
        opts.rejectUnauthorized = false;
        opts.checkServerIdentity = () => undefined;
      }
      const tlsSock = tls.connect(opts, () => {
        tlsSock.setNoDelay(true);
        resolve(tlsSock as unknown as Socket);
      });
      tlsSock.once('error', reject);
      tlsSock.once('timeout', () => {
        tlsSock.destroy();
        reject(new Error(`control-plane TLS connect timeout (${timeoutMs}ms)`));
      });
      sock = tlsSock as unknown as Socket;
    } else {
      sock = net.connect({ host, port, timeout: timeoutMs }, () => {
        sock.setNoDelay(true);
        resolve(sock);
      });
      sock.once('error', reject);
      sock.once('timeout', () => {
        sock.destroy();
        reject(new Error(`control-plane connect timeout (${timeoutMs}ms)`));
      });
    }
  });
}

/** Parsed framing line.  Note that the integer field differs by verb:
 *  - ``OK <count>``      -- count of doubles in the body
 *  - ``SAMPLES <count>`` -- count of uint64 outcomes in the body
 *  - ``METRICS <bytes>`` -- byte length of the body
 *  - ``ERR <code> <msg>`` -- numeric status code + message
 *
 *  Callers therefore interpret `numField` per verb. */
interface ReplyFraming {
  verb: string;
  numField: number;
  remainder: string;
  bodySeed: Buffer;
}

function readUntilNewline(sock: Socket, timeoutMs: number,
                          maxHeaderBytes = 4096): Promise<{ line: Buffer; rest: Buffer }> {
  return new Promise((resolve, reject) => {
    let buf = Buffer.alloc(0);
    const onData = (chunk: Buffer) => {
      buf = Buffer.concat([buf, chunk]);
      const nl = buf.indexOf(0x0a);
      if (nl >= 0) {
        sock.removeListener('data', onData);
        sock.removeListener('error', onError);
        sock.removeListener('end', onEnd);
        const line = buf.subarray(0, nl);
        const rest = buf.subarray(nl + 1);
        resolve({ line, rest });
        return;
      }
      if (buf.length > maxHeaderBytes) {
        sock.removeListener('data', onData);
        sock.removeListener('error', onError);
        sock.removeListener('end', onEnd);
        reject(new Error(
          `control-plane header exceeds ${maxHeaderBytes} bytes`));
      }
    };
    const onError = (e: Error) => {
      sock.removeListener('data', onData);
      sock.removeListener('end', onEnd);
      reject(e);
    };
    const onEnd = () => {
      sock.removeListener('data', onData);
      sock.removeListener('error', onError);
      reject(new Error('control-plane closed during header read'));
    };
    sock.on('data', onData);
    sock.once('error', onError);
    sock.once('end', onEnd);
    sock.setTimeout(timeoutMs, () => {
      sock.removeListener('data', onData);
      sock.removeListener('error', onError);
      sock.removeListener('end', onEnd);
      reject(new Error(`control-plane header read timeout (${timeoutMs}ms)`));
    });
  });
}

function recvExact(sock: Socket, want: number, seed: Buffer,
                   timeoutMs: number): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    let buf = seed;
    if (buf.length >= want) {
      resolve(buf.subarray(0, want));
      return;
    }
    const onData = (chunk: Buffer) => {
      buf = Buffer.concat([buf, chunk]);
      if (buf.length >= want) {
        sock.removeListener('data', onData);
        sock.removeListener('error', onError);
        sock.removeListener('end', onEnd);
        resolve(buf.subarray(0, want));
      }
    };
    const onError = (e: Error) => {
      sock.removeListener('data', onData);
      sock.removeListener('end', onEnd);
      reject(e);
    };
    const onEnd = () => {
      sock.removeListener('data', onData);
      sock.removeListener('error', onError);
      reject(new Error(
        `control-plane closed during body read (${buf.length}/${want} bytes)`));
    };
    sock.on('data', onData);
    sock.once('error', onError);
    sock.once('end', onEnd);
    sock.setTimeout(timeoutMs, () => {
      sock.removeListener('data', onData);
      sock.removeListener('error', onError);
      sock.removeListener('end', onEnd);
      reject(new Error(`control-plane body read timeout (${timeoutMs}ms)`));
    });
  });
}

function parseHeader(line: Buffer, rest: Buffer): ReplyFraming {
  const text = line.toString('ascii');
  const sp = text.indexOf(' ');
  if (sp < 0) {
    return { verb: text, numField: -1, remainder: '', bodySeed: rest };
  }
  const verb = text.slice(0, sp);
  const remainder = text.slice(sp + 1);
  const n = parseInt(remainder, 10);
  return {
    verb,
    numField: Number.isFinite(n) && n >= 0 ? n : -1,
    remainder,
    bodySeed: rest,
  };
}

function rejectionFromErr(remainder: string): ControlPlaneError {
  // remainder is "<code> <msg>" from the server's "ERR <code> <msg>\n".
  const sp = remainder.indexOf(' ');
  const codeStr = sp >= 0 ? remainder.slice(0, sp) : remainder;
  const msg     = sp >= 0 ? remainder.slice(sp + 1) : '';
  const code = parseInt(codeStr, 10);
  if (!Number.isFinite(code)) {
    return new ControlPlaneError(`server rejected: ${remainder}`,
                                  MOONLAB_CONTROL_REJECTED);
  }
  return new ControlPlaneError(`server rejected: ${msg || codeStr}`, code);
}

// ---------------------------------------------------------------------------
// Public client API
// ---------------------------------------------------------------------------

/** Submit a moonlab-circuit-v1 text payload, return the probability
 *  vector (length 2^N). */
export async function submitCircuit(args: SubmitCircuitArgs): Promise<number[]> {
  const timeoutMs = args.timeoutMs ?? 30_000;
  const sock = await openSocket({ ...args, timeoutMs });
  try {
    const body = Buffer.from(args.circuitText, 'utf-8');
    const header = `CIRCUIT ${body.length}\n`;
    sock.write(header);
    sock.write(body);
    const { line, rest } = await readUntilNewline(sock, timeoutMs);
    const fr = parseHeader(line, rest);
    if (fr.verb === 'ERR') throw rejectionFromErr(fr.remainder);
    if (fr.verb !== 'OK' || fr.numField < 0) {
      throw new ControlPlaneError(
        `unexpected framing: ${line.toString('ascii')}`,
        MOONLAB_CONTROL_IO_ERROR);
    }
    // OK numField = count of doubles, body is numField * 8 bytes.
    const num = fr.numField;
    const raw = await recvExact(sock, num * 8, fr.bodySeed, timeoutMs);
    const out: number[] = new Array(num);
    for (let i = 0; i < num; i++) {
      out[i] = raw.readDoubleLE(i * 8);
    }
    return out;
  } finally {
    sock.destroy();
  }
}

/** Submit a moonlab-circuit-v1 text payload + shot count, return the
 *  flat outcome-count array (length 2^N). */
export async function submitShots(args: SubmitShotsArgs): Promise<number[]> {
  const timeoutMs = args.timeoutMs ?? 30_000;
  const sock = await openSocket({ ...args, timeoutMs });
  try {
    const body = Buffer.from(args.circuitText, 'utf-8');
    const header = `SHOTS ${args.numShots} ${body.length}\n`;
    sock.write(header);
    sock.write(body);
    const { line, rest } = await readUntilNewline(sock, timeoutMs);
    const fr = parseHeader(line, rest);
    if (fr.verb === 'ERR') throw rejectionFromErr(fr.remainder);
    if (fr.verb !== 'SAMPLES' || fr.numField < 0) {
      throw new ControlPlaneError(
        `unexpected framing: ${line.toString('ascii')}`,
        MOONLAB_CONTROL_IO_ERROR);
    }
    // SAMPLES numField = count of uint64 outcomes, body is numField * 8 bytes.
    const num = fr.numField;
    const raw = await recvExact(sock, num * 8, fr.bodySeed, timeoutMs);
    const out: number[] = new Array(num);
    for (let i = 0; i < num; i++) {
      // The C side packs uint64 little-endian; for shot counts that
      // fit in 2^53 (16 PB shots) this Number conversion is exact.
      out[i] = Number(raw.readBigUInt64LE(i * 8));
    }
    return out;
  } finally {
    sock.destroy();
  }
}

/** Liveness probe.  Resolves on success, rejects on failure. */
export async function submitHealth(args: SubmitMetricsArgs): Promise<void> {
  const timeoutMs = args.timeoutMs ?? 5_000;
  const sock = await openSocket({ ...args, timeoutMs });
  try {
    sock.write('HEALTH\n');
    const { line } = await readUntilNewline(sock, timeoutMs);
    const fr = parseHeader(line, Buffer.alloc(0));
    if (fr.verb === 'OK') return;
    if (fr.verb === 'ERR') throw rejectionFromErr(fr.remainder);
    throw new ControlPlaneError(
      `unexpected framing: ${line.toString('ascii')}`,
      MOONLAB_CONTROL_IO_ERROR);
  } finally {
    sock.destroy();
  }
}

/** Scrape the Prometheus text-format METRICS body. */
export async function submitMetrics(args: SubmitMetricsArgs): Promise<string> {
  const timeoutMs = args.timeoutMs ?? 5_000;
  const sock = await openSocket({ ...args, timeoutMs });
  try {
    sock.write('METRICS\n');
    const { line, rest } = await readUntilNewline(sock, timeoutMs);
    const fr = parseHeader(line, rest);
    if (fr.verb === 'ERR') throw rejectionFromErr(fr.remainder);
    if (fr.verb !== 'METRICS' || fr.numField < 0) {
      throw new ControlPlaneError(
        `unexpected framing: ${line.toString('ascii')}`,
        MOONLAB_CONTROL_IO_ERROR);
    }
    // METRICS numField = byte length of the body.
    const raw = await recvExact(sock, fr.numField, fr.bodySeed, timeoutMs);
    return raw.toString('utf-8');
  } finally {
    sock.destroy();
  }
}
