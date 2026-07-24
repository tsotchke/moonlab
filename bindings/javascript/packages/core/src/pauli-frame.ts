/**
 * Pauli-frame batch shot sampler (C-side since v1.2.0; JS binding
 * since v1.2.0).
 *
 * Stim-style circuit-level sampling from `src/backends/clifford/
 * pauli_frame.{c,h}`: one reference tableau pass fixes each
 * measurement's deterministic bit, then a bit-packed batch of per-shot
 * Pauli frames supplies the noise-induced flips, reproducing
 * `stim.compile_sampler().sample()` / `compile_detector_sampler()`
 * output distributions.  On wasm32 the C engine compiles its scalar
 * SIMD fallback and single-threaded fan-out.
 *
 * @example
 * ```typescript
 * import { pauliFrame } from '@moonlab/quantum-core';
 *
 * // 1000 shots of a 3-qubit GHZ circuit.
 * const ops = [
 *   pauliFrame.h(0), pauliFrame.cnot(0, 1), pauliFrame.cnot(1, 2),
 *   pauliFrame.measure(0), pauliFrame.measure(1), pauliFrame.measure(2),
 * ];
 * const { numMeasurements, samples } =
 *   await pauliFrame.sampleCircuit(3, ops, 1000, { seed: 42 });
 * // samples is measurement-major: samples[m * 1000 + shot]
 * ```
 *
 * @since v1.2.0
 */

import { getModule } from './wasm-loader';

/** Op kinds accepted by the sampler.  Values match `pf_op_kind_t`. */
export enum PfOpKind {
  H = 0,
  S = 1,
  SDag = 2,
  X = 3,
  Y = 4,
  Z = 5,
  Cnot = 6,
  Cz = 7,
  Swap = 8,
  Reset = 9,
  Measure = 10,
  /** X with probability p (stim X_ERROR). */
  XError = 11,
  /** Z with probability p (stim Z_ERROR). */
  ZError = 12,
  /** Y with probability p (stim Y_ERROR). */
  YError = 13,
  /** Uniform X/Y/Z each p/3 (stim DEPOLARIZE1). */
  Depolarize1 = 14,
  /** Uniform over the 15 two-qubit Paulis, p/15 each. */
  Depolarize2 = 15,
  /** Measure q0; reported outcome flipped with probability p. */
  MeasureNoisy = 16,
}

/** One circuit instruction.  `q1` is used only by two-qubit ops; `p`
 *  only by the noise channels and {@link PfOpKind.MeasureNoisy}. */
export interface PfOp {
  kind: PfOpKind;
  q0: number;
  q1?: number;
  p?: number;
}

/** Hadamard on `q`. */
export function h(q: number): PfOp { return { kind: PfOpKind.H, q0: q }; }
/** Phase gate S on `q`. */
export function s(q: number): PfOp { return { kind: PfOpKind.S, q0: q }; }
/** S-dagger on `q`. */
export function sdag(q: number): PfOp { return { kind: PfOpKind.SDag, q0: q }; }
/** Pauli X on `q` (deterministic; folded into the reference pass). */
export function x(q: number): PfOp { return { kind: PfOpKind.X, q0: q }; }
/** Pauli Y on `q`. */
export function y(q: number): PfOp { return { kind: PfOpKind.Y, q0: q }; }
/** Pauli Z on `q`. */
export function z(q: number): PfOp { return { kind: PfOpKind.Z, q0: q }; }
/** CNOT with control `c`, target `t`. */
export function cnot(c: number, t: number): PfOp {
  return { kind: PfOpKind.Cnot, q0: c, q1: t };
}
/** CZ on `a`, `b`. */
export function cz(a: number, b: number): PfOp {
  return { kind: PfOpKind.Cz, q0: a, q1: b };
}
/** SWAP on `a`, `b`. */
export function swap(a: number, b: number): PfOp {
  return { kind: PfOpKind.Swap, q0: a, q1: b };
}
/** Reset `q` to |0> (fresh random Z-frame per shot). */
export function reset(q: number): PfOp { return { kind: PfOpKind.Reset, q0: q }; }
/** Z-basis measurement of `q`. */
export function measure(q: number): PfOp { return { kind: PfOpKind.Measure, q0: q }; }
/** X error on `q` with probability `p`. */
export function xError(q: number, p: number): PfOp {
  return { kind: PfOpKind.XError, q0: q, p };
}
/** Z error on `q` with probability `p`. */
export function zError(q: number, p: number): PfOp {
  return { kind: PfOpKind.ZError, q0: q, p };
}
/** Y error on `q` with probability `p`. */
export function yError(q: number, p: number): PfOp {
  return { kind: PfOpKind.YError, q0: q, p };
}
/** Single-qubit depolarising error on `q` with probability `p`. */
export function depolarize1(q: number, p: number): PfOp {
  return { kind: PfOpKind.Depolarize1, q0: q, p };
}
/** Two-qubit depolarising error on `a`, `b` with probability `p`. */
export function depolarize2(a: number, b: number, p: number): PfOp {
  return { kind: PfOpKind.Depolarize2, q0: a, q1: b, p };
}
/** Noisy Z measurement of `q`: outcome flipped with probability `p`. */
export function measureNoisy(q: number, p: number): PfOp {
  return { kind: PfOpKind.MeasureNoisy, q0: q, p };
}

/** Options shared by {@link sampleCircuit} and {@link sampleDetectors}. */
export interface SampleOptions {
  /** Base RNG seed (0 or omitted selects the internal default). */
  seed?: number | bigint;
  /** Shot-block count; <= 0 selects all cores (single-threaded on
   *  wasm32 where OpenMP is unavailable). */
  numThreads?: number;
}

/** Result of {@link sampleCircuit}. */
export interface CircuitSamples {
  /** Number of MEASURE / MEASURE_NOISY ops in the circuit. */
  numMeasurements: number;
  /** Measurement-major outcomes: `samples[m * numShots + shot]` is the
   *  m-th measurement's 0/1 byte for the given shot. */
  samples: Uint8Array;
}

type PauliFrameModule = {
  _pauli_frame_circuit_num_measurements: (opsPtr: number, numOps: number) => number;
  _pauli_frame_batch_sample_circuit: (
    numQubits: number, opsPtr: number, numOps: number,
    numShots: number, seed: bigint, numThreads: number, outPtr: number,
  ) => number;
  _pauli_frame_batch_sample_detectors: (
    numQubits: number, opsPtr: number, numOps: number,
    detOffsetsPtr: number, detIndicesPtr: number,
    numDetectors: number, numShots: number, seed: bigint,
    numThreads: number, outPtr: number,
  ) => number;
  _pauli_frame_simd_backend: () => number;
  _pauli_frame_simd_lanes: () => number;
  _malloc: (n: number) => number;
  _free: (p: number) => void;
  HEAPU8: Uint8Array;
  HEAPU32: Uint32Array;
  HEAPF64: Float64Array;
  UTF8ToString: (ptr: number) => string;
};

/* pf_circuit_op_t layout on wasm32:
 *   0  uint8_t  kind
 *   4  uint32_t q0
 *   8  uint32_t q1
 *  16  double   p
 * sizeof = 24 (double aligns to 8). */
const OP_SIZE = 24;

function marshalOps(mod: PauliFrameModule, ops: readonly PfOp[]): number {
  const ptr = mod._malloc(Math.max(ops.length, 1) * OP_SIZE);
  mod.HEAPU8.fill(0, ptr, ptr + ops.length * OP_SIZE);
  for (let i = 0; i < ops.length; i++) {
    const base = ptr + i * OP_SIZE;
    mod.HEAPU8[base] = ops[i].kind;
    mod.HEAPU32[(base + 4) >> 2] = ops[i].q0 >>> 0;
    mod.HEAPU32[(base + 8) >> 2] = (ops[i].q1 ?? 0) >>> 0;
    mod.HEAPF64[(base + 16) >> 3] = ops[i].p ?? 0;
  }
  return ptr;
}

/** Number of measurements the op list produces (buffer-sizing helper,
 *  calls `pauli_frame_circuit_num_measurements`). */
export async function numMeasurements(ops: readonly PfOp[]): Promise<number> {
  const mod = (await getModule()) as unknown as PauliFrameModule;
  const opsPtr = marshalOps(mod, ops);
  try {
    return mod._pauli_frame_circuit_num_measurements(opsPtr, ops.length);
  } finally {
    mod._free(opsPtr);
  }
}

/**
 * Batch-sample a Clifford + measurement circuit over `numShots`
 * independent shots (`pauli_frame_batch_sample_circuit`).
 *
 * @returns measurement-major outcomes; see {@link CircuitSamples}.
 */
export async function sampleCircuit(
  numQubits: number,
  ops: readonly PfOp[],
  numShots: number,
  options: SampleOptions = {},
): Promise<CircuitSamples> {
  const mod = (await getModule()) as unknown as PauliFrameModule;
  const opsPtr = marshalOps(mod, ops);
  let outPtr = 0;
  try {
    const nMeas = mod._pauli_frame_circuit_num_measurements(opsPtr, ops.length);
    outPtr = mod._malloc(Math.max(nMeas * numShots, 1));
    const rc = mod._pauli_frame_batch_sample_circuit(
      numQubits, opsPtr, ops.length, numShots,
      BigInt(options.seed ?? 0), options.numThreads ?? 1, outPtr,
    );
    if (rc < 0) {
      throw new Error(`pauli_frame_batch_sample_circuit failed (rc=${rc})`);
    }
    return {
      numMeasurements: rc,
      samples: mod.HEAPU8.slice(outPtr, outPtr + rc * numShots),
    };
  } finally {
    if (outPtr) mod._free(outPtr);
    mod._free(opsPtr);
  }
}

/**
 * Batch-sample DETECTORS of a Clifford + measurement circuit
 * (`pauli_frame_batch_sample_detectors`).  A detector is the parity of
 * a set of measurement records, reported as the deviation from the
 * noiseless trajectory -- what a decoder consumes.
 *
 * @param detectors one entry per detector: the measurement indices
 *   (counting MEASURE and MEASURE_NOISY ops in circuit order from 0)
 *   whose parity forms the detector.
 * @returns detector-major bytes: `out[d * numShots + shot]`, the
 *   layout {@link UfDecoder.decodeBatch} consumes directly.
 */
export async function sampleDetectors(
  numQubits: number,
  ops: readonly PfOp[],
  detectors: readonly (readonly number[])[],
  numShots: number,
  options: SampleOptions = {},
): Promise<Uint8Array> {
  const mod = (await getModule()) as unknown as PauliFrameModule;
  const numDetectors = detectors.length;
  const totalIndices = detectors.reduce((acc, d) => acc + d.length, 0);

  const opsPtr = marshalOps(mod, ops);
  const offsetsPtr = mod._malloc((numDetectors + 1) * 4);
  const indicesPtr = mod._malloc(Math.max(totalIndices, 1) * 4);
  const outPtr = mod._malloc(Math.max(numDetectors * numShots, 1));
  try {
    let cursor = 0;
    for (let d = 0; d < numDetectors; d++) {
      mod.HEAPU32[(offsetsPtr >> 2) + d] = cursor;
      for (const m of detectors[d]) {
        mod.HEAPU32[(indicesPtr >> 2) + cursor] = m >>> 0;
        cursor++;
      }
    }
    mod.HEAPU32[(offsetsPtr >> 2) + numDetectors] = cursor;

    const rc = mod._pauli_frame_batch_sample_detectors(
      numQubits, opsPtr, ops.length,
      offsetsPtr, indicesPtr, numDetectors, numShots,
      BigInt(options.seed ?? 0), options.numThreads ?? 1, outPtr,
    );
    if (rc < 0) {
      throw new Error(`pauli_frame_batch_sample_detectors failed (rc=${rc})`);
    }
    return mod.HEAPU8.slice(outPtr, outPtr + numDetectors * numShots);
  } finally {
    mod._free(outPtr);
    mod._free(indicesPtr);
    mod._free(offsetsPtr);
    mod._free(opsPtr);
  }
}

/** Name of the compiled SIMD backend:
 *  `"neon" | "avx2" | "avx512" | "scalar"` (always `"scalar"` on
 *  wasm32). */
export async function simdBackend(): Promise<string> {
  const mod = (await getModule()) as unknown as PauliFrameModule;
  return mod.UTF8ToString(mod._pauli_frame_simd_backend());
}

/** SIMD lane width in 64-bit words (1 = scalar fallback). */
export async function simdLanes(): Promise<number> {
  const mod = (await getModule()) as unknown as PauliFrameModule;
  return mod._pauli_frame_simd_lanes();
}
