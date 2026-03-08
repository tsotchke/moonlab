import { describe, expect, it } from 'vitest';
import {
  GPU_BACKEND_NONE,
  GPUBackendSession,
  isUnifiedGPUApiAvailable,
} from '../gpu-backend';

describe('Unified GPU Backend Integration', () => {
  it('exposes unified GPU API in the WASM module', async () => {
    const available = await isUnifiedGPUApiAvailable();
    expect(available).toBe(true);
  });

  it('creates a backend session with auto selection', async () => {
    const session = await GPUBackendSession.create();
    expect(session).not.toBeNull();
    if (session) {
      expect(typeof session.backendType).toBe('number');
      expect(typeof session.nativeAccelerated).toBe('boolean');
      session.dispose();
    }
  });

  it('applies hadamard when backend supports gate ops', async () => {
    const session = await GPUBackendSession.create();
    expect(session).not.toBeNull();
    if (!session) return;

    const stateDim = 2;
    const buffer = session.createBufferFromInterleaved(new Float64Array([
      1, 0, // |0>
      0, 0, // |1>
    ]));

    try {
      const rc = session.hadamard(buffer, 0, stateDim);

      if (session.backendType === GPU_BACKEND_NONE) {
        expect(rc).toBe(-7); // GPU_ERROR_NOT_SUPPORTED
        return;
      }

      expect(rc).toBe(0);
      const probabilities = session.computeProbabilities(buffer, stateDim);
      expect(probabilities[0]).toBeCloseTo(0.5);
      expect(probabilities[1]).toBeCloseTo(0.5);
    } finally {
      session.freeBuffer(buffer);
      session.dispose();
    }
  });

  it('applies pauli-x/pauli-z and cnot when backend supports gate ops', async () => {
    const session = await GPUBackendSession.create();
    expect(session).not.toBeNull();
    if (!session) return;

    const oneQubitStateDim = 2;
    const oneQubitBuffer = session.createBufferFromInterleaved(new Float64Array([
      1, 0,
      0, 0,
    ]));

    const twoQubitStateDim = 4;
    const originalTwoQubit = new Float64Array([
      0.5, 0.0,
      0.2, -0.3,
      0.1, 0.4,
      -0.6, 0.2,
    ]);
    const twoQubitBuffer = session.createBufferFromInterleaved(originalTwoQubit);

    try {
      const xRc = session.pauliX(oneQubitBuffer, 0, oneQubitStateDim);
      const zRc = session.pauliZ(oneQubitBuffer, 0, oneQubitStateDim);
      const cnotRc1 = session.cnot(twoQubitBuffer, 0, 1, twoQubitStateDim);
      const cnotRc2 = session.cnot(twoQubitBuffer, 0, 1, twoQubitStateDim);

      if (session.backendType === GPU_BACKEND_NONE) {
        expect(xRc).toBe(-7);
        expect(zRc).toBe(-7);
        expect(cnotRc1).toBe(-7);
        expect(cnotRc2).toBe(-7);
        return;
      }

      expect(xRc).toBe(0);
      expect(zRc).toBe(0);
      expect(cnotRc1).toBe(0);
      expect(cnotRc2).toBe(0);

      const oneQubitState = session.readInterleavedBuffer(oneQubitBuffer, oneQubitStateDim);
      expect(oneQubitState[0]).toBeCloseTo(0, 6);
      expect(oneQubitState[1]).toBeCloseTo(0, 6);
      expect(oneQubitState[2]).toBeCloseTo(-1, 6);
      expect(oneQubitState[3]).toBeCloseTo(0, 6);

      const twoQubitState = session.readInterleavedBuffer(twoQubitBuffer, twoQubitStateDim);
      for (let i = 0; i < originalTwoQubit.length; i++) {
        expect(twoQubitState[i]).toBeCloseTo(originalTwoQubit[i], 5);
      }
    } finally {
      session.freeBuffer(oneQubitBuffer);
      session.freeBuffer(twoQubitBuffer);
      session.dispose();
    }
  });
});
