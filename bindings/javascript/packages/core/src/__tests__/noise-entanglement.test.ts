import { describe, expect, it, vi } from 'vitest';
import type { QuantumState } from '../quantum-state';
import * as entanglement from '../entanglement';
import * as noise from '../noise';

interface MockModule {
  _noise_depolarizing_single: (s: number, q: number, p: number) => void;
  _noise_depolarizing_two_qubit: (s: number, q1: number, q2: number, p: number) => void;
  _noise_amplitude_damping: (s: number, q: number, gamma: number) => void;
  _noise_phase_damping: (s: number, q: number, gamma: number) => void;
  _noise_pure_dephasing: (s: number, q: number, gamma: number) => void;
  _noise_bit_flip: (s: number, q: number, p: number) => void;
  _noise_phase_flip: (s: number, q: number, p: number) => void;
  _noise_bit_phase_flip: (s: number, q: number, p: number) => void;
  _noise_thermal_relaxation: (s: number, q: number, t1: number, t2: number, t: number) => void;
  _entanglement_entropy_bipartition: (s: number, qubitsBPtr: number, numB: number) => number;
  _entanglement_mutual_information: (
    s: number,
    qubitsAPtr: number,
    numA: number,
    qubitsBPtr: number,
    numB: number,
  ) => number;
  _entanglement_concurrence_2qubit: (s: number) => number;
  _entanglement_negativity_2qubit: (s: number) => number;
  HEAP32: Int32Array;
  _malloc: (n: number) => number;
  _free: (p: number) => void;
}

function makeState(): { state: QuantumState; mod: MockModule; ptr: number } {
  const ptr = 0x1234;
  const mod: MockModule = {
    _noise_depolarizing_single: vi.fn(),
    _noise_depolarizing_two_qubit: vi.fn(),
    _noise_amplitude_damping: vi.fn(),
    _noise_phase_damping: vi.fn(),
    _noise_pure_dephasing: vi.fn(),
    _noise_bit_flip: vi.fn(),
    _noise_phase_flip: vi.fn(),
    _noise_bit_phase_flip: vi.fn(),
    _noise_thermal_relaxation: vi.fn(),
    _entanglement_entropy_bipartition: vi.fn(() => 0.5),
    _entanglement_mutual_information: vi.fn(() => 1.0),
    _entanglement_concurrence_2qubit: vi.fn(() => 1.0),
    _entanglement_negativity_2qubit: vi.fn(() => 0.5),
    HEAP32: new Int32Array(16),
    _malloc: vi.fn(() => 0),
    _free: vi.fn(),
  };

  const state = {
    _internal_module: () => mod,
    _internal_state_pointer: () => ptr,
  } as unknown as QuantumState;

  return { state, mod, ptr };
}

describe('noise channel bindings', () => {
  it('routes every exported channel to its native ABI symbol', () => {
    const { state, mod, ptr } = makeState();

    noise.depolarizing(state, 0, 0.125);
    noise.depolarizing2(state, 0, 1, 0.25);
    noise.amplitudeDamping(state, 1, 0.375);
    noise.phaseDamping(state, 1, 0.5);
    noise.pureDephasing(state, 0, 0.625);
    noise.bitFlip(state, 0, 0.75);
    noise.phaseFlip(state, 1, 0.875);
    noise.bitPhaseFlip(state, 1, 0.2);
    noise.thermalRelaxation(state, 0, 10.0, 8.0, 0.5);

    expect(mod._noise_depolarizing_single).toHaveBeenCalledWith(ptr, 0, 0.125);
    expect(mod._noise_depolarizing_two_qubit).toHaveBeenCalledWith(ptr, 0, 1, 0.25);
    expect(mod._noise_amplitude_damping).toHaveBeenCalledWith(ptr, 1, 0.375);
    expect(mod._noise_phase_damping).toHaveBeenCalledWith(ptr, 1, 0.5);
    expect(mod._noise_pure_dephasing).toHaveBeenCalledWith(ptr, 0, 0.625);
    expect(mod._noise_bit_flip).toHaveBeenCalledWith(ptr, 0, 0.75);
    expect(mod._noise_phase_flip).toHaveBeenCalledWith(ptr, 1, 0.875);
    expect(mod._noise_bit_phase_flip).toHaveBeenCalledWith(ptr, 1, 0.2);
    expect(mod._noise_thermal_relaxation).toHaveBeenCalledWith(ptr, 0, 10.0, 8.0, 0.5);
  });
});

describe('entanglement metric bindings', () => {
  it('marshals subsystem lists and routes scalar metric calls', () => {
    const { state, mod, ptr } = makeState();

    expect(entanglement.vonNeumannEntropy(state, [1, 3])).toBe(0.5);
    expect(mod._malloc).toHaveBeenCalledWith(8);
    expect(Array.from(mod.HEAP32.slice(0, 2))).toEqual([1, 3]);
    expect(mod._entanglement_entropy_bipartition).toHaveBeenCalledWith(ptr, 0, 2);
    expect(mod._free).toHaveBeenCalledWith(0);

    expect(entanglement.mutualInformation(state, [0], [1, 2])).toBe(1.0);
    expect(mod._entanglement_mutual_information).toHaveBeenCalledWith(ptr, 0, 1, 0, 2);
    expect(entanglement.concurrence2Qubit(state)).toBe(1.0);
    expect(entanglement.negativity2Qubit(state)).toBe(0.5);
  });
});
