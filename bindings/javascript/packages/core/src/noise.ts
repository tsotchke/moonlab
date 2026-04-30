/**
 * Noise channel bindings.
 *
 * Apply realistic decoherence channels to a {@link QuantumState}:
 * depolarising, amplitude-damping, phase-damping, pure-dephasing,
 * bit-flip, phase-flip, bit-phase-flip, thermal relaxation.
 *
 * All channels are stochastic (Kraus-operator based).  The state is
 * mutated in place; norm is preserved up to floating-point precision.
 *
 * Underlying C symbols live in `src/quantum/noise.{c,h}`; WASM
 * exports declared in
 * `bindings/javascript/packages/core/emscripten/exports.txt`.
 *
 * @since v0.2.1
 */

import type { QuantumState } from './quantum-state';

interface NoiseModule {
  _noise_depolarizing_single: (s: number, q: number, p: number) => void;
  _noise_depolarizing_two_qubit: (s: number, q1: number, q2: number, p: number) => void;
  _noise_amplitude_damping: (s: number, q: number, gamma: number) => void;
  _noise_phase_damping: (s: number, q: number, gamma: number) => void;
  _noise_pure_dephasing: (s: number, q: number, gamma: number) => void;
  _noise_bit_flip: (s: number, q: number, p: number) => void;
  _noise_phase_flip: (s: number, q: number, p: number) => void;
  _noise_bit_phase_flip: (s: number, q: number, p: number) => void;
  _noise_thermal_relaxation: (s: number, q: number, t1: number, t2: number, t: number) => void;
}

function modAndPtr(state: QuantumState): { mod: NoiseModule; ptr: number } {
  const mod = state._internal_module() as unknown as NoiseModule;
  const ptr = state._internal_state_pointer();
  return { mod, ptr };
}

/** Single-qubit depolarising channel with probability `p`. */
export function depolarizing(state: QuantumState, qubit: number, p: number): void {
  const { mod, ptr } = modAndPtr(state);
  mod._noise_depolarizing_single(ptr, qubit, p);
}

/** Two-qubit depolarising channel with probability `p`. */
export function depolarizing2(
  state: QuantumState, qubit1: number, qubit2: number, p: number,
): void {
  const { mod, ptr } = modAndPtr(state);
  mod._noise_depolarizing_two_qubit(ptr, qubit1, qubit2, p);
}

/** Amplitude-damping channel (T1 relaxation) with rate `gamma` in [0, 1]. */
export function amplitudeDamping(
  state: QuantumState, qubit: number, gamma: number,
): void {
  const { mod, ptr } = modAndPtr(state);
  mod._noise_amplitude_damping(ptr, qubit, gamma);
}

/** Phase-damping channel with rate `gamma` in [0, 1]. */
export function phaseDamping(
  state: QuantumState, qubit: number, gamma: number,
): void {
  const { mod, ptr } = modAndPtr(state);
  mod._noise_phase_damping(ptr, qubit, gamma);
}

/** Pure-dephasing channel with rate `gamma` in [0, 1]. */
export function pureDephasing(
  state: QuantumState, qubit: number, gamma: number,
): void {
  const { mod, ptr } = modAndPtr(state);
  mod._noise_pure_dephasing(ptr, qubit, gamma);
}

/** Bit-flip channel with probability `p`. */
export function bitFlip(state: QuantumState, qubit: number, p: number): void {
  const { mod, ptr } = modAndPtr(state);
  mod._noise_bit_flip(ptr, qubit, p);
}

/** Phase-flip channel with probability `p`. */
export function phaseFlip(state: QuantumState, qubit: number, p: number): void {
  const { mod, ptr } = modAndPtr(state);
  mod._noise_phase_flip(ptr, qubit, p);
}

/** Bit-phase-flip channel with probability `p`. */
export function bitPhaseFlip(state: QuantumState, qubit: number, p: number): void {
  const { mod, ptr } = modAndPtr(state);
  mod._noise_bit_phase_flip(ptr, qubit, p);
}

/** Thermal relaxation with T1, T2 timescales over duration `t`. */
export function thermalRelaxation(
  state: QuantumState,
  qubit: number,
  t1: number,
  t2: number,
  duration: number,
): void {
  const { mod, ptr } = modAndPtr(state);
  mod._noise_thermal_relaxation(ptr, qubit, t1, t2, duration);
}
