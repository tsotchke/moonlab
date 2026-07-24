/**
 * @moonlab/quantum-core
 *
 * WebAssembly quantum computing simulator core library.
 * Provides low-level access to quantum state manipulation,
 * gates, and measurement operations.
 *
 * @example
 * ```typescript
 * import { QuantumState, Circuit } from '@moonlab/quantum-core';
 *
 * // Create a 2-qubit Bell state using fluent API
 * const state = await QuantumState.create({ numQubits: 2 });
 * state.h(0).cnot(0, 1);
 * console.log(state.getProbabilities());  // [0.5, 0, 0, 0.5]
 * state.dispose();
 *
 * // Or use Circuit builder for reusable circuits
 * const circuit = new Circuit(2)
 *   .h(0)
 *   .cnot(0, 1);
 *
 * const state2 = await QuantumState.create({ numQubits: 2 });
 * circuit.apply(state2);
 * state2.dispose();
 * ```
 *
 * @packageDocumentation
 */

// ============================================================================
// Core Classes
// ============================================================================

export { QuantumState } from './quantum-state';
export type { QuantumStateOptions } from './quantum-state';

// Clifford-Assisted MPS + var-D + gauge warmstart + Z2 LGT (since 0.2.1).
export {
  CaMps,
  Warmstart,
  StatusModule,
  varDRun,
  gaugeWarmstart,
  z2Lgt1dBuild,
  z2Lgt1dGaussLaw,
  statusString,
} from './ca-mps';
export type { VarDConfig, Z2LgtPauliSum, PauliByte } from './ca-mps';

// Noise channels (since 0.2.1; correct random-draw plumbing, readout
// error, and DeviceNoiseModel since v1.2.0).
export * as noise from './noise';

// Pauli-frame batch shot sampler (since v1.2.0).  Stim-style
// circuit-level sampling: measurement records and detector events over
// bit-packed per-shot frames.
export * as pauliFrame from './pauli-frame';
export { PfOpKind } from './pauli-frame';
export type { PfOp, CircuitSamples, SampleOptions } from './pauli-frame';

// Union-find QEC decoder (since v1.2.0).  Delfosse-Nickerson cluster
// growth + peeling over a detector error model; consumes the
// detector-major batches pauliFrame.sampleDetectors emits.
export { UfDecoder, UF_BOUNDARY } from './uf-decoder';
export type { UfEdge } from './uf-decoder';

// Entanglement metrics (since 0.2.1).
export * as entanglement from './entanglement';

export { Circuit } from './circuit';
export type {
  Gate,
  GateType,
  SingleQubitGate,
  TwoQubitGate,
  ThreeQubitGate,
  MultiQubitGate,
  ParameterizedGate,
  CircuitStats,
} from './circuit';

// ============================================================================
// Complex Number Utilities
// ============================================================================

export {
  complex,
  magnitude,
  magnitudeSquared,
  phase,
  conjugate,
  add,
  subtract,
  multiply,
  scale,
  divide,
  exp,
  fromPolar,
  toPolar,
  equals,
  toString,
  fromInterleaved,
  toInterleaved,
  innerProduct,
  norm,
  normalize,
  ZERO,
  ONE,
  I,
} from './complex';
export type { Complex } from './complex';

// ============================================================================
// WASM Module Management
// ============================================================================

export { getModule, isLoaded, resetModule, preload } from './wasm-loader';
export type { LoadOptions } from './wasm-loader';

export { WasmMemory } from './memory';
export type { MoonlabModule } from './memory';

// ============================================================================
// Tensor Network / Solvers
// ============================================================================
export {
  TensorNetworkState,
  dmrgTFIMGroundState,
} from './tensor-network';
export type {
  TensorNetworkOptions,
  DMRGResult,
  TFIMOptions,
} from './tensor-network';

// Adaptive-bond two-site TDVP (since 0.4.3).
export { TdvpEngine, EvolutionType } from './tdvp';
export type {
  TdvpCommonOptions,
  HeisenbergOptions,
  TfimOptions,
  TdvpHistoryStep,
} from './tdvp';

// Standalone Aaronson-Gottesman Clifford tableau (since 0.4.5).
export { CliffordTableau } from './clifford';
export type { MeasureResult, SampleAllResult } from './clifford';

// Single-qubit gate-fusion DAG (since 0.4.7).
export { FusedCircuit } from './fusion';
export type { FuseStats, FuseCompileResult } from './fusion';

// MPDO mixed-state simulator (since 0.4.10).
export { Mpdo, PauliCode } from './mpdo';

// CA-PEPS 2D Clifford-assisted simulator (since 0.4.12).
export { CaPeps } from './ca-peps';

// Bell inequality + Mermin GHZ/Klyshko tests (since 0.5.4).
export { BellState, createBellState, chshTest, merminGhzTest, merminKlyshkoTest } from './bell';
export type { BellTestResult } from './bell';

// Grover's quantum search (since 0.5.4).
export { groverSearch, groverOptimalIterations } from './grover';
export type { GroverResult } from './grover';

// Variational Quantum Eigensolver (since 0.5.5; UCCSD ansatz, QNG
// optimizer, string optimizer names, and hyperparameter overrides
// since v1.2.0).
export {
  PauliHamiltonian,
  PauliHamiltonianBuilder,
  VqeSolver,
  OptimizerType,
  resolveOptimizer,
} from './vqe';
export type {
  VqeResult,
  VqeSolverOptions,
  OptimizerName,
  AnsatzName,
} from './vqe';

// Quantum Approximate Optimization Algorithm (since 0.5.5).
export { Graph, IsingModel, QaoaSolver } from './qaoa';
export type { QaoaResult } from './qaoa';

// Topological invariants (since 0.5.6).
export {
  qwzChern, chernQwzProj, chernQwzParallelTransport,
  sshWinding, kitaevChainZ2,
  kaneMeleZ2, bhzZ2, hofstadterChern,
} from './topology';

// Surface code (Clifford-tableau variant; since 0.5.14).
export { SurfaceCode } from './surface-code';
export type { PauliError } from './surface-code';

// libirrep QEC zoo (since 0.6.5).  Eight CSS-code families behind
// one class.  Bridge symbols link into the WASM build but the
// underlying libirrep library does NOT (it lives outside the WASM
// toolchain) -- callers must probe LibirrepQecCode.isAvailable()
// first; expect `false` in the browser today.
export {
  LibirrepQecCode,
  LibirrepError,
  LibirrepNotBuiltError,
  MOONLAB_LIBIRREP_OK,
  MOONLAB_LIBIRREP_NOT_BUILT,
  MOONLAB_LIBIRREP_BAD_ARG,
  MOONLAB_LIBIRREP_INTERNAL,
  MOONLAB_LIBIRREP_OOM,
} from './libirrep-qec';

// QGTL-shaped circuit-ingestion surface (since 0.6.8).  Mirrors the
// v0.6.6 C contract; gate-type enum matches QGTL's gate_type_t
// numerically so codes copied from QGTL examples work unchanged.
export {
  QgtlCircuit,
  QgtlError,
  GateType as QgtlGateType,
  MOONLAB_QGTL_OK,
  MOONLAB_QGTL_BAD_ARG,
  MOONLAB_QGTL_OOM,
  MOONLAB_QGTL_UNSUPPORTED,
  MOONLAB_QGTL_INTERNAL,
} from './qgtl';
export type { QgtlExecOptions, QgtlResults } from './qgtl';

// Distributed scheduler MVP (since 0.7.0).  Cloud-platform contract:
// circuit + shot count + worker fan-out -> merged outcomes.  Same as
// QGTL the symbols link into the WASM build but tests auto-skip
// until the next WASM rebuild picks up the v0.7.0 surface.
export { Job, SchedulerError } from './scheduler';
export type { JobResults } from './scheduler';

// Decoder-bench dispatcher (since 0.7.3).  Five-slot QEC decoder zoo
// reachable from JS.  Same auto-skip pattern in tests pending WASM
// rebuild.
export {
  DecoderSlot,
  DecoderError,
  DecoderNotBuiltError,
  decode as decoderDecode,
  slotAvailable as decoderSlotAvailable,
  slotName as decoderSlotName,
  MOONLAB_DECODER_OK,
  MOONLAB_DECODER_NOT_BUILT,
  MOONLAB_DECODER_BAD_ARG,
  MOONLAB_DECODER_INFEASIBLE,
  MOONLAB_DECODER_OOM,
  // Decoder runtime registry (since v1.0.3).
  decoderRegistryAvailable,
  numDecoders,
  listDecoders,
  lookupDecoder,
  registerDecoder,
  unregisterDecoder,
  decodeByName,
} from './decoder';
export type { CodeGeometry, DecoderCallback } from './decoder';
export {
  MOONLAB_SCHED_OK,
  MOONLAB_SCHED_BAD_ARG,
  MOONLAB_SCHED_OOM,
  MOONLAB_SCHED_INTERNAL,
  MOONLAB_SCHED_BUFFER_TOO_SMALL,
  // Vendor-noise profile registry + completion hook (since v1.0.3).
  vendorNoiseProfileRegistryAvailable,
  registerVendorNoiseProfile,
  unregisterVendorNoiseProfile,
  lookupVendorNoiseProfile,
  listVendorNoiseProfiles,
  setCompletionHook,
  clearCompletionHook,
} from './scheduler';
export type {
  VendorNoiseProfile,
  CompletionInfo,
  CompletionCallback,
} from './scheduler';

// Control-plane Node client (since v0.9.4).  Node-only -- needs raw
// TCP, so browsers must go through a WebSocket gateway (separate).
export {
  submitCircuit as controlPlaneSubmitCircuit,
  submitShots   as controlPlaneSubmitShots,
  submitHealth  as controlPlaneSubmitHealth,
  submitMetrics as controlPlaneSubmitMetrics,
  ControlPlaneError,
  MOONLAB_CONTROL_OK,
  MOONLAB_CONTROL_BAD_ARG,
  MOONLAB_CONTROL_AUTH_REQUIRED,
  MOONLAB_CONTROL_AUTH_BAD,
  MOONLAB_CONTROL_IO_ERROR,
  MOONLAB_CONTROL_REJECTED,
  MOONLAB_CONTROL_OOM,
  MOONLAB_CONTROL_RATE_LIMITED,
  MOONLAB_CONTROL_SERVER_BUSY,
} from './control-plane';
export type {
  TlsOptions as ControlPlaneTlsOptions,
  SubmitCircuitArgs as ControlPlaneSubmitCircuitArgs,
  SubmitShotsArgs as ControlPlaneSubmitShotsArgs,
  SubmitMetricsArgs as ControlPlaneSubmitMetricsArgs,
} from './control-plane';

export {
  isWebGPUAvailable,
  initializeWebGPUBackend,
  getActiveTensorGPUBackend,
  isTensorGPUAvailable,
} from './webgpu';
export type { TensorGpuBackend } from './webgpu';

export {
  GPUBackendSession,
  backendTypeName,
  isUnifiedGPUApiAvailable,
  GPU_BACKEND_NONE,
  GPU_BACKEND_METAL,
  GPU_BACKEND_WEBGPU,
  GPU_BACKEND_OPENCL,
  GPU_BACKEND_VULKAN,
  GPU_BACKEND_CUDA,
  GPU_BACKEND_CUQUANTUM,
  GPU_BACKEND_AUTO,
} from './gpu-backend';
export type { GPUBackendTypeCode } from './gpu-backend';

// ============================================================================
// Version Info
// ============================================================================

export const VERSION = '1.2.0';

// ============================================================================
// Type-only Exports
// ============================================================================

/**
 * Basis state represented as a binary string
 * e.g., "00", "01", "10", "11" for 2 qubits
 */
export type BasisState = string;

/**
 * Measurement result with basis state and probability
 */
export interface MeasurementResult {
  basisState: number;
  bitString: BasisState;
  probability: number;
}

/**
 * State vector as array of complex amplitudes
 */
export type StateVector = import('./complex').Complex[];

/**
 * Probability distribution over basis states
 */
export type ProbabilityDistribution = Float64Array;
