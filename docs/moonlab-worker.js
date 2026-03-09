/* eslint-disable no-restricted-globals */
const STATE_STRUCT_SIZE = 256;
const AMPLITUDES_OFFSET = 8;
const DEFAULT_ANGLE = Math.PI / 2;
const GPU_BACKEND_NONE = 0;
const GPU_BACKEND_WEBGPU = 2;
const GPU_BACKEND_AUTO = 7;
const H2_REFERENCE_ENERGY_HARTREE = -1.137283834488;
const HARTREE_TO_KCAL_MOL = 627.5094740631;

let modulePromise = null;
let moduleInstance = null;
const gpuSession = {
  initialized: false,
  available: false,
  ctxPtr: 0,
  backendType: GPU_BACKEND_NONE,
  nativeAccelerated: false,
  reason: 'uninitialized',
  probabilityFailureCount: 0,
  lastProbabilityFailureAt: 0,
};
let lastGpuCircuitFailureAt = 0;

const resetGpuSession = () => {
  gpuSession.initialized = false;
  gpuSession.available = false;
  gpuSession.ctxPtr = 0;
  gpuSession.backendType = GPU_BACKEND_NONE;
  gpuSession.nativeAccelerated = false;
  gpuSession.reason = 'uninitialized';
  gpuSession.probabilityFailureCount = 0;
  gpuSession.lastProbabilityFailureAt = 0;
};

const releaseGpuContext = (module) => {
  if (!module || !gpuSession.ctxPtr) return;
  if (typeof module._gpu_compute_free === 'function') {
    try {
      module._gpu_compute_free(gpuSession.ctxPtr);
    } catch (_err) {
      // Ignore cleanup errors while decommissioning a broken runtime.
    }
  }
  gpuSession.ctxPtr = 0;
};

const describeError = (error) => {
  if (error instanceof Error) {
    return `${error.name}: ${error.message}`;
  }
  return String(error);
};

const isFatalWasmRuntimeError = (error) => {
  const detail =
    error instanceof Error ? `${error.name}: ${error.message}\n${error.stack || ''}` : String(error);
  return /memory access out of bounds|runtimeerror:\s*unreachable|aborted?\(/i.test(detail);
};

const disableGpuPathForModule = (module, reason) => {
  releaseGpuContext(module);
  gpuSession.available = false;
  gpuSession.initialized = true;
  gpuSession.backendType = GPU_BACKEND_NONE;
  gpuSession.nativeAccelerated = false;
  gpuSession.reason = reason;
};

const resetModuleRuntime = () => {
  releaseGpuContext(moduleInstance);
  resetGpuSession();
  moduleInstance = null;
  modulePromise = null;
};

const resetModuleAfterFatalGpuRuntime = (reason, error) => {
  resetModuleRuntime();
  console.warn(`[moonlab-worker] resetting wasm module after ${reason}: ${describeError(error)}`);
};

const getModule = async () => {
  if (moduleInstance) return moduleInstance;
  if (!modulePromise) {
    importScripts('./moonlab.js');
    if (typeof MoonlabModule !== 'function') {
      throw new Error('MoonlabModule factory not available');
    }
    modulePromise = MoonlabModule({
      locateFile: (file, scriptDirectory) => {
        if (file.endsWith('.wasm')) {
          return new URL(file, self.location.href).href;
        }
        return (scriptDirectory || '') + file;
      },
    });
  }
  moduleInstance = await modulePromise;
  if (moduleInstance.ready) {
    await moduleInstance.ready;
  }
  return moduleInstance;
};

const createState = (module, numQubits) => {
  const statePtr = module._malloc(STATE_STRUCT_SIZE);
  if (!statePtr) {
    throw new Error('Failed to allocate quantum state');
  }
  const rc = module._quantum_state_init(statePtr, numQubits);
  if (rc !== 0) {
    module._free(statePtr);
    throw new Error(`quantum_state_init failed: ${rc}`);
  }
  return statePtr;
};

const freeState = (module, statePtr) => {
  if (!statePtr) return;
  module._quantum_state_free(statePtr);
  module._free(statePtr);
};

const setAmplitudes = (module, statePtr, amplitudes) => {
  const amplitudesPtr = module.HEAPU32[(statePtr + AMPLITUDES_OFFSET) >> 2];
  const offset = amplitudesPtr >> 3;
  module.HEAPF64.set(amplitudes, offset);
};

const getProbabilities = (module, statePtr, dim) => {
  const outPtr = module._malloc(dim * 8);
  if (!outPtr) {
    throw new Error('Failed to allocate probability buffer');
  }
  module._measurement_probability_distribution(statePtr, outPtr);
  const offset = outPtr >> 3;
  const probs = new Float64Array(dim);
  probs.set(module.HEAPF64.subarray(offset, offset + dim));
  module._free(outPtr);
  return probs;
};

const getAmplitudes = (module, statePtr, dim) => {
  const amplitudesPtr = module.HEAPU32[(statePtr + AMPLITUDES_OFFSET) >> 2];
  const offset = amplitudesPtr >> 3;
  const amplitudes = new Float64Array(dim * 2);
  amplitudes.set(module.HEAPF64.subarray(offset, offset + dim * 2));
  return amplitudes;
};

const normalizeAngle = (angle) => {
  const twoPi = Math.PI * 2;
  let value = (angle + Math.PI) % twoPi;
  if (value < 0) value += twoPi;
  return value - Math.PI;
};

const readCString = (module, ptr) => {
  if (!ptr) return '';
  if (typeof module.UTF8ToString === 'function') {
    return module.UTF8ToString(ptr);
  }
  let end = ptr;
  while (end < module.HEAPU8.length && module.HEAPU8[end] !== 0) end++;
  return new TextDecoder().decode(module.HEAPU8.subarray(ptr, end));
};

const topStatesFromProbabilities = (probabilities, numQubits, limit = 6) => {
  const ranked = [];
  for (let i = 0; i < probabilities.length; i++) {
    const p = probabilities[i];
    if (!Number.isFinite(p) || p <= 0) continue;
    ranked.push({
      index: i,
      bitstring: i.toString(2).padStart(numQubits, '0'),
      probability: p,
    });
  }
  ranked.sort((a, b) => b.probability - a.probability);
  return ranked.slice(0, limit);
};

const pauliExpectationFromAmplitudes = (amplitudes, pauliString) => {
  const numQubits = pauliString.length;
  const dim = 1 << numQubits;
  let expectationRe = 0;

  for (let i = 0; i < dim; i++) {
    const aiRe = amplitudes[i * 2];
    const aiIm = amplitudes[i * 2 + 1];
    if (aiRe === 0 && aiIm === 0) continue;

    let j = i;
    let phaseRe = 1;
    let phaseIm = 0;

    for (let q = 0; q < numQubits; q++) {
      const op = pauliString[q];
      const bit = (i >> q) & 1;
      if (op === 'I') continue;

      if (op === 'Z') {
        if (bit) {
          phaseRe = -phaseRe;
          phaseIm = -phaseIm;
        }
      } else if (op === 'X') {
        j ^= 1 << q;
      } else if (op === 'Y') {
        j ^= 1 << q;
        const prevRe = phaseRe;
        const prevIm = phaseIm;
        if (bit) {
          // multiply by -i
          phaseRe = prevIm;
          phaseIm = -prevRe;
        } else {
          // multiply by +i
          phaseRe = -prevIm;
          phaseIm = prevRe;
        }
      }
    }

    const ajRe = amplitudes[j * 2];
    const ajIm = amplitudes[j * 2 + 1];
    const tmpRe = phaseRe * ajRe - phaseIm * ajIm;
    const tmpIm = phaseRe * ajIm + phaseIm * ajRe;

    // conj(ai) * tmp
    expectationRe += aiRe * tmpRe + aiIm * tmpIm;
  }

  return expectationRe;
};

const loadH2Hamiltonian = (module, bondDistance = 0.7414) => {
  const fallback = {
    numQubits: 2,
    bondDistance: 0.7414,
    nuclearRepulsion: 0.715104339,
    terms: [
      { pauli: 'II', coefficient: -1.0523732 },
      { pauli: 'IZ', coefficient: 0.39793742 },
      { pauli: 'ZI', coefficient: -0.39793742 },
      { pauli: 'ZZ', coefficient: -0.0112801 },
      { pauli: 'XX', coefficient: 0.1809312 },
    ],
  };

  if (typeof module._vqe_create_h2_hamiltonian !== 'function' || typeof module._pauli_hamiltonian_free !== 'function') {
    return fallback;
  }

  const hamiltonianPtr = module._vqe_create_h2_hamiltonian(bondDistance);
  if (!hamiltonianPtr) {
    return fallback;
  }

  try {
    const numQubits = module.HEAPU32[hamiltonianPtr >> 2];
    const numTerms = module.HEAPU32[(hamiltonianPtr >> 2) + 1];
    const termsPtr = module.HEAPU32[(hamiltonianPtr >> 2) + 2];
    const nuclearRepulsion = module.HEAPF64[(hamiltonianPtr + 16) >> 3];
    const parsedBondDistance = module.HEAPF64[(hamiltonianPtr + 32) >> 3];

    if (numQubits !== 2 || !termsPtr || numTerms === 0 || numTerms > 128) {
      return fallback;
    }

    const terms = [];
    for (let i = 0; i < numTerms; i++) {
      const termPtr = termsPtr + i * 16;
      const coefficient = module.HEAPF64[termPtr >> 3];
      const pauliPtr = module.HEAPU32[(termPtr + 8) >> 2];
      const termQubits = module.HEAPU32[(termPtr + 12) >> 2];
      if (!pauliPtr || termQubits !== 2 || !Number.isFinite(coefficient)) continue;
      const pauli = readCString(module, pauliPtr);
      if (!pauli || pauli.length !== 2) continue;
      if (Math.abs(coefficient) < 1e-12) continue;
      terms.push({ pauli, coefficient });
    }

    return terms.length
      ? {
          numQubits,
          bondDistance: Number.isFinite(parsedBondDistance) ? parsedBondDistance : bondDistance,
          nuclearRepulsion: Number.isFinite(nuclearRepulsion) ? nuclearRepulsion : 0,
          terms,
        }
      : fallback;
  } finally {
    module._pauli_hamiltonian_free(hamiltonianPtr);
  }
};

const probabilityMassFromAmplitudes = (amplitudes) => {
  if (!(amplitudes instanceof Float64Array)) return 1;
  let total = 0;
  for (let i = 0; i + 1 < amplitudes.length; i += 2) {
    const re = amplitudes[i];
    const im = amplitudes[i + 1];
    total += re * re + im * im;
  }
  return Number.isFinite(total) && total > 0 ? total : 1;
};

const isProbabilityDistributionValid = (probabilities, expectedTotal = 1) => {
  if (!(probabilities instanceof Float64Array) || probabilities.length === 0) {
    return { ok: false, total: 0, max: 0 };
  }

  let total = 0;
  let max = 0;
  for (let i = 0; i < probabilities.length; i++) {
    const p = probabilities[i];
    if (!Number.isFinite(p) || p < -1e-10) {
      return { ok: false, total, max };
    }
    total += p;
    if (p > max) max = p;
  }

  if (!Number.isFinite(total) || total <= 1e-12) {
    return { ok: false, total, max };
  }

  const tolerance = Math.max(1e-3, Math.abs(expectedTotal) * 0.05);
  const massOk = Math.abs(total - expectedTotal) <= tolerance;
  const maxOk = max > 1e-12;
  return { ok: massOk && maxOk, total, max };
};

const hasUnifiedGpuApi = (module) =>
  typeof module._gpu_compute_init === 'function' &&
  typeof module._gpu_compute_free === 'function' &&
  typeof module._gpu_get_backend_type === 'function' &&
  typeof module._gpu_buffer_create_from_data === 'function' &&
  typeof module._gpu_buffer_create === 'function' &&
  typeof module._gpu_buffer_read === 'function' &&
  typeof module._gpu_buffer_free === 'function' &&
  (typeof module._gpu_compute_probabilities_u32 === 'function' ||
    typeof module._gpu_compute_probabilities === 'function');

const ensureGpuSession = (module) => {
  if (gpuSession.initialized) return gpuSession;
  gpuSession.initialized = true;

  if (!hasUnifiedGpuApi(module)) {
    gpuSession.reason = 'unified-gpu-api-unavailable';
    return gpuSession;
  }

  const preferred = GPU_BACKEND_WEBGPU;
  let ctxPtr = module._gpu_compute_init(preferred);
  if (!ctxPtr) {
    // Keep AUTO as a secondary probe in case backend selection policy changes.
    ctxPtr = module._gpu_compute_init(GPU_BACKEND_AUTO);
  }
  if (!ctxPtr) {
    gpuSession.reason = 'gpu-context-init-failed';
    return gpuSession;
  }

  const backendType = module._gpu_get_backend_type(ctxPtr);
  if (backendType !== GPU_BACKEND_WEBGPU) {
    module._gpu_compute_free(ctxPtr);
    gpuSession.reason = `non-webgpu-backend-${backendType}`;
    return gpuSession;
  }

  gpuSession.available = true;
  gpuSession.ctxPtr = ctxPtr;
  gpuSession.backendType = backendType;
  gpuSession.nativeAccelerated =
    typeof module._gpu_is_native_accelerated === 'function' &&
    module._gpu_is_native_accelerated(ctxPtr) !== 0;
  gpuSession.reason = 'ok';
  return gpuSession;
};

const noteGpuProbabilityFailure = (reason) => {
  gpuSession.probabilityFailureCount += 1;
  const now = Date.now();
  if (now - gpuSession.lastProbabilityFailureAt > 2000) {
    console.warn(`[moonlab-worker] WebGPU probability path failed: ${reason}; using CPU fallback for this request`);
    gpuSession.lastProbabilityFailureAt = now;
  }
};

const noteGpuCircuitFailure = (reason) => {
  const now = Date.now();
  if (now - lastGpuCircuitFailureAt > 2000) {
    console.warn(`[moonlab-worker] WebGPU circuit path failed: ${reason}; using CPU fallback`);
    lastGpuCircuitFailureAt = now;
  }
};

const gpuCallHadamard = (module, ctxPtr, bufferPtr, qubit, stateDim) => {
  if (typeof module._gpu_hadamard_u32 === 'function') {
    return module._gpu_hadamard_u32(ctxPtr, bufferPtr, qubit, stateDim >>> 0);
  }
  if (typeof module._gpu_hadamard === 'function') {
    return module._gpu_hadamard(ctxPtr, bufferPtr, qubit, BigInt(stateDim));
  }
  return -7;
};

const gpuCallPauliX = (module, ctxPtr, bufferPtr, qubit, stateDim) => {
  if (typeof module._gpu_pauli_x_u32 === 'function') {
    return module._gpu_pauli_x_u32(ctxPtr, bufferPtr, qubit, stateDim >>> 0);
  }
  if (typeof module._gpu_pauli_x === 'function') {
    return module._gpu_pauli_x(ctxPtr, bufferPtr, qubit, BigInt(stateDim));
  }
  return -7;
};

const gpuCallPauliZ = (module, ctxPtr, bufferPtr, qubit, stateDim) => {
  if (typeof module._gpu_pauli_z_u32 === 'function') {
    return module._gpu_pauli_z_u32(ctxPtr, bufferPtr, qubit, stateDim >>> 0);
  }
  if (typeof module._gpu_pauli_z === 'function') {
    return module._gpu_pauli_z(ctxPtr, bufferPtr, qubit, BigInt(stateDim));
  }
  return -7;
};

const gpuCallPhase = (module, ctxPtr, bufferPtr, qubit, theta, stateDim) => {
  if (typeof module._gpu_phase_u32 === 'function') {
    return module._gpu_phase_u32(ctxPtr, bufferPtr, qubit, theta, stateDim >>> 0);
  }
  if (typeof module._gpu_phase === 'function') {
    return module._gpu_phase(ctxPtr, bufferPtr, qubit, theta, BigInt(stateDim));
  }
  return -7;
};

const gpuCallCnot = (module, ctxPtr, bufferPtr, control, target, stateDim) => {
  if (typeof module._gpu_cnot_u32 === 'function') {
    return module._gpu_cnot_u32(ctxPtr, bufferPtr, control, target, stateDim >>> 0);
  }
  if (typeof module._gpu_cnot === 'function') {
    return module._gpu_cnot(ctxPtr, bufferPtr, control, target, BigInt(stateDim));
  }
  return -7;
};

const gpuComputeProbabilities = (module, ctxPtr, amplitudesBufferPtr, stateDim) => {
  const probabilitiesBuffer = module._gpu_buffer_create(ctxPtr, stateDim * 8);
  if (!probabilitiesBuffer) {
    throw new Error('gpu_buffer_create failed for probabilities');
  }

  try {
    const rc =
      typeof module._gpu_compute_probabilities_u32 === 'function'
        ? module._gpu_compute_probabilities_u32(
            ctxPtr,
            amplitudesBufferPtr,
            probabilitiesBuffer,
            stateDim >>> 0
          )
        : module._gpu_compute_probabilities(
            ctxPtr,
            amplitudesBufferPtr,
            probabilitiesBuffer,
            BigInt(stateDim)
          );
    if (rc !== 0) {
      throw new Error(`gpu_compute_probabilities failed: ${rc}`);
    }

    const outPtr = module._malloc(stateDim * 8);
    if (!outPtr) {
      throw new Error('Failed to allocate probability output buffer');
    }
    const readRc = module._gpu_buffer_read(probabilitiesBuffer, outPtr, stateDim * 8, 0);
    if (readRc !== 0) {
      module._free(outPtr);
      throw new Error(`gpu_buffer_read probabilities failed: ${readRc}`);
    }
    const probabilities = new Float64Array(stateDim);
    probabilities.set(module.HEAPF64.subarray(outPtr >> 3, (outPtr >> 3) + stateDim));
    module._free(outPtr);
    return probabilities;
  } finally {
    module._gpu_buffer_free(probabilitiesBuffer);
  }
};

const gpuApplyGate = (module, ctxPtr, amplitudesBufferPtr, gate, stateDim) => {
  const angle = typeof gate.angle === 'number' ? gate.angle : DEFAULT_ANGLE;
  switch (gate.type) {
    case 'H':
      return gpuCallHadamard(module, ctxPtr, amplitudesBufferPtr, gate.qubit, stateDim);
    case 'X':
      return gpuCallPauliX(module, ctxPtr, amplitudesBufferPtr, gate.qubit, stateDim);
    case 'Y': {
      const xRc = gpuCallPauliX(module, ctxPtr, amplitudesBufferPtr, gate.qubit, stateDim);
      if (xRc !== 0) return xRc;
      return gpuCallPauliZ(module, ctxPtr, amplitudesBufferPtr, gate.qubit, stateDim);
    }
    case 'Z':
      return gpuCallPauliZ(module, ctxPtr, amplitudesBufferPtr, gate.qubit, stateDim);
    case 'S':
      return gpuCallPhase(module, ctxPtr, amplitudesBufferPtr, gate.qubit, Math.PI / 2, stateDim);
    case 'T':
      return gpuCallPhase(module, ctxPtr, amplitudesBufferPtr, gate.qubit, Math.PI / 4, stateDim);
    case 'Sdg':
      return gpuCallPhase(module, ctxPtr, amplitudesBufferPtr, gate.qubit, -Math.PI / 2, stateDim);
    case 'Tdg':
      return gpuCallPhase(module, ctxPtr, amplitudesBufferPtr, gate.qubit, -Math.PI / 4, stateDim);
    case 'P':
    case 'Rz':
      // Rz differs from phase by a global phase only, which does not affect probabilities.
      return gpuCallPhase(module, ctxPtr, amplitudesBufferPtr, gate.qubit, angle, stateDim);
    case 'Rx': {
      const h1 = gpuCallHadamard(module, ctxPtr, amplitudesBufferPtr, gate.qubit, stateDim);
      if (h1 !== 0) return h1;
      const p = gpuCallPhase(module, ctxPtr, amplitudesBufferPtr, gate.qubit, angle, stateDim);
      if (p !== 0) return p;
      return gpuCallHadamard(module, ctxPtr, amplitudesBufferPtr, gate.qubit, stateDim);
    }
    case 'Ry':
    case 'U': {
      // U gate in this demo path maps to U3(theta, 0, 0) == Ry(theta).
      const p1 = gpuCallPhase(module, ctxPtr, amplitudesBufferPtr, gate.qubit, -Math.PI / 2, stateDim);
      if (p1 !== 0) return p1;
      const rx = gpuApplyGate(module, ctxPtr, amplitudesBufferPtr, { ...gate, type: 'Rx' }, stateDim);
      if (rx !== 0) return rx;
      return gpuCallPhase(module, ctxPtr, amplitudesBufferPtr, gate.qubit, Math.PI / 2, stateDim);
    }
    case 'CNOT':
      if (typeof gate.controlQubit !== 'number') return -6;
      return gpuCallCnot(
        module,
        ctxPtr,
        amplitudesBufferPtr,
        gate.controlQubit,
        gate.qubit,
        stateDim
      );
    case 'CZ':
      if (typeof gate.controlQubit !== 'number') return -6;
      // CZ = H(target) CNOT(control,target) H(target)
      return (
        gpuCallHadamard(module, ctxPtr, amplitudesBufferPtr, gate.qubit, stateDim) ||
        gpuCallCnot(module, ctxPtr, amplitudesBufferPtr, gate.controlQubit, gate.qubit, stateDim) ||
        gpuCallHadamard(module, ctxPtr, amplitudesBufferPtr, gate.qubit, stateDim)
      );
    case 'CY':
      if (typeof gate.controlQubit !== 'number') return -6;
      // CY = Sdg(target) CNOT(control,target) S(target)
      return (
        gpuCallPhase(module, ctxPtr, amplitudesBufferPtr, gate.qubit, -Math.PI / 2, stateDim) ||
        gpuCallCnot(module, ctxPtr, amplitudesBufferPtr, gate.controlQubit, gate.qubit, stateDim) ||
        gpuCallPhase(module, ctxPtr, amplitudesBufferPtr, gate.qubit, Math.PI / 2, stateDim)
      );
    case 'SWAP':
      if (typeof gate.controlQubit !== 'number') return -6;
      return (
        gpuCallCnot(module, ctxPtr, amplitudesBufferPtr, gate.controlQubit, gate.qubit, stateDim) ||
        gpuCallCnot(module, ctxPtr, amplitudesBufferPtr, gate.qubit, gate.controlQubit, stateDim) ||
        gpuCallCnot(module, ctxPtr, amplitudesBufferPtr, gate.controlQubit, gate.qubit, stateDim)
      );
    case 'DCX':
      if (typeof gate.controlQubit !== 'number') return -6;
      return (
        gpuCallCnot(module, ctxPtr, amplitudesBufferPtr, gate.controlQubit, gate.qubit, stateDim) ||
        gpuCallCnot(module, ctxPtr, amplitudesBufferPtr, gate.qubit, gate.controlQubit, stateDim)
      );
    case 'M':
    case 'Reset':
    case 'Barrier':
      return 0;
    default:
      return null;
  }
};

const applyGate = (module, statePtr, gate, warnings) => {
  const angle = typeof gate.angle === 'number' ? gate.angle : DEFAULT_ANGLE;
  switch (gate.type) {
    case 'H':
      module._gate_hadamard(statePtr, gate.qubit);
      return;
    case 'X':
      module._gate_pauli_x(statePtr, gate.qubit);
      return;
    case 'Y':
      module._gate_pauli_y(statePtr, gate.qubit);
      return;
    case 'Z':
      module._gate_pauli_z(statePtr, gate.qubit);
      return;
    case 'S':
      module._gate_s(statePtr, gate.qubit);
      return;
    case 'T':
      module._gate_t(statePtr, gate.qubit);
      return;
    case 'Sdg':
      module._gate_s_dagger(statePtr, gate.qubit);
      return;
    case 'Tdg':
      module._gate_t_dagger(statePtr, gate.qubit);
      return;
    case 'SX':
      module._gate_rx(statePtr, gate.qubit, Math.PI / 2);
      return;
    case 'Rx':
      module._gate_rx(statePtr, gate.qubit, angle);
      return;
    case 'Ry':
      module._gate_ry(statePtr, gate.qubit, angle);
      return;
    case 'Rz':
      module._gate_rz(statePtr, gate.qubit, angle);
      return;
    case 'P':
      module._gate_phase(statePtr, gate.qubit, angle);
      return;
    case 'U':
      module._gate_u3(statePtr, gate.qubit, angle, 0, 0);
      return;
    case 'CNOT':
      if (typeof gate.controlQubit !== 'number') {
        warnings.push('CNOT gate missing control qubit.');
        return;
      }
      module._gate_cnot(statePtr, gate.controlQubit, gate.qubit);
      return;
    case 'CY':
      if (typeof gate.controlQubit !== 'number') {
        warnings.push('CY gate missing control qubit.');
        return;
      }
      module._gate_cy(statePtr, gate.controlQubit, gate.qubit);
      return;
    case 'CZ':
      if (typeof gate.controlQubit !== 'number') {
        warnings.push('CZ gate missing control qubit.');
        return;
      }
      module._gate_cz(statePtr, gate.controlQubit, gate.qubit);
      return;
    case 'CP':
      if (typeof gate.controlQubit !== 'number') {
        warnings.push('CP gate missing control qubit.');
        return;
      }
      module._gate_cphase(statePtr, gate.controlQubit, gate.qubit, angle);
      return;
    case 'SWAP':
      if (typeof gate.controlQubit !== 'number') {
        warnings.push('SWAP gate missing second qubit.');
        return;
      }
      module._gate_swap(statePtr, gate.controlQubit, gate.qubit);
      return;
    case 'DCX':
      if (typeof gate.controlQubit !== 'number') {
        warnings.push('DCX gate missing second qubit.');
        return;
      }
      module._gate_cnot(statePtr, gate.controlQubit, gate.qubit);
      module._gate_cnot(statePtr, gate.qubit, gate.controlQubit);
      return;
    case 'CCX':
      if (typeof gate.controlQubit !== 'number' || typeof gate.controlQubit2 !== 'number') {
        warnings.push('CCX gate missing control qubits.');
        return;
      }
      module._gate_toffoli(statePtr, gate.controlQubit, gate.controlQubit2, gate.qubit);
      return;
    case 'CCZ':
      if (typeof gate.controlQubit !== 'number' || typeof gate.controlQubit2 !== 'number') {
        warnings.push('CCZ gate missing control qubits.');
        return;
      }
      module._gate_hadamard(statePtr, gate.qubit);
      module._gate_toffoli(statePtr, gate.controlQubit, gate.controlQubit2, gate.qubit);
      module._gate_hadamard(statePtr, gate.qubit);
      return;
    case 'CSWAP':
      if (typeof gate.controlQubit !== 'number' || typeof gate.controlQubit2 !== 'number') {
        warnings.push('CSWAP gate missing qubits.');
        return;
      }
      module._gate_fredkin(statePtr, gate.controlQubit, gate.qubit, gate.controlQubit2);
      return;
    case 'CH':
    case 'iSWAP':
      warnings.push(`Gate ${gate.type} is not supported in the WASM build.`);
      return;
    case 'M':
    case 'Reset':
    case 'Barrier':
      return;
    default:
      warnings.push(`Unknown gate type: ${gate.type}`);
      return;
  }
};

const runCircuitCpu = async (payload) => {
  const module = await getModule();
  const warnings = [];
  const numQubits = payload.numQubits;
  const gates = payload.gates || [];
  const statePtr = createState(module, numQubits);
  try {
    for (const gate of gates) {
      applyGate(module, statePtr, gate, warnings);
    }
    const dim = Math.pow(2, numQubits);
    const probabilities = getProbabilities(module, statePtr, dim);
    return { probabilities, warnings, backend: 'cpu', nativeAccelerated: false };
  } finally {
    freeState(module, statePtr);
  }
};

const runCircuitGpu = (module, payload) => {
  const session = ensureGpuSession(module);
  if (!session.available) {
    return null;
  }

  const numQubits = payload.numQubits;
  const gates = payload.gates || [];
  const dim = Math.pow(2, numQubits);
  if (!Number.isFinite(dim) || dim < 1 || dim > 0xffffffff) {
    return null;
  }

  const amplitudes = new Float64Array(dim * 2);
  amplitudes[0] = 1.0;
  const expectedMass = probabilityMassFromAmplitudes(amplitudes);

  const amplitudesPtr = module._malloc(amplitudes.byteLength);
  if (!amplitudesPtr) return null;
  module.HEAPF64.set(amplitudes, amplitudesPtr >> 3);

  const amplitudesBuffer = module._gpu_buffer_create_from_data(
    session.ctxPtr,
    amplitudesPtr,
    amplitudes.byteLength
  );
  module._free(amplitudesPtr);
  if (!amplitudesBuffer) return null;

  try {
    for (const gate of gates) {
      const rc = gpuApplyGate(module, session.ctxPtr, amplitudesBuffer, gate, dim);
      if (rc === null) {
        return null;
      }
      if (rc !== 0) {
        return null;
      }
    }

    let probabilities;
    try {
      probabilities = gpuComputeProbabilities(module, session.ctxPtr, amplitudesBuffer, dim);
    } catch (err) {
      noteGpuProbabilityFailure(`runCircuit exception: ${err instanceof Error ? err.message : String(err)}`);
      return null;
    }
    const validation = isProbabilityDistributionValid(probabilities, expectedMass);
    if (!validation.ok) {
      noteGpuProbabilityFailure(
        `runCircuit invalid output (sum=${validation.total.toExponential(3)}, max=${validation.max.toExponential(3)})`
      );
      return null;
    }
    return {
      probabilities,
      warnings: [],
      backend: 'webgpu',
      nativeAccelerated: session.nativeAccelerated,
    };
  } finally {
    module._gpu_buffer_free(amplitudesBuffer);
  }
};

const runCircuit = async (payload) => {
  if (payload?.cleanupAfterRun === true) {
    return runCircuitCpu(payload);
  }
  const module = await getModule();
  try {
    const gpuResult = runCircuitGpu(module, payload);
    if (gpuResult) {
      return gpuResult;
    }
  } catch (err) {
    noteGpuCircuitFailure(describeError(err));
    if (isFatalWasmRuntimeError(err)) {
      resetModuleAfterFatalGpuRuntime('runCircuit', err);
    } else {
      disableGpuPathForModule(module, 'disabled-after-runCircuit-error');
    }
  }
  return runCircuitCpu(payload);
};

const probabilitiesFromAmplitudesCpu = async (payload) => {
  const module = await getModule();
  const numQubits = payload.numQubits;
  const amplitudes = payload.amplitudes;
  const statePtr = createState(module, numQubits);
  try {
    setAmplitudes(module, statePtr, amplitudes);
    const dim = Math.pow(2, numQubits);
    return { probabilities: getProbabilities(module, statePtr, dim), backend: 'cpu' };
  } finally {
    freeState(module, statePtr);
  }
};

const probabilitiesFromAmplitudesGpu = (module, payload) => {
  const session = ensureGpuSession(module);
  if (!session.available) return null;

  const numQubits = payload.numQubits;
  const amplitudes = payload.amplitudes;
  const dim = Math.pow(2, numQubits);
  if (!Number.isFinite(dim) || dim < 1 || dim > 0xffffffff) {
    return null;
  }
  if (!(amplitudes instanceof Float64Array) || amplitudes.length !== dim * 2) {
    return null;
  }
  const expectedMass = probabilityMassFromAmplitudes(amplitudes);

  const amplitudesPtr = module._malloc(amplitudes.byteLength);
  if (!amplitudesPtr) return null;
  module.HEAPF64.set(amplitudes, amplitudesPtr >> 3);

  const amplitudesBuffer = module._gpu_buffer_create_from_data(
    session.ctxPtr,
    amplitudesPtr,
    amplitudes.byteLength
  );
  module._free(amplitudesPtr);
  if (!amplitudesBuffer) return null;

  try {
    let probabilities;
    try {
      probabilities = gpuComputeProbabilities(module, session.ctxPtr, amplitudesBuffer, dim);
    } catch (err) {
      noteGpuProbabilityFailure(
        `probabilitiesFromAmplitudes exception: ${err instanceof Error ? err.message : String(err)}`
      );
      return null;
    }
    const validation = isProbabilityDistributionValid(probabilities, expectedMass);
    if (!validation.ok) {
      noteGpuProbabilityFailure(
        `probabilitiesFromAmplitudes invalid output (sum=${validation.total.toExponential(
          3
        )}, max=${validation.max.toExponential(3)})`
      );
      return null;
    }
    return {
      probabilities,
      backend: 'webgpu',
    };
  } finally {
    module._gpu_buffer_free(amplitudesBuffer);
  }
};

const probabilitiesFromAmplitudes = async (payload) => {
  const module = await getModule();
  try {
    const gpuResult = probabilitiesFromAmplitudesGpu(module, payload);
    if (gpuResult) {
      return gpuResult;
    }
  } catch (err) {
    noteGpuProbabilityFailure(`probabilitiesFromAmplitudes exception: ${describeError(err)}`);
    if (isFatalWasmRuntimeError(err)) {
      resetModuleAfterFatalGpuRuntime('probabilitiesFromAmplitudes', err);
    } else {
      disableGpuPathForModule(module, 'disabled-after-probabilities-error');
    }
  }
  return probabilitiesFromAmplitudesCpu(payload);
};

const runGroverExample = async () => {
  const module = await getModule();
  const numQubits = 10;
  const markedState = 42;
  const markedStateArg = BigInt(markedState);
  const dim = 1 << numQubits;
  const statePtr = createState(module, numQubits);

  try {
    for (let q = 0; q < numQubits; q++) {
      module._gate_hadamard(statePtr, q);
    }

    const iterations =
      typeof module._grover_optimal_iterations === 'function'
        ? Math.max(1, module._grover_optimal_iterations(numQubits) | 0)
        : Math.max(1, Math.floor((Math.PI / 4) * Math.sqrt(dim)));

    for (let i = 0; i < iterations; i++) {
      let rc = -1;
      if (typeof module._grover_iteration === 'function') {
        rc = module._grover_iteration(statePtr, markedStateArg);
      } else if (typeof module._grover_oracle === 'function' && typeof module._grover_diffusion === 'function') {
        const oracleRc = module._grover_oracle(statePtr, markedStateArg);
        const diffusionRc = oracleRc === 0 ? module._grover_diffusion(statePtr) : oracleRc;
        rc = diffusionRc;
      }
      if (rc !== 0) {
        throw new Error(`Grover iteration failed at step ${i + 1}: ${rc}`);
      }
    }

    const probabilities = getProbabilities(module, statePtr, dim);
    const successProbability = probabilities[markedState] || 0;
    let foundState = 0;
    let maxProbability = probabilities[0] || 0;
    for (let i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > maxProbability) {
        maxProbability = probabilities[i];
        foundState = i;
      }
    }

    return {
      algorithm: 'grover',
      numQubits,
      markedState,
      iterations,
      oracleCalls: iterations,
      foundState,
      successProbability,
      topStates: topStatesFromProbabilities(probabilities, numQubits, 8),
      probabilities,
    };
  } finally {
    freeState(module, statePtr);
  }
};

const runTeleportationExample = async () => {
  const module = await getModule();
  const statePtr = createState(module, 3);
  const sourcePtr = createState(module, 1);

  try {
    // Source state |psi> = T·H|0>
    module._gate_hadamard(sourcePtr, 0);
    module._gate_t(sourcePtr, 0);
    const sourceBloch = {
      x: module._measurement_expectation_x(sourcePtr, 0),
      y: module._measurement_expectation_y(sourcePtr, 0),
      z: module._measurement_expectation_z(sourcePtr, 0),
    };

    // Prepare |psi> on q0
    module._gate_hadamard(statePtr, 0);
    module._gate_t(statePtr, 0);

    // Entangled pair on q1, q2
    module._gate_hadamard(statePtr, 1);
    module._gate_cnot(statePtr, 1, 2);

    // Bell measurement pre-rotations
    module._gate_cnot(statePtr, 0, 1);
    module._gate_hadamard(statePtr, 0);

    const m0 = module._measurement_single_qubit(statePtr, 0, Math.random()) ? 1 : 0;
    const m1 = module._measurement_single_qubit(statePtr, 1, Math.random()) ? 1 : 0;

    if (m1) module._gate_pauli_x(statePtr, 2);
    if (m0) module._gate_pauli_z(statePtr, 2);

    const targetBloch = {
      x: module._measurement_expectation_x(statePtr, 2),
      y: module._measurement_expectation_y(statePtr, 2),
      z: module._measurement_expectation_z(statePtr, 2),
    };
    const blochDot =
      sourceBloch.x * targetBloch.x +
      sourceBloch.y * targetBloch.y +
      sourceBloch.z * targetBloch.z;
    const fidelity = Math.max(0, Math.min(1, 0.5 * (1 + blochDot)));
    const probabilities = getProbabilities(module, statePtr, 8);

    return {
      algorithm: 'quantum-teleportation',
      measurementBits: { m0, m1 },
      sourceBloch,
      targetBloch,
      fidelity,
      topStates: topStatesFromProbabilities(probabilities, 3, 8),
      probabilities,
    };
  } finally {
    freeState(module, sourcePtr);
    freeState(module, statePtr);
  }
};

const evaluateVqeH2Energy = (module, statePtr, params, hamiltonian) => {
  module._quantum_state_reset(statePtr);

  module._gate_ry(statePtr, 0, params[0]);
  module._gate_ry(statePtr, 1, params[1]);
  module._gate_cnot(statePtr, 0, 1);
  module._gate_ry(statePtr, 0, params[2]);
  module._gate_ry(statePtr, 1, params[3]);
  module._gate_cnot(statePtr, 1, 0);

  const amplitudes = getAmplitudes(module, statePtr, 4);
  let energy = hamiltonian.nuclearRepulsion || 0;
  for (const term of hamiltonian.terms) {
    const expectation = term.pauli === 'II' ? 1 : pauliExpectationFromAmplitudes(amplitudes, term.pauli);
    energy += term.coefficient * expectation;
  }

  return { energy, amplitudes };
};

const runVqeH2Example = async () => {
  const module = await getModule();
  const hamiltonian = loadH2Hamiltonian(module, 0.7414);
  const statePtr = createState(module, 2);

  try {
    const paramCount = 4;
    let evaluations = 0;
    let iterations = 0;
    let bestEnergy = Infinity;
    let bestParams = new Float64Array(paramCount);
    let bestAmplitudes = new Float64Array(8);

    for (let restart = 0; restart < 4; restart++) {
      const params = new Float64Array(paramCount);
      for (let p = 0; p < paramCount; p++) {
        params[p] = normalizeAngle((Math.random() * 2 - 1) * Math.PI);
      }

      let step = 0.8;
      let evalResult = evaluateVqeH2Energy(module, statePtr, params, hamiltonian);
      evaluations += 1;
      if (evalResult.energy < bestEnergy) {
        bestEnergy = evalResult.energy;
        bestParams = params.slice();
        bestAmplitudes = evalResult.amplitudes;
      }

      for (let iter = 0; iter < 60; iter++) {
        iterations += 1;
        let improved = false;

        for (let p = 0; p < paramCount; p++) {
          const baselineValue = params[p];
          let localBestValue = baselineValue;
          let localBestEnergy = evalResult.energy;
          const deltas = [step, -step, step * 0.5, -step * 0.5];

          for (const delta of deltas) {
            params[p] = normalizeAngle(baselineValue + delta);
            const trial = evaluateVqeH2Energy(module, statePtr, params, hamiltonian);
            evaluations += 1;
            if (trial.energy < localBestEnergy) {
              localBestEnergy = trial.energy;
              localBestValue = params[p];
              evalResult = trial;
              improved = true;
            }
          }

          params[p] = localBestValue;
          if (localBestEnergy < bestEnergy) {
            bestEnergy = localBestEnergy;
            bestParams = params.slice();
            bestAmplitudes = evalResult.amplitudes;
          }
        }

        step *= improved ? 0.92 : 0.7;
        if (step < 1e-3) break;
      }
    }

    const finalProbabilities = new Float64Array(4);
    for (let i = 0; i < 4; i++) {
      const re = bestAmplitudes[i * 2];
      const im = bestAmplitudes[i * 2 + 1];
      finalProbabilities[i] = re * re + im * im;
    }

    const chemicalAccuracyKcalMol =
      Math.abs(bestEnergy - H2_REFERENCE_ENERGY_HARTREE) * HARTREE_TO_KCAL_MOL;

    return {
      algorithm: 'vqe-h2',
      bondDistance: hamiltonian.bondDistance,
      energyHartree: bestEnergy,
      referenceEnergyHartree: H2_REFERENCE_ENERGY_HARTREE,
      chemicalAccuracyKcalMol,
      convergedToChemicalAccuracy: chemicalAccuracyKcalMol <= 1.0,
      iterations,
      evaluations,
      parameters: bestParams,
      hamiltonian: {
        nuclearRepulsion: hamiltonian.nuclearRepulsion,
        terms: hamiltonian.terms,
      },
      topStates: topStatesFromProbabilities(finalProbabilities, 2, 4),
      probabilities: finalProbabilities,
    };
  } finally {
    freeState(module, statePtr);
  }
};

const runExampleAlgorithm = async (payload) => {
  const id = payload?.id;
  if (id === 'grover') return runGroverExample();
  if (id === 'quantum-teleportation') return runTeleportationExample();
  if (id === 'vqe-h2') return runVqeH2Example();
  throw new Error(`Unsupported example algorithm: ${String(id)}`);
};

const dmrgWeights = async (payload) => {
  const module = await getModule();
  const numSites = payload.numSites;
  const g = payload.g;
  const start = performance.now();

  const resultPtrPtr = module._malloc(4);
  module.HEAPU32[resultPtrPtr >> 2] = 0;

  const mpsPtr = module._dmrg_tfim_ground_state(numSites, g, 0, resultPtrPtr);
  if (!mpsPtr) {
    module._free(resultPtrPtr);
    throw new Error('dmrg_tfim_ground_state failed');
  }

  const resultPtr = module.HEAPU32[resultPtrPtr >> 2];
  let energy;
  let variance;
  if (resultPtr) {
    const base = resultPtr >> 3;
    energy = module.HEAPF64[base];
    variance = module.HEAPF64[base + 1];
    module._dmrg_result_free(resultPtr);
  }
  module._free(resultPtrPtr);

  const dim = Math.pow(2, numSites);
  const outPtr = module._malloc(dim * 16);
  if (!outPtr) {
    module._tn_mps_free(mpsPtr);
    throw new Error('Failed to allocate statevector buffer');
  }

  const rc = module._tn_mps_to_statevector(mpsPtr, outPtr);
  if (rc !== 0) {
    module._free(outPtr);
    module._tn_mps_free(mpsPtr);
    throw new Error(`tn_mps_to_statevector failed: ${rc}`);
  }

  const offset = outPtr >> 3;
  const amps = module.HEAPF64.subarray(offset, offset + dim * 2);
  const weights = new Float64Array(dim);
  let total = 0;
  for (let i = 0; i < dim; i++) {
    const re = amps[i * 2];
    const im = amps[i * 2 + 1];
    const prob = re * re + im * im;
    weights[i] = prob;
    total += prob;
  }
  if (total > 0) {
    for (let i = 0; i < dim; i++) {
      weights[i] /= total;
    }
  }

  module._free(outPtr);
  module._tn_mps_free(mpsPtr);

  return {
    weights,
    energy,
    variance,
    elapsedMs: performance.now() - start,
  };
};

let requestQueue = Promise.resolve();

const handleWorkerMessage = async (event) => {
  const { id, type, payload } = event.data || {};
  if (!id || !type) return;
  const cleanupAfterRun =
    payload?.cleanupAfterRun === true &&
    (type === 'runCircuit' || type === 'runExampleAlgorithm');

  try {
    let result;
    let transfer = [];

    if (type === 'init') {
      const module = await getModule();
      const session = ensureGpuSession(module);
      result = {
        ready: true,
        webgpu: {
          available: session.available,
          nativeAccelerated: session.nativeAccelerated,
          reason: session.reason,
        },
      };
    } else if (type === 'runCircuit') {
      const response = await runCircuit(payload);
      result = response;
      transfer = [response.probabilities.buffer];
    } else if (type === 'probabilitiesFromAmplitudes') {
      const response = await probabilitiesFromAmplitudes(payload);
      result = response;
      transfer = [response.probabilities.buffer];
    } else if (type === 'dmrgWeights') {
      const response = await dmrgWeights(payload);
      result = response;
      transfer = [response.weights.buffer];
    } else if (type === 'runExampleAlgorithm') {
      const response = await runExampleAlgorithm(payload);
      result = response;
      transfer = [];
      if (response.probabilities instanceof Float64Array) {
        transfer.push(response.probabilities.buffer);
      }
      if (response.parameters instanceof Float64Array) {
        transfer.push(response.parameters.buffer);
      }
    } else {
      throw new Error(`Unknown worker request: ${type}`);
    }

    self.postMessage({ id, ok: true, result }, transfer);
  } catch (error) {
    self.postMessage({
      id,
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    });
  } finally {
    if (cleanupAfterRun) {
      resetModuleRuntime();
    }
  }
};

self.onmessage = (event) => {
  requestQueue = requestQueue.then(() => handleWorkerMessage(event)).catch((error) => {
    console.error('[moonlab-worker] unhandled request failure', error);
  });
};
