/* eslint-disable no-restricted-globals */
const STATE_STRUCT_SIZE = 256;
const AMPLITUDES_OFFSET = 8;
const DEFAULT_ANGLE = Math.PI / 2;

let modulePromise = null;
let moduleInstance = null;

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

const runCircuit = async (payload) => {
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
    return { probabilities, warnings };
  } finally {
    freeState(module, statePtr);
  }
};

const probabilitiesFromAmplitudes = async (payload) => {
  const module = await getModule();
  const numQubits = payload.numQubits;
  const amplitudes = payload.amplitudes;
  const statePtr = createState(module, numQubits);
  try {
    setAmplitudes(module, statePtr, amplitudes);
    const dim = Math.pow(2, numQubits);
    return getProbabilities(module, statePtr, dim);
  } finally {
    freeState(module, statePtr);
  }
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

self.onmessage = async (event) => {
  const { id, type, payload } = event.data || {};
  if (!id || !type) return;

  try {
    let result;
    let transfer = [];

    if (type === 'init') {
      await getModule();
      result = { ready: true };
    } else if (type === 'runCircuit') {
      const response = await runCircuit(payload);
      result = response;
      transfer = [response.probabilities.buffer];
    } else if (type === 'probabilitiesFromAmplitudes') {
      const probabilities = await probabilitiesFromAmplitudes(payload);
      result = { probabilities };
      transfer = [probabilities.buffer];
    } else if (type === 'dmrgWeights') {
      const response = await dmrgWeights(payload);
      result = response;
      transfer = [response.weights.buffer];
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
  }
};
