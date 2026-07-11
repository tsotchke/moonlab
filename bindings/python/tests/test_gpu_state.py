"""
Tests for the v1.1 GPU state surface: QuantumState.create_gpu,
sync_to_host / sync_from_host, and the is_gpu property.

The binding contract has two halves and this file tests whichever one
the linked libquantumsim exposes:

- CUDA builds: create_gpu returns a GPU-backed state, gates dispatch
  transparently, and syncs move amplitudes between device and host.
- Non-CUDA builds (the weak-stub path): create_gpu raises QuantumError
  carrying the QS_ERROR_NOT_SUPPORTED code, and the sync methods are
  documented no-ops on CPU states.
"""

import numpy as np
import pytest

from moonlab import QuantumState, QuantumError
from moonlab.core import statevector_to_numpy


def _gpu_available() -> bool:
    """Probe once whether this libquantumsim can allocate a GPU state."""
    try:
        state = QuantumState.create_gpu(2)
    except (QuantumError, RuntimeError):
        return False
    del state
    return True


GPU_AVAILABLE = _gpu_available()


class TestCpuFallbackContract:
    """The half of the contract every build must honor."""

    def test_cpu_state_is_not_gpu(self):
        state = QuantumState(3)
        assert state.is_gpu is False

    def test_sync_to_host_is_noop_on_cpu_state(self):
        state = QuantumState(2)
        state.h(0)
        before = statevector_to_numpy(state)
        assert state.sync_to_host() is state
        after = statevector_to_numpy(state)
        np.testing.assert_allclose(before, after)

    def test_sync_from_host_is_noop_on_cpu_state(self):
        state = QuantumState(2)
        assert state.sync_from_host() is state
        assert abs(state.probability(0) - 1.0) < 1e-10

    def test_create_gpu_rejects_bad_qubit_counts(self):
        for bad in (0, -1, 32, 64):
            with pytest.raises((ValueError, QuantumError)):
                QuantumState.create_gpu(bad)


@pytest.mark.skipif(GPU_AVAILABLE, reason="CUDA build: stub path not reachable")
class TestNonCudaBuild:
    """Weak-stub path: creation must fail loudly, never silently CPU."""

    def test_create_gpu_raises_not_supported(self):
        with pytest.raises((QuantumError, RuntimeError)) as excinfo:
            QuantumState.create_gpu(4)
        message = str(excinfo.value).lower()
        assert "gpu" in message or "-7" in message or "not" in message


@pytest.mark.skipif(not GPU_AVAILABLE, reason="libquantumsim built without CUDA")
class TestCudaBuild:
    """Real-GPU path: exercised on CUDA validation lanes."""

    def test_create_gpu_state_shape(self):
        state = QuantumState.create_gpu(4)
        assert state.num_qubits == 4
        assert state.state_dim == 16
        assert state.is_gpu is True

    def test_gpu_bell_pair_round_trip(self):
        state = QuantumState.create_gpu(2)
        state.h(0)
        state.cnot(0, 1)
        state.sync_to_host()
        probs = np.abs(statevector_to_numpy(state)) ** 2
        np.testing.assert_allclose(probs[0], 0.5, atol=1e-10)
        np.testing.assert_allclose(probs[3], 0.5, atol=1e-10)
        np.testing.assert_allclose(probs[1], 0.0, atol=1e-10)
        np.testing.assert_allclose(probs[2], 0.0, atol=1e-10)

    def test_gpu_norm_preserved_over_gates(self):
        state = QuantumState.create_gpu(5)
        for q in range(5):
            state.h(q)
        state.rx(1, 0.7)
        state.rz(3, -1.3)
        state.cnot(0, 4)
        state.sync_to_host()
        probs = np.abs(statevector_to_numpy(state)) ** 2
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-9)
