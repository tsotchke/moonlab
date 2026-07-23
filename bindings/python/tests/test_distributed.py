"""Tests for the distributed (MPI) binding: moonlab.distributed.

The binding contract has two halves; this file tests whichever one the
linked libquantumsim exposes:

- Non-MPI builds (the default CPU library, no QSIM_ENABLE_MPI): the module
  still imports, is_mpi_available() is False, and constructing a
  DistributedState or calling init_mpi() raises an informative
  MpiUnavailableError naming -DQSIM_ENABLE_MPI=ON.
- MPI builds: init_mpi() + partitioned gates + measurement / probability /
  expectation run and, on a single rank (size == 1), match the single-node
  QuantumState reference.

The single-rank path is exercised in-process here; multi-rank behaviour is
what ``mpirun -np N python -m pytest`` (or a dedicated launcher) covers on a
cluster.  A single rank still drives every real C entry point
(partition_state_create, dist_*, collective_*), just with the partition
degenerating to one owner.
"""

import numpy as np
import pytest

import moonlab.distributed as dist
from moonlab.distributed import (
    DistributedState,
    MpiUnavailableError,
    init_mpi,
    finalize_mpi,
    is_mpi_available,
)

MPI = is_mpi_available()


# =============================================================================
# Contract every build must honour.
# =============================================================================


def test_module_imports():
    """The module always imports regardless of MPI support."""
    assert hasattr(dist, "DistributedState")
    assert hasattr(dist, "init_mpi")
    assert hasattr(dist, "finalize_mpi")
    assert isinstance(is_mpi_available(), bool)


def test_finalize_is_safe_without_init():
    """finalize_mpi is a safe no-op when nothing was initialised."""
    # Must not raise on either build.
    finalize_mpi()


@pytest.mark.skipif(MPI, reason="MPI build: no-MPI error path not reachable")
class TestNoMpiBuild:
    """Non-MPI library: distributed construction must fail loudly."""

    def test_init_mpi_raises_informative(self):
        with pytest.raises(MpiUnavailableError) as excinfo:
            init_mpi()
        msg = str(excinfo.value)
        assert "QSIM_ENABLE_MPI=ON" in msg
        assert "MPI" in msg

    def test_distributed_state_raises_informative(self):
        with pytest.raises(MpiUnavailableError) as excinfo:
            DistributedState(num_qubits=4)
        assert "QSIM_ENABLE_MPI=ON" in str(excinfo.value)

    def test_mpi_unavailable_is_runtime_error(self):
        assert issubclass(MpiUnavailableError, RuntimeError)


# =============================================================================
# MPI build, single-rank in-process checks.
# =============================================================================


@pytest.mark.skipif(not MPI, reason="libquantumsim built without QSIM_ENABLE_MPI")
class TestMpiBuildSingleRank:
    """Real distributed engine, exercised on a single rank (size == 1)."""

    @pytest.fixture(scope="class", autouse=True)
    def _mpi(self):
        ctx = init_mpi()
        yield ctx
        finalize_mpi()

    def test_context_rank_size(self, _mpi):
        assert _mpi.rank >= 0
        assert _mpi.size >= 1
        # In-process (no mpirun) this is a single rank.
        if _mpi.size == 1:
            assert _mpi.rank == 0
            assert _mpi.is_root

    def test_ground_state_probability(self, _mpi):
        state = DistributedState(num_qubits=3)
        # Fresh |000>: all probability on basis state 0.
        assert abs(state.probability(0) - 1.0) < 1e-9
        for b in (1, 2, 3, 4, 5, 6, 7):
            assert abs(state.probability(b)) < 1e-9

    def test_single_qubit_superposition_probability(self, _mpi):
        state = DistributedState(num_qubits=3)
        state.h(0)
        # P(qubit 0 == |1>) == 0.5.
        assert abs(state.qubit_probability(0) - 0.5) < 1e-9
        # measure_probability is the doc-guide alias.
        assert abs(state.measure_probability(0) - 0.5) < 1e-9

    def test_bell_pair_matches_reference(self, _mpi):
        """H(0) + CNOT(0,1) yields (|00>+|11>)/sqrt(2): matches QuantumState."""
        if _mpi.size != 1:
            pytest.skip("reference comparison is single-rank only")

        state = DistributedState(num_qubits=2)
        state.h(0).cnot(0, 1)

        p00 = state.probability(0)
        p11 = state.probability(3)
        p01 = state.probability(1)
        p10 = state.probability(2)
        assert abs(p00 - 0.5) < 1e-9
        assert abs(p11 - 0.5) < 1e-9
        assert abs(p01) < 1e-9
        assert abs(p10) < 1e-9

        # Cross-check against the single-node core engine.
        from moonlab import QuantumState
        ref = QuantumState(2)
        ref.h(0).cnot(0, 1)
        ref_probs = ref.probabilities()
        np.testing.assert_allclose(
            [p00, p01, p10, p11],
            [ref_probs[0], ref_probs[1], ref_probs[2], ref_probs[3]],
            atol=1e-9,
        )

    def test_expectation_z_ground_and_excited(self, _mpi):
        state = DistributedState(num_qubits=2)
        # |00>: <Z0> = +1.
        assert abs(state.expectation_z(0) - 1.0) < 1e-9
        # X flips qubit 0 to |1>: <Z0> = -1.
        state.x(0)
        assert abs(state.expectation_z(0) + 1.0) < 1e-9

    def test_expectation_x_on_plus_state(self, _mpi):
        state = DistributedState(num_qubits=2)
        state.h(0)  # |+> on qubit 0.
        assert abs(state.expectation_x(0) - 1.0) < 1e-9
        # <Z> on a |+> state is 0.
        assert abs(state.expectation_z(0)) < 1e-9

    def test_expectation_y_on_i_state(self, _mpi):
        state = DistributedState(num_qubits=2)
        # S H |0> = |+i>, eigenstate of Y with eigenvalue +1.
        state.h(0).s(0)
        assert abs(state.expectation_y(0) - 1.0) < 1e-9

    def test_zz_correlation_bell(self, _mpi):
        state = DistributedState(num_qubits=2)
        state.h(0).cnot(0, 1)
        # Bell state is perfectly Z-correlated: <Z0 Z1> = +1.
        assert abs(state.correlation_zz(0, 1) - 1.0) < 1e-9

    def test_measure_all_collapses_to_ground(self, _mpi):
        state = DistributedState(num_qubits=3)
        # |000> deterministically measures to 0.
        outcome = state.measure_all(seed=12345)
        assert outcome == 0

    def test_measure_single_qubit_deterministic(self, _mpi):
        state = DistributedState(num_qubits=2)
        state.x(1)  # qubit 1 -> |1>.
        assert state.measure(1, seed=7) == 1
        assert state.measure(0, seed=7) == 0

    def test_sample_many_shape(self, _mpi):
        state = DistributedState(num_qubits=3)
        state.h(0).h(1).h(2)  # uniform over 8 states.
        samples = state.sample_many(64, seed=99)
        assert samples.shape == (64,)
        assert samples.dtype == np.uint64
        assert samples.max() < 8
