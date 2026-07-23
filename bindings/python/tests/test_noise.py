"""Physical-effect tests for moonlab.noise.

Every channel here is the real C trajectory unravelling, so the CPTP-map
effects (purity decay toward maximally mixed, Bloch-vector shrinkage,
|1> -> |0> relaxation) only appear after averaging the reduced density
operator over many trajectories.  The tests reconstruct that average and
assert on real numbers -- purity, Bloch components, populations -- not on
"did not crash".
"""

import numpy as np
import pytest

from moonlab import QuantumState
from moonlab.noise import (
    DepolarizingChannel,
    AmplitudeDamping,
    PhaseDamping,
    BitFlip,
    PhaseFlip,
    BitPhaseFlip,
    ThermalRelaxation,
    ReadoutError,
    DeviceNoiseModel,
)

# Pauli matrices for Bloch-vector reconstruction.
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def single_qubit_dm(state):
    """1-qubit reduced density matrix of a pure QuantumState (n == 1)."""
    psi = state.get_statevector()
    return np.outer(psi, np.conj(psi))


def averaged_dm(prepare, channel_apply, shots=4000):
    """Average the single-qubit density matrix over many trajectories.

    prepare() returns a fresh 1-qubit QuantumState; channel_apply(state)
    applies one trajectory in place.
    """
    rho = np.zeros((2, 2), dtype=complex)
    for _ in range(shots):
        s = prepare()
        channel_apply(s)
        rho += single_qubit_dm(s)
    return rho / shots


def bloch(rho):
    return np.array([
        np.real(np.trace(rho @ _X)),
        np.real(np.trace(rho @ _Y)),
        np.real(np.trace(rho @ _Z)),
    ])


def purity(rho):
    return float(np.real(np.trace(rho @ rho)))


# ---------------------------------------------------------------------------
# (a) import works
# ---------------------------------------------------------------------------

def test_import_and_availability():
    import moonlab
    assert moonlab._NOISE_AVAILABLE is True
    for name in ("DepolarizingChannel", "AmplitudeDamping", "PhaseDamping",
                 "BitFlip", "PhaseFlip", "BitPhaseFlip", "ThermalRelaxation",
                 "ReadoutError", "DeviceNoiseModel"):
        assert name in moonlab.__all__
        assert hasattr(moonlab, name)


# ---------------------------------------------------------------------------
# (b) physical effects
# ---------------------------------------------------------------------------

def test_depolarizing_p075_drives_to_maximally_mixed():
    """The C depolarizing convention (1-p)rho + (p/3)(XrhoX+YrhoY+ZrhoZ)
    saturates the maximally mixed state at p = 3/4 (see noise.h): on |0>
    the averaged Bloch vector -> 0 and purity -> 0.5.
    """
    ch = DepolarizingChannel(probability=0.75, seed=1)

    def prep():
        return QuantumState(1)  # |0>

    rho = averaged_dm(prep, lambda s: ch.apply(s, 0), shots=8000)
    b = bloch(rho)
    assert np.linalg.norm(b) < 0.05, f"Bloch vector not shrunk: {b}"
    assert abs(purity(rho) - 0.5) < 0.02, f"purity {purity(rho)} not ~0.5"


def test_depolarizing_p1_residual_bloch_is_minus_third():
    """At p=1 (no identity branch) on |0> the three Paulis fire with equal
    weight: X,Y send |0>->|1> (Z=-1), Z keeps |0> (Z=+1), so <Z> -> -1/3.
    Documents that p=1 is not the maximally mixed point for this channel.
    """
    ch = DepolarizingChannel(probability=1.0, seed=1)

    def prep():
        return QuantumState(1)

    rho = averaged_dm(prep, lambda s: ch.apply(s, 0), shots=8000)
    assert abs(bloch(rho)[2] - (-1.0 / 3.0)) < 0.03, f"<Z>={bloch(rho)[2]}"


def test_depolarizing_p0_is_noop():
    ch = DepolarizingChannel(probability=0.0, seed=2)
    s = QuantumState(1)
    s.h(0)
    before = s.get_statevector()
    ch.apply(s, 0)
    after = s.get_statevector()
    assert np.allclose(before, after), "p=0 depolarizing altered the state"


def test_depolarizing_partial_shrinks_bloch_by_factor():
    """Depolarizing shrinks the Bloch vector by (1 - 4p/3) on a |+> state."""
    p = 0.3
    ch = DepolarizingChannel(probability=p, seed=7)

    def prep():
        s = QuantumState(1)
        s.h(0)  # |+>, Bloch = (1, 0, 0)
        return s

    rho = averaged_dm(prep, lambda s: ch.apply(s, 0), shots=8000)
    bx = bloch(rho)[0]
    expected = 1.0 - 4.0 * p / 3.0
    assert abs(bx - expected) < 0.03, f"<X>={bx}, expected ~{expected}"


def test_amplitude_damping_gamma1_relaxes_one_to_zero():
    """gamma=1 amplitude damping on |1> deterministically gives |0>."""
    ch = AmplitudeDamping(gamma=1.0, seed=3)
    s = QuantumState(1)
    s.x(0)  # |1>
    ch.apply(s, 0)
    assert s.probability(0) > 1.0 - 1e-9, f"P(0)={s.probability(0)}"
    assert s.probability(1) < 1e-9


def test_amplitude_damping_population_decay_rate():
    """Averaged P(1) after damping ~ (1 - gamma) * initial P(1)."""
    gamma = 0.4
    ch = AmplitudeDamping(gamma=gamma, seed=11)

    def prep():
        s = QuantumState(1)
        s.x(0)
        return s

    rho = averaged_dm(prep, lambda s: ch.apply(s, 0), shots=8000)
    p1 = float(np.real(rho[1, 1]))
    assert abs(p1 - (1.0 - gamma)) < 0.03, f"P(1)={p1}, expected ~{1 - gamma}"


def test_amplitude_damping_gamma0_noop():
    ch = AmplitudeDamping(gamma=0.0, seed=4)
    s = QuantumState(1)
    s.h(0)
    before = s.get_statevector()
    ch.apply(s, 0)
    assert np.allclose(before, s.get_statevector())


def test_phase_damping_kills_coherence_preserves_population():
    """Phase damping on |+> destroys <X> but keeps P(0)=P(1)=1/2."""
    ch = PhaseDamping(gamma=1.0, seed=5)

    def prep():
        s = QuantumState(1)
        s.h(0)
        return s

    rho = averaged_dm(prep, lambda s: ch.apply(s, 0), shots=8000)
    b = bloch(rho)
    assert abs(b[0]) < 0.05, f"<X>={b[0]} should vanish"
    assert abs(np.real(rho[0, 0]) - 0.5) < 0.03
    assert abs(np.real(rho[1, 1]) - 0.5) < 0.03


def test_bit_flip_averages_populations():
    """Bit flip with p on |0>: averaged P(1) ~ p."""
    p = 0.25
    ch = BitFlip(p, seed=6)

    def prep():
        return QuantumState(1)

    rho = averaged_dm(prep, lambda s: ch.apply(s, 0), shots=8000)
    p1 = float(np.real(rho[1, 1]))
    assert abs(p1 - p) < 0.03, f"P(1)={p1}, expected ~{p}"


def test_phase_flip_p1_flips_sign_of_plus():
    """Z on |+> gives |->: deterministic <X> = -1 at p=1."""
    ch = PhaseFlip(1.0, seed=8)
    s = QuantumState(1)
    s.h(0)
    ch.apply(s, 0)
    rho = single_qubit_dm(s)
    assert bloch(rho)[0] < -1.0 + 1e-6


def test_bit_phase_flip_p1_applies_y():
    """Y on |0> gives i|1>: P(1) = 1 at p=1."""
    ch = BitPhaseFlip(1.0, seed=9)
    s = QuantumState(1)
    ch.apply(s, 0)
    assert s.probability(1) > 1.0 - 1e-9


def test_thermal_relaxation_relaxes_excited_population():
    """T1 relaxation reduces excited-state population of |1> on average."""
    # gate_time comparable to T1 for a visible effect.
    ch = ThermalRelaxation(t1=50e-9, t2=70e-9, gate_time=30e-9, seed=12)
    gamma1 = 1.0 - np.exp(-30e-9 / 50e-9)

    def prep():
        s = QuantumState(1)
        s.x(0)
        return s

    rho = averaged_dm(prep, lambda s: ch.apply(s, 0), shots=8000)
    p1 = float(np.real(rho[1, 1]))
    assert abs(p1 - (1.0 - gamma1)) < 0.04, f"P(1)={p1}, expected ~{1 - gamma1}"


def test_thermal_relaxation_excited_population_nonzero_fixed_point():
    """With excited_population > 0, |0> gains some excited population."""
    ch = ThermalRelaxation(t1=20e-9, t2=30e-9, gate_time=60e-9,
                           excited_population=0.5, seed=13)

    def prep():
        return QuantumState(1)  # |0>

    rho = averaged_dm(prep, lambda s: ch.apply(s, 0), shots=8000)
    p1 = float(np.real(rho[1, 1]))
    assert p1 > 0.1, f"excited population not populated: P(1)={p1}"


def test_readout_error_flip_statistics():
    """ReadoutError.apply_outcome flips 0->1 at the configured rate."""
    ro = ReadoutError(error_0_to_1=0.3, error_1_to_0=0.1, seed=14)
    n = 20000
    ones = sum(ro.apply_outcome(0) for _ in range(n))
    assert abs(ones / n - 0.3) < 0.02
    zeros = sum(1 - ro.apply_outcome(1) for _ in range(n))
    assert abs(zeros / n - 0.1) < 0.02


def test_readout_error_zero_is_noop():
    ro = ReadoutError(error_0_to_1=0.0, error_1_to_0=0.0, seed=15)
    assert all(ro.apply_outcome(0) == 0 for _ in range(1000))
    assert all(ro.apply_outcome(1) == 1 for _ in range(1000))


def test_readout_from_confusion_matrix():
    ro = ReadoutError([[0.99, 0.01], [0.02, 0.98]], seed=16)
    assert abs(ro.error_0_to_1 - 0.01) < 1e-12
    assert abs(ro.error_1_to_0 - 0.02) < 1e-12


def test_device_noise_model_single_qubit_depolarizes():
    """DeviceNoiseModel single-qubit profile shrinks a |+> Bloch vector."""
    dev = DeviceNoiseModel(num_qubits=2, single_qubit_error=0.5, seed=21)

    def prep():
        s = QuantumState(2)
        s.h(0)
        return s

    rho = np.zeros((2, 2), dtype=complex)
    shots = 4000
    for _ in range(shots):
        s = prep()
        dev.apply_single(s, 0)
        psi = s.get_statevector().reshape(2, 2)
        # partial trace over qubit 1 (index bit 1); qubit 0 is the low bit.
        rho += psi @ psi.conj().T
    rho /= shots
    bx = np.real(np.trace(rho @ _X))
    assert bx < 0.9, f"device noise did not depolarize qubit 0: <X>={bx}"


def test_device_noise_model_zero_error_is_noop():
    dev = DeviceNoiseModel(num_qubits=1, single_qubit_error=0.0, seed=22)
    s = QuantumState(1)
    s.h(0)
    before = s.get_statevector()
    dev.apply_single(s, 0)
    assert np.allclose(before, s.get_statevector()), "zero-error device model changed state"


def test_device_noise_model_thermal_profile_relaxes():
    """DeviceNoiseModel with T1/T2 relaxes an excited qubit on average."""
    dev = DeviceNoiseModel(num_qubits=1, t1=40e-9, t2=60e-9, gate_time=40e-9,
                           single_qubit_error=0.0, seed=23)

    p1_sum = 0.0
    shots = 6000
    for _ in range(shots):
        s = QuantumState(1)
        s.x(0)
        dev.apply_single(s, 0)
        p1_sum += s.probability(1)
    p1 = p1_sum / shots
    gamma1 = 1.0 - np.exp(-40e-9 / 40e-9)
    assert abs(p1 - (1.0 - gamma1)) < 0.05, f"P(1)={p1}, expected ~{1 - gamma1}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
