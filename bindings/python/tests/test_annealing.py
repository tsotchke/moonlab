import math

from moonlab.annealing import (
    AnnealConfig,
    AnnealSchedule,
    anneal_ising,
    anneal_qubo,
    qubo_to_ising,
)


def test_qubo_conversion_energy_parity():
    Q = [[-1.5, 0.7, -0.2], [0.3, 2.0, 0.9], [-0.4, 0.1, -0.5]]
    h, J, offset = qubo_to_ising(Q, offset=0.25)
    for bits in range(8):
        x = [(bits >> i) & 1 for i in range(3)]
        z = [1 - 2 * value for value in x]
        qubo = 0.25 + sum(x[i] * Q[i][j] * x[j] for i in range(3) for j in range(3))
        ising = offset + sum(h[i] * z[i] for i in range(3))
        ising += sum(J[i][j] * z[i] * z[j] for i in range(3) for j in range(i + 1, 3))
        assert abs(qubo - ising) < 2e-14


def test_seeded_qubo_anneal_is_reproducible_and_exact():
    config = AnnealConfig(
        total_time=12.0,
        num_steps=1200,
        num_samples=128,
        seed=0x123456789ABCDEF0,
        schedule=AnnealSchedule.COSINE,
    )
    Q = [[-1.0, 1.0], [1.0, -1.0]]
    first = anneal_qubo(Q, offset=1.0, config=config)
    second = anneal_qubo(Q, offset=1.0, config=config)
    assert first.effective_seed == config.seed
    assert first.samples == second.samples
    assert first.sample_energies == second.sample_energies
    assert first.ground_degeneracy == 2
    assert first.ground_energy == 0.0
    assert first.problem_gap == 1.0
    assert first.best_energy == 0.0
    assert first.success_probability > 0.95
    assert math.isclose(first.final_norm, 1.0, abs_tol=1e-11)


def test_one_qubit_ising_ground_state():
    result = anneal_ising(
        [-1.0],
        [[0.0]],
        config=AnnealConfig(total_time=12.0, num_steps=1200, num_samples=64, seed=7),
    )
    assert result.ground_bitstring == 0
    assert result.ground_energy == -1.0
    assert result.problem_gap == 2.0
    assert result.success_probability > 0.98
