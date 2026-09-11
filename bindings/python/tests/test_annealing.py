import math
import pytest

from moonlab.annealing import (
    AnnealConfig,
    AnnealSchedule,
    SchedulePoint,
    anneal_ising,
    anneal_qubo,
    make_pause_schedule,
    make_quench_schedule,
    qubo_to_ising,
    reverse_anneal_ising,
    reverse_anneal_qubo,
    zephyr_embed_ising,
    zephyr_find_clique_embedding,
    zephyr_graph_create,
    zephyr_unembed_samples,
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


def test_pause_and_quench_schedules():
    pause_pts = make_pause_schedule(s_pause=0.4, pause_start_frac=0.3, pause_duration_frac=0.2)
    assert len(pause_pts) == 4
    assert pause_pts[0] == SchedulePoint(0.0, 0.0)
    assert pause_pts[1] == SchedulePoint(0.3, 0.4)
    assert pause_pts[2] == SchedulePoint(0.5, 0.4)
    assert pause_pts[3] == SchedulePoint(1.0, 1.0)

    quench_pts = make_quench_schedule(s_quench=0.6, quench_start_frac=0.5)
    assert len(quench_pts) == 3
    assert quench_pts[0] == SchedulePoint(0.0, 0.0)
    assert quench_pts[1] == SchedulePoint(0.5, 0.6)
    assert quench_pts[2] == SchedulePoint(1.0, 1.0)

    config = AnnealConfig(
        total_time=10.0,
        num_steps=1000,
        schedule=AnnealSchedule.PIECEWISE,
        schedule_points=pause_pts,
        seed=42,
    )
    res = anneal_ising([-1.0], [[0.0]], config=config)
    assert res.ground_bitstring == 0
    assert res.success_probability > 0.95


def test_reverse_anneal_ising_and_qubo():
    # Ferromagnetic 2-qubit system: ground states |00>, |11> (E=-1.0),
    # excited states |01>, |10> (E=+1.0).
    # Start from excited state |01> (bitstring 1)
    res_ising = reverse_anneal_ising(
        [0.0, 0.0],
        [[0.0, -1.0], [-1.0, 0.0]],
        initial_bitstring=1,
        s_target=0.45,
        hold_fraction=0.2,
        config=AnnealConfig(total_time=10.0, num_steps=1000, num_samples=128, seed=42),
    )
    assert res_ising.best_energy == -1.0
    assert res_ising.best_bitstring in (0, 3)

    # QUBO where ground states are |10> (1) and |01> (2) with E=0.0;
    # excited states are |00> (0) and |11> (3) with E=1.0.
    # Start from excited state |00> (bitstring 0)
    Q = [[-1.0, 1.0], [1.0, -1.0]]
    res_qubo = reverse_anneal_qubo(
        Q,
        offset=1.0,
        initial_bitstring=0,
        s_target=0.45,
        hold_fraction=0.2,
        config=AnnealConfig(total_time=10.0, num_steps=1000, num_samples=128, seed=42),
    )
    assert res_qubo.best_energy == 0.0
    assert res_qubo.best_bitstring in (1, 2)


def test_per_qubit_anneal_offsets():
    config = AnnealConfig(
        total_time=8.0,
        num_steps=800,
        num_samples=64,
        anneal_offsets=[-0.05, 0.05],
        seed=123,
    )
    res = anneal_ising([-1.0, -1.0], [[0.0, 0.0], [0.0, 0.0]], config=config)
    assert res.ground_bitstring == 0
    assert res.success_probability > 0.9


def test_zephyr_graph_creation():
    z1 = zephyr_graph_create(1)
    assert z1.m == 1 and z1.t == 4
    assert z1.num_qubits == 48
    assert z1.num_couplers == 280
    assert len(z1.couplers) == 280

    z2 = zephyr_graph_create(2)
    assert z2.m == 2 and z2.t == 4
    assert z2.num_qubits == 160
    assert z2.num_couplers == 1224
    assert len(z2.couplers) == 1224


def test_zephyr_clique_embedding_and_unembed():
    emb = zephyr_find_clique_embedding(4, 1)
    assert emb.num_logical == 4
    assert len(emb.chains) == 4
    for chain in emb.chains:
        assert len(chain) == 2

    log_h = [0.2, -0.3, 0.1, -0.4]
    log_J = [
        [0.0, -0.5, -0.3, -0.2],
        [-0.5, 0.0, -0.4, -0.1],
        [-0.3, -0.4, 0.0, -0.6],
        [-0.2, -0.1, -0.6, 0.0],
    ]
    phys_h, phys_J = zephyr_embed_ising(1, emb, log_h, log_J, chain_strength=2.5)
    assert len(phys_h) == 48
    assert len(phys_J) == 48
    assert len(phys_J[0]) == 48

    all_zeros = 0
    all_ones = sum(1 << q for chain in emb.chains for q in chain)
    broken = 1 << emb.chains[0][0]

    log_samples, break_fracs = zephyr_unembed_samples(emb, [all_zeros, all_ones, broken])
    assert log_samples == [0, 15, 0]
    assert math.isclose(break_fracs[0], 0.0, abs_tol=1e-12)
    assert math.isclose(break_fracs[1], 0.0, abs_tol=1e-12)
    assert math.isclose(break_fracs[2], 0.25, abs_tol=1e-12)

    emb8 = zephyr_find_clique_embedding(8, 2)
    assert emb8.num_logical == 8
    emb12 = zephyr_find_clique_embedding(12, 2)
    assert emb12.num_logical == 12

