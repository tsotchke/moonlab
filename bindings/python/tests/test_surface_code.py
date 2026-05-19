"""Python integration tests for moonlab.surface_code.

Mirrors the Rust unit tests added in v0.5.12.  Pins the surface
against regression with seven physical-meaningful invariants on the
distance-3 rotated surface code.
"""

from __future__ import annotations

import pytest

from moonlab.surface_code import SurfaceCode


def test_distance_3_layout():
    code = SurfaceCode(distance=3, rng_seed=1)
    assert code.distance == 3
    assert code.num_data_qubits == 9
    assert code.num_ancillas_per_sector == 4


def test_rejects_even_distance():
    with pytest.raises(ValueError):
        SurfaceCode(distance=2)
    with pytest.raises(ValueError):
        SurfaceCode(distance=4)


def test_rejects_distance_one():
    with pytest.raises(ValueError):
        SurfaceCode(distance=1)


def test_data_index_maps_row_col():
    code = SurfaceCode(distance=3, rng_seed=1)
    # Each (row, col) pair returns a distinct linear index in [0, 9).
    indices = {code.data_index(r, c) for r in range(3) for c in range(3)}
    assert len(indices) == 9
    for idx in indices:
        assert 0 <= idx < 9


def test_data_index_out_of_range():
    code = SurfaceCode(distance=3, rng_seed=1)
    with pytest.raises(IndexError):
        code.data_index(3, 0)
    with pytest.raises(IndexError):
        code.data_index(0, 3)


def test_z_stabilisers_idempotent_on_zero_state():
    code = SurfaceCode(distance=3, rng_seed=42)
    code.measure_z_syndromes()
    w0 = code.syndrome_weight()
    # Measuring Z stabilisers again on the same (now-stabilised)
    # state shouldn't change the syndrome weight.
    code.measure_z_syndromes()
    w1 = code.syndrome_weight()
    assert w0 == w1


def test_x_error_lights_z_stabilisers():
    code = SurfaceCode(distance=3, rng_seed=42)
    q = code.data_index(1, 1)
    code.apply_error(q, 'X')
    code.measure_z_syndromes()
    assert code.syndrome_weight() > 0


def test_z_error_lights_x_stabilisers():
    code = SurfaceCode(distance=3, rng_seed=7)
    q = code.data_index(1, 1)
    code.apply_error(q, 'Z')
    code.measure_x_syndromes()
    assert code.syndrome_weight() > 0


def test_unknown_error_type_rejected():
    code = SurfaceCode(distance=3, rng_seed=1)
    with pytest.raises(ValueError):
        code.apply_error(0, 'W')  # type: ignore[arg-type]


def test_qubit_index_out_of_range():
    code = SurfaceCode(distance=3, rng_seed=1)
    with pytest.raises(IndexError):
        code.apply_error(100, 'X')
