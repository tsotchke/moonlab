"""moonlab's decoders under a real `sinter.collect` run.

This is the end-to-end integration proof: sinter builds the circuits,
samples them, ships tasks to worker processes, and asks each registered
decoder for predictions, exactly as a published threshold study would.
moonlab's correlated union-find decoder is compared head to head with
sinter's built-in ``pymatching`` decoder on the same tasks, and the
comparison table is printed with shot counts, error counts, logical
error rates and the likelihood-ratio confidence intervals sinter itself
reports.

Marked ``slow``: it spawns worker processes and samples tens of
thousands of shots.  Run with ``-m slow`` (or without ``-m "not slow"``).
"""

import pickle

import pytest

stim = pytest.importorskip("stim")
sinter = pytest.importorskip("sinter")
pytest.importorskip("pymatching")

from moonlab.qec.sinter import (  # noqa: E402
    MoonlabCorrelatedDecoder,
    MoonlabUFDecoder,
    make_custom_decoders,
)

DISTANCES = [3, 5]
NOISE_P = 0.005
MAX_SHOTS = 40000
DECODERS = ["pymatching", "moonlab_uf_correlated"]


def _tasks():
    tasks = []
    for d in DISTANCES:
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=d, rounds=d,
            after_clifford_depolarization=NOISE_P,
            before_measure_flip_probability=NOISE_P,
            after_reset_flip_probability=NOISE_P,
            before_round_data_depolarization=NOISE_P)
        tasks.append(sinter.Task(
            circuit=circuit,
            json_metadata={"d": d, "r": d, "p": NOISE_P},
        ))
    return tasks


def test_custom_decoders_are_registered_and_picklable():
    """sinter ships decoders to workers, so they must survive a pickle."""
    decoders = make_custom_decoders()
    assert set(decoders) == {"moonlab_uf", "moonlab_uf_correlated"}
    assert isinstance(decoders["moonlab_uf"], MoonlabUFDecoder)
    assert isinstance(decoders["moonlab_uf_correlated"],
                      MoonlabCorrelatedDecoder)
    assert decoders["moonlab_uf"].correlated is False
    assert decoders["moonlab_uf_correlated"].correlated is True
    for name, decoder in decoders.items():
        revived = pickle.loads(pickle.dumps(decoder))
        assert type(revived) is type(decoder), name
        assert revived == decoder


def test_compiled_decoder_round_trips_a_dem():
    """The compiled decoder honours sinter's bit-packed contract."""
    import numpy as np

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=3, rounds=3,
        after_clifford_depolarization=NOISE_P,
        before_measure_flip_probability=NOISE_P,
        after_reset_flip_probability=NOISE_P)
    dem = circuit.detector_error_model(decompose_errors=True)
    compiled = MoonlabCorrelatedDecoder().compile_decoder_for_dem(dem=dem)

    shots = 256
    packed_det = circuit.compile_detector_sampler(seed=11).sample(
        shots, separate_observables=True, bit_packed=True)[0]
    predictions = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=packed_det)
    assert predictions.dtype == np.uint8
    assert predictions.shape == (shots, (dem.num_observables + 7) // 8)


@pytest.mark.slow
def test_sinter_collect_beats_or_matches_pymatching():
    stats = sinter.collect(
        num_workers=2,
        tasks=_tasks(),
        decoders=DECODERS,
        custom_decoders=make_custom_decoders(),
        max_shots=MAX_SHOTS,
        print_progress=False,
    )
    assert stats, "sinter.collect returned no statistics"

    table = {}
    for stat in stats:
        d = stat.json_metadata["d"]
        fit = sinter.fit_binomial(
            num_shots=stat.shots - stat.discards,
            num_hits=stat.errors,
            max_likelihood_factor=1000.0)
        table[(stat.decoder, d)] = (stat, fit)

    header = (f"{'decoder':<24} {'d':>2} {'shots':>8} {'errors':>7} "
              f"{'logical error rate':>19} {'CI low':>10} {'CI high':>10}")
    print("\n" + header)
    print("-" * len(header))
    for decoder in DECODERS:
        for d in DISTANCES:
            stat, fit = table[(decoder, d)]
            shots = stat.shots - stat.discards
            print(f"{decoder:<24} {d:>2} {shots:>8} {stat.errors:>7} "
                  f"{fit.best:>19.6f} {fit.low:>10.6f} {fit.high:>10.6f}")
    print()

    for d in DISTANCES:
        assert ("pymatching", d) in table, f"pymatching missing for d={d}"
        assert ("moonlab_uf_correlated", d) in table, \
            f"moonlab_uf_correlated missing for d={d}"

        pm_stat, pm_fit = table[("pymatching", d)]
        ml_stat, ml_fit = table[("moonlab_uf_correlated", d)]

        assert ml_stat.shots > 0
        assert ml_stat.shots - ml_stat.discards >= MAX_SHOTS // 2, (
            f"d={d}: moonlab only completed {ml_stat.shots} shots")

        assert ml_fit.best <= pm_fit.high, (
            f"d={d}: moonlab_uf_correlated logical error rate "
            f"{ml_fit.best:.6f} exceeds the top of pymatching's confidence "
            f"interval [{pm_fit.low:.6f}, {pm_fit.high:.6f}]")

    # A distance-5 surface code must suppress errors relative to d=3;
    # a decoder that silently returned zeros would not.
    ml3 = table[("moonlab_uf_correlated", 3)][1].best
    ml5 = table[("moonlab_uf_correlated", 5)][1].best
    assert ml5 < ml3, (
        f"no error suppression with distance: d=3 {ml3:.6f}, d=5 {ml5:.6f}")
    assert ml3 > 0.0, "moonlab reported zero logical errors, which is suspect"
