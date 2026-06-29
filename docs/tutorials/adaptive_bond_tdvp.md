# Archived Moonlab Documentation: Tutorial: Adaptive-bond TDVP

This local Moonlab document is retained as archived vendor text for the QGTL integration audit; current supported claims are measured by `scripts/moonlab_doc_claim_audit.py` and grounded against `external/moonlab/README.md`, `external/moonlab/CMakeLists.txt`, and `docs/MOONLAB_OPEN_CORE_INTEGRATION.md`.

The historical text below is preserved as an archival snapshot, not as current release documentation.

```text
# Tutorial: Adaptive-bond TDVP

This tutorial walks through Moonlab's v0.4 time-dependent variational
principle (TDVP) integrator with the entropy-feedback PID controller
that adapts each bond's truncation dimension to a target accuracy
budget.  By the end you will have evolved an MPS state in real and
imaginary time from C, Python, and Rust, and verified the three
physics-validation criteria that pin the controller's correctness.

Prerequisites:

- A Release build of `libquantumsim` (`build_release/`) configured
  with `-DQSIM_BUILD_TESTS=ON -DQSIM_BUILD_BENCHMARKS=ON`.
- Working familiarity with two-site TDVP and the operator-sum
  representation of an MPS at the level of Schollwoeck [1] and
  Haegeman et al. [2].
- For the Python sections: `numpy` and the optional
  `MOONLAB_LIB_DIR` environment variable pointing at the build that
  contains `libquantumsim.0.4.0.dylib` (or later).
- For the Rust sections: the workspace at `bindings/rust/moonlab`,
  built once with `MOONLAB_LIB_DIR=<...>/build_release cargo build`.

The complete C, Python, and Rust API contracts live in
[`../reference/tdvp-api.md`](../reference/tdvp-api.md).  The design
note that motivates the algorithm is at
[`../research/adaptive_bond_tdvp.md`](../research/adaptive_bond_tdvp.md).

## 1. The problem the controller solves

Two-site TDVP truncates each bond after every adjacent-pair update
by keeping only the top `chi` singular values.  Picking `chi` too
small drops physical entanglement; picking it too large wastes
compute on bonds that need a fraction of the kept singular weight.
The v0.3 integrator solved this with a global, user-supplied
`max_bond_dim` cap, paid uniformly across the chain regardless of
where the entanglement actually concentrated.

The v0.4 adaptive-bond controller [3] measures the post-SVD
spectrum entropy `S` at each bond and the truncated entropy
`S_chi` at the current working chi.  The error signal
`e = S - S_chi` drives a PID update that adjusts the per-bond chi
toward the configured `target_entropy_error`.  Each inter-site
bond carries its own controller state; integral and derivative
terms accumulate across sweeps.

## 2. Building a real-time evolution from C

Real-time two-site TDVP is symplectic on the MPS manifold [2] and
therefore conserves `<H>` for a time-independent Hamiltonian.  The
PID controller only changes which singular values are kept; it
must not break that conservation.

[archived fence delimiter: ```c]
#include "moonlab/algorithms/tensor_network/tdvp.h"
#include "moonlab/algorithms/tensor_network/dmrg.h"
#include "moonlab/algorithms/tensor_network/tn_state.h"
#include <stdio.h>

int main(void) {
    const uint32_t n = 8;
    mpo_t *mpo = mpo_heisenberg_create(n, /*J=*/1.0, /*Delta=*/1.0,
                                        /*h=*/0.0);

    tn_state_config_t mcfg = tn_state_config_create(/*max_bond=*/32,
                                                     /*cutoff=*/1e-12);
    tn_mps_state_t *mps = dmrg_init_random_mps(n, /*chi_init=*/8, &mcfg);

    tdvp_config_t cfg = tdvp_config_adaptive(/*eps_S=*/1e-3);
    cfg.evolution_type            = TDVP_REAL_TIME;
    cfg.dt                        = 0.02;
    cfg.adaptive_bond.chi_ceiling = 32;
    cfg.max_bond_dim              = 32;

    tdvp_engine_t *engine = tdvp_engine_create(mps, mpo, &cfg);

    tdvp_result_t result = {0};
    double E0 = 0.0;
    for (int step = 0; step < 5; step++) {
        tdvp_step(engine, &result);
        if (step == 0) E0 = result.energy;
        const double drift =
            (result.energy - E0) / (E0 != 0.0 ? E0 : 1.0);
        printf("step %d: E = %+.6f  (drift = %.2e)\n",
               step, result.energy, drift);
    }

    tdvp_result_clear(&result);
    tdvp_engine_free(engine);
    tn_mps_free(mps);
    mpo_free(mpo);
    return 0;
}
[archived fence delimiter: ```]

The expected output on `libquantumsim.0.4.0` shows the relative
energy drift staying near `2e-5` over five steps, well inside the
symplectic-integrator envelope.  This is the acceptance criterion
pinned by [`tests/unit/test_tdvp_adaptive_energy_conservation.c`](../../tests/unit/test_tdvp_adaptive_energy_conservation.c).

## 3. The same evolution from Python

`moonlab.tdvp` (since v0.4.0) is an idiomatic ctypes wrapper around
the C surface:

[archived fence delimiter: ```python]
from moonlab.tdvp import (
    TdvpConfig, TdvpEngine, EvolutionType,
    mpo_heisenberg, random_mps,
)

mpo = mpo_heisenberg(num_sites=8, J=1.0, Delta=1.0, h=0.0)
mps = random_mps(num_sites=8, chi_init=8, max_bond_dim=32)

config = TdvpConfig.adaptive(target_entropy_error=1e-3)
config.evolution_type = EvolutionType.REAL_TIME
config.dt = 0.02
config.adaptive_bond.chi_ceiling = 32
config.max_bond_dim = 32

engine = TdvpEngine(mps, mpo, config)
E0 = engine.step().energy
for step in range(5):
    result = engine.step()
    drift = (result.energy - E0) / abs(E0)
    print(f"step {step}: E = {result.energy:+.6f}  (drift = {drift:.2e})")
[archived fence delimiter: ```]

`result.bond_chi_distribution` is exposed as a NumPy `uint32` array
of length `num_qubits - 1`, ready for plotting or assertion.

## 4. Imaginary-time evolution: ground states

Imaginary-time TDVP projects toward the ground state of `H` at a
rate set by the spectral gap.  On the 8-site critical
transverse-field Ising model (`g = 1`), 30 imag-time steps at
`dt = 0.05` reach `|E - E_DMRG| / |E_DMRG| < 2%`.

[archived fence delimiter: ```python]
from moonlab.tdvp import (
    TdvpConfig, TdvpEngine, EvolutionType,
    mpo_tfim, random_mps,
)

mpo = mpo_tfim(num_sites=8, J=1.0, h=1.0)        # g = h / J = 1
mps = random_mps(num_sites=8, chi_init=8, max_bond_dim=32)

config = TdvpConfig.adaptive(target_entropy_error=1e-3)
config.evolution_type = EvolutionType.IMAGINARY_TIME
config.dt = 0.05
config.adaptive_bond.chi_ceiling = 32
config.max_bond_dim = 32

engine = TdvpEngine(mps, mpo, config)
energies = [engine.step().energy for _ in range(30)]

# Reference: DMRG at chi = 32 returns E0 ~= -9.84 on this system.
print(f"E_final  = {energies[-1]:+.4f}")
print(f"E_first  = {energies[0]:+.4f}")
print(f"dropped  = {energies[0] - energies[-1]:+.4f}")
[archived fence delimiter: ```]

The energy must drop monotonically (modulo PID overshoot) over the
30 steps -- the acceptance test pinned by
[`tests/unit/test_tdvp_adaptive_tfim_ground.c`](../../tests/unit/test_tdvp_adaptive_tfim_ground.c)
requires `|E_final - E_DMRG| / |E_DMRG| < 3%` and reproducibly
achieves 1.98%.

## 5. Rust: full control over the bond-chi trajectory

The Rust wrapper at `moonlab::tdvp` exposes the same surface plus
RAII handles and zero-cost iteration over `Vec<u32>`
distributions:

[archived fence delimiter: ```rust]
use moonlab::tdvp::{
    EvolutionType, Mpo, Mps, TdvpConfig, TdvpEngine,
};

fn main() {
    let mpo = Mpo::heisenberg(8, 1.0, 1.0, 0.0).unwrap();
    let mps = Mps::random(8, 8, 32, 1e-12).unwrap();

    let mut config = TdvpConfig::adaptive(1e-3);
    config.evolution_type = EvolutionType::ImaginaryTime;
    config.dt = 0.05;
    config.adaptive_bond.chi_ceiling = 32;
    config.max_bond_dim = 32;

    let mut engine = TdvpEngine::new(mps, mpo, config).unwrap();
    for step in 0..30 {
        let result = engine.step().unwrap();
        let mean_chi: f64 = result
            .bond_chi_distribution
            .iter()
            .map(|&c| c as f64)
            .sum::<f64>()
            / result.bond_chi_distribution.len() as f64;
        if step % 5 == 0 {
            println!(
                "step {step:2}: E = {:+.6}, mean chi = {mean_chi:.2}",
                result.energy
            );
        }
    }
}
[archived fence delimiter: ```]

Run with
`MOONLAB_LIB_DIR=$(pwd)/build_release cargo run --example tdvp_demo`;
the example program ships a richer version of this snippet plus a
real-time energy-conservation section and a PID-gain micro-sweep.

## 6. Inspecting and asserting on the bond-chi distribution

The controller's per-bond decisions are visible through three
parallel surfaces:

| Language | Surface |
|---|---|
| C | `result.bond_chi_distribution` (heap `uint32_t *`, length `result.n_bonds`); accessor `tdvp_bond_chi(engine, b)`. |
| Python | `result.bond_chi_distribution` (NumPy `uint32` array); accessor `engine.bond_chi(b)`. |
| Rust | `result.bond_chi_distribution` (`Vec<u32>`); accessor `engine.bond_chi(b)`. |

For history tracking across all steps of a run, the C `tdvp_history_t`
struct keeps a flat row-major `capacity * n_bonds` buffer.  Use it
when you want to plot `chi(bond, time)` heatmaps:

[archived fence delimiter: ```c]
tdvp_history_t *hist = tdvp_history_create(/*initial_capacity=*/100);
for (int step = 0; step < 100; step++) {
    tdvp_step(engine, &result);
    tdvp_history_add(hist, &result);
}
// hist->bond_chi_history is now a (100, n - 1) uint32_t array.
tdvp_history_free(hist);
[archived fence delimiter: ```]

## 7. Choosing the PID gains

The reference-paper gains [3] (`kp = 0.5`, `ki = 0.05`, `kd = 0.1`,
`alpha = 8`, `chi_floor = 4`, `chi_ceiling = 4096`) are tuned for
24-site Heisenberg dynamics.  Moonlab's stability sweep [4] verified
that these defaults stay stable across a 3 x 3 x 3 grid spanning
factors of two around each gain on 8-site Heisenberg systems.  If
your model has a substantially different entanglement-growth profile:

- Lower `kp` and `kd` if you see chi oscillating between successive
  steps.
- Raise `target_entropy_error` if the controller is paying too
  much (per-bond chi saturates near `chi_ceiling` and wall time
  scales like `chi^3`).
- Lower `target_entropy_error` if observables look noisy at
  expected accuracy.
- Pin `chi_floor` to the value that the gap structure of `H`
  demands; `chi_floor = 4` is safe for most spin chains but you may
  need `chi_floor = 16` or higher for critical 2D models.

Always re-run the energy-conservation and stability checks after
tuning the gains.

## 8. Acceptance tests as documentation

Every numerical claim in this tutorial is pinned by one of the
v0.4 unit tests:

| Acceptance criterion | Test |
|---|---|
| Legacy callers unchanged | `tests/unit/test_tdvp_adaptive_config.c` |
| Real-time energy conservation | `tests/unit/test_tdvp_adaptive_energy_conservation.c` |
| Imag-time TFIM ground state | `tests/unit/test_tdvp_adaptive_tfim_ground.c` |
| PID stability sweep | `tests/unit/test_tdvp_adaptive_pid_stability.c` |

Plus the cross-binding end-to-end coverage:

- `bindings/python/tests/test_tdvp.py` (9 cases).
- `bindings/rust/moonlab/src/tdvp.rs` `#[cfg(test)] mod tests` (5 cases).

If your local `ctest` run diverges from the numerical values quoted
here by more than the stated tolerances, please open an issue.

## References

[1] U. Schollwoeck, "The density-matrix renormalization group in
    the age of matrix product states", *Ann. Phys.* **326**, 96
    (2011).

[2] J. Haegeman, C. Lubich, I. Oseledets, B. Vandereycken, and
    F. Verstraete, "Unifying time evolution and optimization with
    matrix product states", *Phys. Rev. B* **94**, 165116 (2016).
    Two-site TDVP foundations.

[3] *Adaptive bond TDVP via entropy-feedback PID*, arXiv:2604.03960.

[4] Moonlab v0.4.0 release notes, `CHANGELOG.md`.

## See also

- [`../reference/tdvp-api.md`](../reference/tdvp-api.md): full C /
  Python / Rust API reference.
- [`../research/adaptive_bond_tdvp.md`](../research/adaptive_bond_tdvp.md):
  algorithm derivation and acceptance criteria.
- [`getting_started.md`](getting_started.md): build instructions
  and first quantum program.
- [`mpdo_noise.md`](mpdo_noise.md): companion v0.3 module (noise
  simulation via MPDO).
- [`topological_band_structure.md`](topological_band_structure.md):
  companion v0.3 module (quantum geometric tensor).
```
