# Scaling / large-n differential -- historical findings and closure

Mission: push scale and circuit diversity past the adjacent-only cross-diff
corpus to find scale-dependent cross-backend divergences BEYOND the already-known
`n>=10` forward-circuit `tn_mps` bug. Backends: dense / tn_mps / Clifford tableau
/ MPS-var-D / DMRG / TDVP. Independent numpy statevector reference for n<=16;
structure-specific analytic checks for larger n. Report with repro; do not fix
`src/`; never loosen tolerances.

Every historical divergence below was confirmed with the **dense** backend still matching an
**independent** oracle (numpy statevector for the circuit corpus; a dense
Lanczos-Krylov `exp(-iHt)` verified against `scipy.linalg.expm` for TDVP; dense
power-iteration ED for DMRG/var-D). So dense is the trustworthy anchor and the
diverging backend is the one at fault.

Reproduce the circuit corpus (pure function of seed, PYTHONHASHSEED-independent):

    python3 tests/scaling/gen_scaling_corpus.py --out-dir DIR \
            --seed 0x5CA11B16 --qubits 10,12,14,16 --depths 8,32,128 --small-longrange
    scaling_diff DIR/corpus.txt --tn-max-n 14

Build the drivers standalone against a prebuilt libquantumsim:

    MOONLAB_LIB_DIR=/path/to/build cmake -S tests/scaling -B build/scaling
    cmake --build build/scaling -j2

---

## Summary

| id | subsystem | resolution | regression evidence | status |
|----|-----------|------------|---------------------|--------|
| S1 | tn_mps adjacent 2q gate | establish the required mixed-canonical gauge before the two-site SVD rescale | the full scaling circuit corpus now reports `new=0 known=0` | closed |
| S2 | real-time TDVP | the report came from a stale pre-projector-splitting build; current projector splitting is correct at bulk sites | `unit_tdvp_bulk_site` plus the n=10 TDVP-vs-Krylov scaling check | not a bug on current code |

The quarantine was removed after the complete harness passed with zero known
and zero new divergences. `KNOWN_DIVERGENCES.txt` is intentionally empty.

Clean **negative** results (no bug found -- these subsystems scale correctly):
Clifford tableau at n=50,100 (GHZ + linear-cluster stabilizer structure);
MPS-var-D optimizer at n=8,16,32 (variational bound, monotonicity, eps-escape
convergence, uint32 bond-width >255); DMRG ground energy vs ED at n=8,10,12.

---

## S1 -- tn_mps adjacent two-qubit gate lost norm on a "bulk" bond (closed)

**What the corpus caught.** Many `tn_mps` prob/expectation divergences at n as
low as 6, on forward-only, Toffoli-free circuits -- so NOT the reversed-CNOT bug
and (nominally) below the documented `n>=10` threshold. Examples (seed
`0x5CA11B16`), each independently reproduced by the sibling lane's `diff_backends`
driver at byte-identical magnitude:

| case id | seed | class | n | tn vs dense (prob) |
|---------|------|-------|---|--------------------|
| `clifford_local_n06_d008_s4379` | 1410620350602298233 | clifford_local (adjacent only) | 6 | 1.172e-02 |
| `mps_brick_n06_d008_s7606` | 3293429194283709958 | mps_brick (adjacent only) | 6 | 2.125e-01 |
| `clifford_longrange_n08_d008_s2468` | 16482231628993209448 | clifford_longrange | 8 | 2.344e-02 |
| `rot_longrange_n08_d004_s9549` | 16242786031782696265 | rot_longrange (depth 4!) | 8 | 9.788e-02 |

**Root cause (localized + confirmed).** Prefix-bisection puts the first
divergence at an *adjacent* 2q gate applied after both of its neighbours are
already entangled. Minimal hand repro -- 4 qubits, forward, pure Clifford,
all-adjacent:

    H(0)  CX(0,1)   # Bell pair on (0,1)  -> left bond non-trivial
    H(2)  CX(2,3)   # Bell pair on (2,3)  -> right bond non-trivial
    CZ(1,2)         # adjacent gate on the bulk bond (1,2)
    => tn norm = 0.5, EVERY probability halved; dense/reference exact.

Diagnostics that pin the mechanism:
- **norm = 0.5**, all amplitudes scaled by 1/sqrt(2): the state *shape* is right,
  only the overall normalization is lost -> a normalization bug, not indexing.
- **bond-dimension independent**: chi = 8,16,64,256,1024 all give dev 1.250e-01
  -> not truncation.
- **`bell1_cz` control passes** (one side product, `rr_dim=1`): dev 1.1e-16.
  The defect requires the left AND right outer bonds to BOTH be non-trivial.
- **`tn_mps_mixed_canonicalize(state, 1)` before the CZ fixes it** (dev -> 1.1e-16,
  norm -> 1.0). `left_canonicalize` and `normalize` do NOT fix it.

So `apply_gate_2q_adjacent` renormalizes the two-site tensor's singular values by
their Frobenius norm assuming mixed-canonical gauge at the bond, but the gate
path leaves the MPS in `TN_CANONICAL_NONE`. On a bulk bond `||theta||_F != 1`
and the rescale corrupts the norm.

**Relation to the known bug.** This is almost certainly the same defect the
tn-lane tracks as "n>=10 forward diverges ~3e-2", but the corpus proves the true
trigger is "adjacent 2q gate on a both-sides-entangled bond", reachable at
**n=4 with 5 gates on a pure Clifford circuit**. The long-range / SWAP-network
divergences are the same defect: the SWAP network makes a non-adjacent gate
adjacent on a bulk bond. Fix direction: mixed-canonicalize to the gate bond
(or otherwise account for `||theta||_F`) before the singular-value rescale.

**Repro commands.**

    # localize the first diverging gate in any case:
    scaling_diff <corpus.txt> --tn-max-n 8 --verbose
    # the minimal repro is baked into scaling_diff --selftest's tn leg
    # (H0 CX(0,2) at n=3 passes; two Bell pairs + CZ(1,2) at n=4 is the failure).

---

## S2 -- stale real-time TDVP result from before projector splitting (closed)

**What the driver caught.** `scaling_dmrg_tdvp` evolves `|0>^n` under TFIM
(J=1, h=1) and compares `<Z_0>(0.4)` to a dense Lanczos-Krylov `exp(-iHt)`
reference (verified vs `scipy.linalg.expm` to 1e-6; exact value +0.71237 for all
n>=2). Real-time two-site TDVP:

| n | TDVP `<Z0>(0.4)` | exact | norm |
|---|------------------|-------|------|
| 2 | +0.71237 | +0.71237 | 1.000 |
| 3 | +0.16857 | +0.71237 | 0.990 |
| 4 | -0.11073 | +0.71237 | 0.954 |
| 5 | +0.00085 | +0.71237 | 0.921 |
| 12 | -0.42149 | +0.71237 | 0.960 |

**Characterization.**
- Exact for n<=2 (a single bond); breaks at **n=3**, the first chain with an
  interior "bulk" site (site 1 has non-trivial left AND right environment).
- **dt-independent** (dt=0.02 and dt=0.002 give the same wrong value) -> not
  integrator order.
- **bond-independent** (bond 16..256 identical) -> not truncation.
- Affects **both one-site and two-site** variants identically -> the shared
  interior-site environment / effective-Hamiltonian machinery, not the SVD split.
- Norm decays monotonically with n.
- **DMRG, which uses the same two-site SVD split, is exact vs ED** at n=8,10,12
  (~1e-6). So this is a real-time-TDVP-specific defect, distinct from S1.

The earlier imaginary-time comparison also judged the state at insufficient
projection time. At `T=4` the n=8 finite-time energy error is about 3.2e-2; with
the unchanged `dt=0.1` flow evolved to `T=8`, it satisfies both the variational
bound and the independent ED convergence target. Both checks are now hard gates.

**Suspect.** `src/algorithms/tensor_network/tdvp.c` -- the environment tensors /
effective Hamiltonian for interior sites in the real-time sweep. The "breaks once
an interior bulk site exists" signature echoes S1's both-sides-non-trivial
trigger, suggesting Moonlab's MPS interior-site handling is fragile in more than
one subsystem.

**Repro.**

    scaling_dmrg_tdvp --verbose      # DMRG and TDVP checks pass with no quarantine

---

## Backends that agree everywhere (anchors + clean negatives)

- **dense vs numpy reference**: exact (<=1e-10) on every corpus case, every n up
  to 16. The absolute anchor; never quarantined.
- **Clifford tableau vs numpy reference**: exact (<=1e-9) on every clifford
  family, and vs analytic stabilizer structure (GHZ parity `<Z_iZ_{i+1}>=+1`,
  linear-cluster stabilizers `<X_i Z_{i-1} Z_{i+1}>=+1`) at n=50 and n=100.
  329 large-n structural checks, 0 divergences.
- **MPS-var-D** (`scaling_var_d`): TFIM + Heisenberg at n=8,16,32 -- variational
  lower bound `E >= -sum|c_k|` holds, `E >= ED E0` holds at n=8, energy is
  monotone non-increasing, eps-escape terminates with `converged=1`, coarser eps
  never accepts more gates. `max_bond_dim` fields hold 256/300/512/1024/4096 and
  a grown MPS bond of 300 reports back un-wrapped (uint32, not uint8/255).
- **DMRG vs ED** (`scaling_dmrg_tdvp`): TFIM and Heisenberg ground energies match
  dense power-iteration ED to ~1e-6 at n=8,10,12.
