# Archived Moonlab Documentation: Moonlab v1.0 head-to-head benchmark report

This local Moonlab document is retained as archived vendor text for the QGTL integration audit; current supported claims are measured by `scripts/moonlab_doc_claim_audit.py` and grounded against `external/moonlab/README.md`, `external/moonlab/CMakeLists.txt`, and `docs/MOONLAB_OPEN_CORE_INTEGRATION.md`.

The historical text below is preserved as an archival snapshot, not as current release documentation.

```text
# Moonlab v1.0 head-to-head benchmark report

This document defines the three head-to-head benchmarks that Moonlab v1.0
claims for its public release.  It states the exact protocol per
comparison, lists the moonlab-side invocation (a benchmark binary in this
repository), and lists the exact competitor-side invocation that the user
must run on their own hardware to fill in the missing column.

The moonlab numbers are produced by the harnesses in
`benchmarks/v1_comparison/` and committed only when produced by a real
run on the target hardware.  The competitor numbers below are listed as
`TBD` until the user reports them.

---

## Methodology

- **Platform under test**: production hardware operated by the user.
  Each table reports the actual machine + compiler used; do not copy a
  number into a table without filling those columns.
- **Compiler / build flags**: `cmake -B build -DCMAKE_BUILD_TYPE=Release`
  with the host-default `cc` (Apple clang on macOS; gcc / clang on
  Linux).  Override with `-DCMAKE_C_COMPILER=...` if a specific compiler
  is required for the report.
- **Repetitions**: each comparison is run **3 times** on each side;
  report the **median** of the three runs.  Spread (max - min) of the
  three should be reported as a sanity column.
- **Reset between runs**: a warm-up run is allowed but is not counted.
  All other runs go on a freshly booted process to avoid jitted-cache
  bias.
- **What gets logged**: every harness emits a JSON file (schema is
  listed per-comparison below).  Place each median run's JSON into
  `benchmarks/v1_comparison/results/`.
- **Honesty rule**: if the competitor side is faster or more accurate
  than Moonlab, the table records that fact.  A v1.0 release that
  claims "most powerful" without measurement is not what this document
  is for.

The shared build command:

[archived fence delimiter: ```]
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ca_mps_kagome_bench
cmake --build build --target surface_threshold_bench
cmake --build build --target sv_random_bench
[archived fence delimiter: ```]

All three binaries land at `build/<target>` and accept a positional
output-path argument plus a small set of flags documented per harness
(`-h`-equivalent: pass an unknown flag and the harness prints usage).

---

## D1: CA-MPS vs ITensor (kagome-12 frustrated AFM)

### Setup

- **Lattice**: 6 x 2 kagome torus with full periodic boundary conditions
  in both directions; 12 sites, 24 unique bonds.
- **Hamiltonian** (Pauli convention):
[archived fence delimiter:   ```]
  H = 0.25 * sum_<i,j>  (X_i X_j + Y_i Y_j + Z_i Z_j)
[archived fence delimiter:   ```]
  which is the spin-1/2 Heisenberg AFM with J_spin = 1.
- **Reference energy**: E_0 = -5.444875216, from Lauchli-Sudan-Sorensen
  PRB 83, 212401 (2011) Table I cluster "12", cross-checked by libirrep
  exact diagonalisation on the (S^z = 0, k = 0, A1) sector.
- **Protocol per chi**:
  1. Build the CA-MPS initial product state (alternating Neel pattern on
     the three kagome sublattices, then a global Hadamard layer).
  2. Imaginary-time TEBD on bond Pauli rotations: warm tau = 0.1 for 80
     sweeps, then tau = 0.03 for 120 sweeps (configurable via
     `--steps`).
  3. Renormalise after each full Trotter sweep
     (`moonlab_ca_mps_normalize`).
  4. Measure `<H>` via `moonlab_ca_mps_expect_pauli_sum`.
- **Chi sweep**: 16, 32, 64, 128.
- **Metric**: ground-state energy error |E - E_0|, wall-clock per chi,
  and peak resident-set-size for the chi = 128 run.

### Run Moonlab

[archived fence delimiter: ```]
./build/ca_mps_kagome_bench \
    benchmarks/v1_comparison/results/ca_mps_kagome.json \
    --chi 64,128,256 \
    --steps 200
[archived fence delimiter: ```]

Output JSON schema: `moonlab/v1_comparison/ca_mps_kagome`.  Each record
in `runs[]` carries `{chi, energy, error_vs_prb, wall_clock_s,
peak_rss_bytes, bond_dim_final}`.

### Convergence regime

Kagome 12 is frustrated 2D; entanglement grows fast.  Approximate
residuals vs. `E_PRB = -5.44488` at this lattice size:

| chi  | typical residual `|E - E_PRB|` |
|------|--------------------------------|
| 16   | ~1.4 J (plumbing smoke only)   |
| 64   | ~0.5 J                         |
| 128  | ~0.2 J                         |
| 256  | ~1e-2 J                        |
| 512+ | <1e-3 J (benchmark regime)     |

The v1.0 default chi list `{64, 128, 256}` is the regime where the
head-to-head vs ITensor is interpretable; neither library has
fully converged but the wall-clock ratio at the same chi is the
metric we care about.

### Run ITensor (competitor)

ITensor.jl DMRG on the same lattice and Hamiltonian.  Save the script
below as `bench/itensor_kagome.jl` in a directory of the user's choice,
then run:

[archived fence delimiter: ```]
julia --project=. bench/itensor_kagome.jl \
    --chi 64,128,256 \
    --sweeps 30 \
    --out itensor_kagome.json
[archived fence delimiter: ```]

The script must:
1. Build the same 6 x 2 kagome bond list with full periodic boundary
   conditions in both directions (24 bonds).
2. Construct `H = 0.25 * sum_<i,j> (X_i X_j + Y_i Y_j + Z_i Z_j)` with
   `OpSum` over those 24 bonds.
3. Run two-site DMRG at each chi in `{64, 128, 256}` with the
   sweep schedule and cutoff that ITensor.jl recommends for kagome (the
   exact ITensor incantation is the user's responsibility -- the
   competitor side is the one being measured, not engineered by
   Moonlab).
4. Emit `{chi, energy, error_vs_prb, wall_clock_s, peak_rss_bytes}` in
   the same `runs[]` shape so the two JSONs can be diffed directly.

### Results

Fill this table from the median of three runs each.

| chi  | Moonlab E      | err vs E_0 | Moonlab wall (s) | ITensor E | ITensor wall (s) |
| ---- | -------------- | ---------- | ---------------- | --------- | ---------------- |
| 16   | TBD            | TBD        | TBD              | TBD       | TBD              |
| 32   | TBD            | TBD        | TBD              | TBD       | TBD              |
| 64   | TBD            | TBD        | TBD              | TBD       | TBD              |
| 128  | TBD            | TBD        | TBD              | TBD       | TBD              |

Peak RSS at chi = 128: Moonlab TBD MB, ITensor TBD MB.

---

## D2: Surface code threshold sweep vs Stim

### Setup

- **Code**: toric code, d x d torus.  Data qubits live on edges
  (2 * d * d total).  Z-star checks at vertices have weight 4.  Two
  logical qubits per torus; the harness reports a failure when either
  of them fails.
- **Noise**: code-capacity X-only bit-flip noise, i.i.d. at rate p on
  every data qubit; no measurement errors.
- **Decoder**: minimum-weight perfect matching.  For <= 10 defects the
  matching is computed by exact brute-force enumeration ((n-1)!!
  pairings); for larger defect populations a greedy + 2-opt heuristic
  is used.  Geodesic correction strings are applied on the torus.
- **Logical failure**: parity of the residual X pattern on either of
  the two logical-Z chains (row-0 horizontal edges, column-0 vertical
  edges).
- **Sweep**: d in {3, 5, 7}, p in {0.001, 0.003, 0.01, 0.03, 0.1},
  n_shots = 10000.
- **Metric**: logical failure rate p_log(d, p) + standard error
  sqrt(p_log * (1 - p_log) / n) + wall-clock per (d, p) cell.
- **Literature anchor**: threshold for this code-capacity model is
  p_th ~= 0.103 (Dennis-Kitaev-Landahl-Preskill, arXiv:quant-ph/0110143).
  The Moonlab curves should cross near p_th; below it, p_log shrinks
  with d; above it, p_log grows with d.

### Run Moonlab

[archived fence delimiter: ```]
./build/surface_threshold_bench \
    benchmarks/v1_comparison/results/surface_threshold.json \
    --shots 10000
[archived fence delimiter: ```]

Output JSON schema: `moonlab/v1_comparison/surface_threshold`.  Each
record in `runs[]` carries `{d, p, shots, fails, p_logical, std_error,
wall_clock_s, peak_rss_bytes}`.

### Run Stim (competitor)

Stim + PyMatching on the same code-capacity model.  Save the script
below as `bench/stim_threshold.py` and run:

[archived fence delimiter: ```]
python bench/stim_threshold.py \
    --distances 3,5,7 \
    --p 0.001,0.003,0.01,0.03,0.1 \
    --shots 10000 \
    --out stim_threshold.json
[archived fence delimiter: ```]

The script must:
1. For each d, build a Stim circuit for the toric code with X-only
   `DEPOLARIZE1(p)`-equivalent bit-flip noise on every data qubit and
   one round of perfect Z-stabiliser measurements.  Decode with
   `pymatching.Matching` over the detector-error-model.
2. Use exactly the (d, p) grid from the Moonlab side and the same
   `--shots 10000`.
3. Emit `{d, p, shots, fails, p_logical, std_error, wall_clock_s,
   peak_rss_bytes}` in `runs[]`.

Note: Stim is the published reference for this benchmark; Moonlab will
generally lose on wall-clock per shot because Stim is a vectorised
Pauli-frame simulator while the Moonlab harness is a straightforward
single-shot loop.  Stim's strength is throughput; the apples-to-apples
question is whether the Moonlab `p_logical(d, p)` curves agree to within
~1 sigma -- the wall-clock column shows the throughput ratio.

### Results

Fill this table from the median of three runs each.

#### Logical error rates

| d | p     | Moonlab p_log | Stim p_log | abs diff |
| - | ----- | ------------- | ---------- | -------- |
| 3 | 0.001 | TBD           | TBD        | TBD      |
| 3 | 0.003 | TBD           | TBD        | TBD      |
| 3 | 0.01  | TBD           | TBD        | TBD      |
| 3 | 0.03  | TBD           | TBD        | TBD      |
| 3 | 0.1   | TBD           | TBD        | TBD      |
| 5 | 0.001 | TBD           | TBD        | TBD      |
| 5 | 0.003 | TBD           | TBD        | TBD      |
| 5 | 0.01  | TBD           | TBD        | TBD      |
| 5 | 0.03  | TBD           | TBD        | TBD      |
| 5 | 0.1   | TBD           | TBD        | TBD      |
| 7 | 0.001 | TBD           | TBD        | TBD      |
| 7 | 0.003 | TBD           | TBD        | TBD      |
| 7 | 0.01  | TBD           | TBD        | TBD      |
| 7 | 0.03  | TBD           | TBD        | TBD      |
| 7 | 0.1   | TBD           | TBD        | TBD      |

#### Throughput

| d | Moonlab wall total (s) | Stim wall total (s) |
| - | ---------------------- | ------------------- |
| 3 | TBD                    | TBD                 |
| 5 | TBD                    | TBD                 |
| 7 | TBD                    | TBD                 |

---

## D3: State-vector vs Qiskit-Aer

### Setup

- **Circuit family**: brick-wall random.  For each of `depth = 50`
  layers:
  1. Draw a random permutation of {0, ..., N-1} via Fisher-Yates.
  2. Pair adjacent entries `(perm[0], perm[1]), (perm[2], perm[3]),
     ...` (the last qubit is idle if N is odd, but we sweep even N).
  3. Apply a Haar-random `U(4)` to each pair, generated via the
     Mezzadri 2007 construction (QR of an i.i.d. complex-Gaussian 4 x 4
     matrix with phase correction).
- **N sweep**: 20, 22, 24, 26, 28, 30.
- **Shots**: 1024 Born-rule computational-basis samples per circuit
  (no collapse; uses `measurement_sample`).
- **Metric**: wall-clock (circuit + sampling, both reported separately),
  peak resident-set-size.
- **Memory budget**: 16 B per amplitude, so N = 30 requires ~16 GiB just
  for the state vector.  The harness will refuse N > MOONLAB_MAX_QUBITS
  (32 on a 64-bit native host, 30 on wasm32); the user is responsible
  for ensuring the host has enough RAM.

### Run Moonlab

[archived fence delimiter: ```]
./build/sv_random_bench \
    benchmarks/v1_comparison/results/sv_random.json \
    --qubits 20,22,24,26,28,30 \
    --depth 50 \
    --shots 1024
[archived fence delimiter: ```]

Output JSON schema: `moonlab/v1_comparison/sv_random`.  Each record in
`runs[]` carries `{N, depth, shots, wall_clock_s, circuit_s, sampling_s,
peak_rss_bytes, expected_state_dim_bytes}`.

### Run Qiskit-Aer (competitor)

Qiskit-Aer's statevector backend with the same circuit family.  Save
the script below as `bench/qiskit_aer_random.py` and run:

[archived fence delimiter: ```]
python bench/qiskit_aer_random.py \
    --qubits 20,22,24,26,28,30 \
    --depth 50 \
    --shots 1024 \
    --out qiskit_aer_random.json
[archived fence delimiter: ```]

The script must:
1. Match the Moonlab RNG seed and the Mezzadri 2007 Haar construction
   to keep the circuits identical (or, if a deterministic match is too
   fiddly, draw fresh Haar U(4)s using `scipy.stats.unitary_group` --
   the result is still the same distribution and the median walltime
   over three runs is the comparison metric).
2. Run with `AerSimulator(method="statevector")` and `transpile(opt=0)`
   to avoid Aer doing free work that Moonlab is not doing.
3. Emit `{N, depth, shots, wall_clock_s, circuit_s, sampling_s,
   peak_rss_bytes}` in `runs[]`.

### Results

Fill this table from the median of three runs each.

| N  | Moonlab wall (s) | Qiskit-Aer wall (s) | Moonlab peak RSS (GiB) | Aer peak RSS (GiB) |
| -- | ---------------- | ------------------- | ---------------------- | ------------------ |
| 20 | TBD              | TBD                 | TBD                    | TBD                |
| 22 | TBD              | TBD                 | TBD                    | TBD                |
| 24 | TBD              | TBD                 | TBD                    | TBD                |
| 26 | TBD              | TBD                 | TBD                    | TBD                |
| 28 | TBD              | TBD                 | TBD                    | TBD                |
| 30 | TBD              | TBD                 | TBD                    | TBD                |

---

## How to reproduce

From a clean checkout:

[archived fence delimiter: ```]
# 1. Configure + build all three v1 comparison harnesses.
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ca_mps_kagome_bench
cmake --build build --target surface_threshold_bench
cmake --build build --target sv_random_bench

# 2. Run the Moonlab side of each comparison three times; keep the
#    median by wall-clock.
mkdir -p benchmarks/v1_comparison/results
for i in 1 2 3; do
    ./build/ca_mps_kagome_bench       benchmarks/v1_comparison/results/ca_mps_kagome.run$i.json       --chi 16,32,64,128 --steps 200
    ./build/surface_threshold_bench   benchmarks/v1_comparison/results/surface_threshold.run$i.json   --shots 10000
    ./build/sv_random_bench           benchmarks/v1_comparison/results/sv_random.run$i.json           --qubits 20,22,24,26,28,30 --depth 50 --shots 1024
done

# 3. Run the competitor side per the per-comparison invocations
#    above.  Place results in benchmarks/v1_comparison/results/ next
#    to the Moonlab JSONs.

# 4. Pick the median of three runs per cell, populate the tables in
#    this document, and commit.
[archived fence delimiter: ```]

When the tables are populated, drop the `TBD` entries and add a
"results provenance" footer listing the machine, OS, compiler version,
and date of the run.  Anything that has not been measured stays `TBD` --
a v1.0 claim that this document is the head-to-head report only holds
once every cell has a number behind it.
```
