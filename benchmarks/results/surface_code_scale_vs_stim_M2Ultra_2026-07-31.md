# Task #59: does the stabilizer stack reach Stim's headline scale?

Machine: Apple M2 Ultra, 24 cores, 192 GB, macOS 24.1.0, AppleClang, Release build.
Stim 1.15.0, installed on the same machine. Both engines run the same circuit
family; at d = 5, 9, 15 they run the byte-identical circuit.

Artifacts: `surface_code_scale_vs_stim_M2Ultra_2026-07-31.json` (this directory),
harness `benchmarks/surface_code_scale.c` + `benchmarks/surface_code_scale_stim.py`.

## Verdict

No. The stack runs the distance-101 circuit correctly and nothing in the data
structures breaks at that size, but the setup phase is three orders of magnitude
off Stim and the sampler is 7-8x off per core.

At d = 101 the circuit is 20401 qubits, 7.2M gates, 1.04M measurements and
1030200 detectors -- Stim's published headline workload. Moonlab needs
**166.5 s** before it can emit the first shot. Stim's detector sampler needs
**0.040 s**. After setup Moonlab samples 64 shots/s on one core against Stim's
437 shots/s, and needs all 24 cores to reach 771 shots/s.

The two gaps are not the same kind of problem, and the headline ratio is
misleading in Moonlab's favour as well as against it. Both are broken out below.

## Numbers

Distance d, d rounds, circuit-level depolarising noise p = 0.001 (DEPOLARIZE1 on
data before each round and after each 1q Clifford, DEPOLARIZE2 after each CNOT,
X_ERROR after reset and before measurement -- stim's four generator knobs all set
to p). All times are wall clock on the machine above.

### Circuit

| d | qubits | gates | measurements | detectors |
|---:|---:|---:|---:|---:|
| 25 | 1249 | 108674 | 16225 | 15600 |
| 51 | 5201 | 925802 | 135201 | 132600 |
| 75 | 11249 | 2947274 | 427425 | 421800 |
| 101 | 20401 | 7201602 | 1040401 | 1030200 |

### Setup / analysis phase

`moonlab` is `pf_compute_reference()`, the Aaronson-Gottesman tableau pass that
`pauli_frame_batch_sample_detectors()` runs before any shot is produced.
`stim reference_sample()` is the genuinely equivalent computation.
`stim detector-sampler setup` is `compile_detector_sampler()` plus the first
`sample()` call -- what Stim actually needs before it starts emitting detectors.

| d | moonlab | stim `reference_sample()` | stim detector-sampler setup | vs equivalent | vs what stim needs |
|---:|---:|---:|---:|---:|---:|
| 25 | 0.091 s | 0.0089 s | 0.00070 s | 10x | 128x |
| 51 | 3.234 s | 0.2656 s | 0.00564 s | 12x | 600x |
| 75 | 26.116 s | 2.0224 s | 0.01735 s | 13x | 1578x |
| 101 | 166.538 s | 8.9040 s | 0.03988 s | 19x | 3633x |

Peak RSS for the moonlab pass: 11 / 77 / 243 / 752 MB. Memory is not the problem.

These are the numbers at HEAD. The same measurement before the fix described
below was 0.152 / 5.622 / 45.359 / 270.681 s.

### Sampling phase

Stim's sampler is single-threaded, so the one-thread column is the core-fair
comparison. Byte-per-detector output on both sides.

| d | moonlab, 1 thread | moonlab, 24 threads | stim, 1 thread | stim/moonlab per core |
|---:|---:|---:|---:|---:|
| 25 | 4377 shots/s | 46707 shots/s | 34203 shots/s | 7.8x |
| 51 | 490 shots/s | 6373 shots/s | 4048 shots/s | 8.3x |
| 75 | 158 shots/s | 2192 shots/s | 1201 shots/s | 7.6x |
| 101 | 64 shots/s | 771 shots/s | 437 shots/s | 6.8x |

Stim's kHz claim holds on this machine up to d = 75 (1201 shots/s) and falls to
437 shots/s at d = 101. Moonlab reaches kHz at d = 101 only by spending 24 cores.

## Reading the two gaps

**Sampling is a constant factor.** 7.8x, 8.3x, 7.6x, 6.8x across a 16x range in
qubit count. The Pauli-frame sampler's shot-bit-packing scales correctly: 64
shots to a 64-bit word, W = ceil(shots/64) words per row, gate cost O(W)
independent of shot count, NEON-widened, and the detector reduction happens
per shot-block so the full measurement record is never materialised at full
width. Nothing here is a wall. It is a flat 7-8x of per-core efficiency.

**Setup is a wall.** 128x, 600x, 1578x, 3633x -- the ratio grows with d, which
is the signature of a scaling problem rather than a constant factor.

But the 3633x number overstates the algorithmic deficit, and saying so is part
of reporting this honestly. Stim's detector sampler does not compute a reference
sample at all. A detector is by definition a set of measurements whose parity is
deterministic, so its reference parity is fixed and contributes nothing; Stim
skips the work. Moonlab computes the full reference trajectory and then XORs it
back out in `det_ref`. Measured directly: Stim's `reference_sample()` costs
8.90 s at d = 101 while its detector-sampler setup costs 0.040 s, a factor of
223 between the same library doing the work and skipping it.

So the honest decomposition at d = 101 is:

- **19x**: Moonlab's reference pass against Stim's reference pass. A large
  constant factor with mild superlinear drift (10x -> 19x over the range), caused
  by the tableau layout, diagnosed below.
- **223x on top of that**: Moonlab does the reference pass at all on the detector
  path, where Stim has established it is unnecessary.

## Bottleneck diagnosis

Instrumented run at d = 51 (counters compiled into a copy of `clifford.c`;
`src/` unmodified):

```
measurements: deterministic=270402  random=2600
row_get calls=491852   column-reads=5116244504
contributing rows per deterministic measurement = 1.67
wall = 5.095 s  ->  1.0 ns per column read
```

The algorithm is not the problem: only 1.67 tableau rows contribute to a typical
deterministic measurement. The cost is entirely in *fetching* each of those rows.

`clifford.c` stores the tableau bit-packed and **column-major**: for each qubit
column j, the X-bits and Z-bits across all 2n rows are a contiguous
`w = ceil(2n/64)` word vector. That is the right layout for gates -- a gate on
qubit q touches two columns and all 2n rows, so it is a handful of word-parallel
loops over w words, matching Stim's per-gate cost.

It is the wrong layout for measurement. `row_get()` extracts one row:

```c
for (size_t j = 0; j < n; j++) {
    if (getbit(xcol(t, j), r)) xb[j >> 6] |= (uint64_t)1 << (j & 63);
    if (getbit(zcol(t, j), r)) zb[j >> 6] |= (uint64_t)1 << (j & 63);
}
```

That is **2n memory operations to fetch 2n bits** which would be 2n/64 contiguous
words in a row-major store -- a factor of 64 in operation count before locality is
considered. Consecutive columns are `w * 8` bytes apart (5104 bytes at d = 101),
so every one of those reads lands on a different cache line.

The scaling follows from this directly. Total work is `~1.67 * M * 2n` scattered
reads for M measurements at HEAD (it was twice that before the reset fix below).
At d = 101 that is `1.67 * 1.04e6 * 40802 = 7.1e10` reads; measured 166.5 s gives
2.3 ns each. At d = 51 the tableau is 13.6 MB and fits in the system-level cache, so
reads cost 1.0 ns; at d = 101 it is 208 MB and they go to DRAM. That cache
cliff is why the measured exponent rises from d^5.0 over d = 25 -> 51 to d^5.6
over d = 51 -> 75.

This is the same tension Stim resolves explicitly: it keeps the tableau in the
gate-friendly layout and performs a blocked bit-transpose
(`TableauTransposedRaii`) to get row-major access for a batch of measurements,
then transposes back, amortizing the transpose across the batch.

## Data-structure scaling assessment

- **Tableau**: bit-packed, column-major, `n^2/2` bytes. 208 MB at n = 20401.
  No `n^2` memory wall in practice. Hard cap `n > 100000` rejected in
  `clifford_tableau_create`; d = 223 would be the first distance to hit it.
  Indices are `size_t` throughout; no int-width limit is reached at this scale.
- **Frame sampler**: scales correctly in both qubits and shots, as measured.
- **Measurement record**: stored one byte per bit. `mbuf = malloc(nmeas * block_shots)`
  is ~45 MB per thread at d = 101 / 1024 shots / 24 threads, and the detector
  output buffer is `n_det * shots` = 1.05 GB. Bit-packing the record and the
  output would cut both 8x. Not a correctness or scaling wall, but it is the
  reason the sampler is memory-bound at large d, and it is a plausible share of
  the flat 7-8x per-core sampling gap.

## Correctness gates

All three pass at d = 5, 9, 15.

| gate | result |
|---|---|
| noiseless run leaves every detector quiet | pass at d = 5, 9, 15, 25, 51 |
| measurement/detector counts match stim's `rotated_memory_z` | 145/385/801 and 120/336/720, exact |
| stim builds the detector error model from our circuit | yes (detectors and observable all deterministic) |
| per-detector fire rates vs stim, same circuit, 20000 shots | worst deviation 2.79 / 3.65 / 3.58 sigma, 0 of 120 / 720 / 3360 detectors beyond 4 sigma |

Bulk detector fractions at p = 0.001 sit at 0.0176-0.0181 (moonlab) against
0.0183-0.0191 (stim's own generated circuit), the residual being the small
difference between the two constructions' noise placement; on the shared circuit
the engines agree within counting statistics.

## Fix that landed

`pf_compute_reference()` ran a full Aaronson-Gottesman measurement for every
RESET as well as every MEASURE. QEC circuits measure each ancilla and reset it
immediately, so half of all deterministic measurements were recomputing what the
previous op had just established -- the instrumented d = 51 run shows 270402
deterministic measurements against 135201 measurement ops.

A per-qubit Z-eigenstate cache removes them. It is sound because a Z-basis
measurement of a different qubit cannot evict Z_q from the stabilizer group, and
output is bit-identical because the deterministic branch consumes no RNG.

Before / after on the analysis phase:

| d | before | after | speedup |
|---:|---:|---:|---:|
| 25 | 0.152 s | 0.091 s | 1.67x |
| 51 | 5.622 s | 3.234 s | 1.74x |
| 75 | 45.359 s | 26.116 s | 1.74x |
| 101 | 270.681 s | 166.538 s | 1.63x |

`unit_clifford`, `unit_clifford_rowsum`, `unit_clifford_pauli_api` and
`unit_pauli_frame` all pass, and all three correctness gates above still pass.

This does not change the verdict. It is a constant factor on a gap that is
structural.

## Scoped follow-up (not attempted here)

Two changes, in order of value. Neither is a tuning exercise; both change how the
reference phase works, which is why they are called out rather than attempted.

**1. Do not compute a reference sample on the detector path.** Detectors are
deterministic by construction, so `det_ref` is determined and the reference
trajectory is not needed to produce detector output. The one thing
`pf_compute_reference()` supplies that the frame pass genuinely consumes is
`m_kind` -- which measurements are random, used to inject fresh Z-frame entropy
at those points -- and that is derivable without a full tableau simulation.
Projected impact: the entire setup phase disappears from the detector path,
166.5 s -> approximately zero at d = 101. This is the whole gap for any decoder
pipeline, which is the dominant consumer. Measured precedent: Stim's own
detector setup is 223x cheaper than its `reference_sample()`.

**2. Batched transpose for the tableau.** Keep the column-major store for gates
and add a blocked bit-transpose amortized across each round's ancilla
measurements, so `row_get()` becomes 2n/64 sequential words instead of 2n
scattered reads. Projected impact: 50-100x on the reference pass from the
operation-count factor of 64 plus recovered locality, which would put d = 101 in
the 2-3 s range against Stim's 8.90 s. Required for measurement sampling, where
a reference trajectory genuinely is needed and item 1 does not apply.

Item 1 is the larger win and the smaller change. Item 2 is what closes the
remaining 19x.

## Reproduce

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DQSIM_BUILD_BENCHMARKS=ON
cmake --build build -j8 --target surface_code_scale
pip install stim
python3 benchmarks/surface_code_scale_stim.py \
    --d 25 51 75 101 --xcheck-d 5 9 15 --p 0.001 \
    --shots 1024 --st-shots 512 \
    --bin ./build/surface_code_scale \
    --out benchmarks/results/surface_code_scale_vs_stim_M2Ultra_2026-07-31.json
```

Roughly 8 minutes, dominated by the d = 101 analysis pass. Set
`MOONLAB_SC_NO_ZCACHE=1` to reproduce the pre-fix analysis timings. Single
distance, human-readable:

```sh
./build/surface_code_scale --d 101 --p 0.001 --shots 1024
./build/surface_code_scale --d 25 --verify --shots 256      # noiseless gate
./build/surface_code_scale --d 9 --dump-stim /tmp/c.stim    # circuit for stim
```
