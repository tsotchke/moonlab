# Archived Moonlab Documentation: Benchmark corpus

This local Moonlab document is retained as archived vendor text for the QGTL integration audit; current supported claims are measured by `scripts/moonlab_doc_claim_audit.py` and grounded against `external/moonlab/README.md`, `external/moonlab/CMakeLists.txt`, and `docs/MOONLAB_OPEN_CORE_INTEGRATION.md`.

The historical text below is preserved as an archival snapshot, not as current release documentation.

```text
# Benchmark corpus

The `tools/bench/` suite packages every manifest-emitting bench into
a single reproducible run, with a diff tool that can gate CI on
regressions.

## Reproducing

[archived fence delimiter: ```]
tools/bench/run_corpus.sh /tmp/bench-$(date +%s)
[archived fence delimiter: ```]

Each bench drops a pretty-printed JSON manifest (git SHA, build
info, host info, per-row timings with stddev/min/max) plus its raw
stdout.  `bench_chern_mosaic_hq` additionally emits a CSV and a
PPM.

Default timing replicas: 5 for `bench_tensor_matmul_eshkol`, 3 for
`bench_chern_kpm` and `bench_dmrg_workspace`.  Override globally:

[archived fence delimiter: ```]
MOONLAB_BENCH_N=10 tools/bench/run_corpus.sh /tmp/bench-10x
[archived fence delimiter: ```]

## Diffing

[archived fence delimiter: ```]
tools/bench/diff_corpus.py REFERENCE_DIR CANDIDATE_DIR [--fail-on PCT]
[archived fence delimiter: ```]

Walks both directories, flattens the `metrics` sub-objects,
matches entries by `(bench, key)`, prints ratios, and can exit
non-zero if any bench regressed by more than `PCT` percent.

CI usage (proposed):

[archived fence delimiter: ```]
tools/bench/run_corpus.sh artifacts/bench-ci
tools/bench/diff_corpus.py tools/bench/canonical/m2-ultra artifacts/bench-ci --fail-on 25
[archived fence delimiter: ```]

## Canonical reference

`tools/bench/canonical/` holds one manifest set captured on an
Apple M2 Ultra (24-core) host at commit 938b1be (2026-04-21).  Any
CI that wants a "no-surprise-regression" gate can diff against
these numbers.

Host-specific: a reviewer on different hardware should NOT use
these as absolute numbers — they are strictly a *regression
reference* for the same SKU.  Re-capture on your own hardware
with `run_corpus.sh` and store alongside for comparison.

## What's in a manifest

See `docs/benchmarks/reproducible-benchmarks.md` for the
per-manifest schema.  Every entry has:

- Git SHA of the build.
- `git_dirty` flag (loudly flags results that came from an
  uncommitted tree; never ship a release number with this set).
- Compiler + platform + host + CPU + memory.
- Run start / finish ISO timestamps, elapsed seconds.
- Metrics: the bench-specific data, including stddev / min / max
  per timing row for benches wired with `bench_stats.h`.
```
