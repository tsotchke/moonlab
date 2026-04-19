/**
 * @file manifest.h
 * @brief Reproducibility manifest for benchmarks and demos.
 *
 * Every published Moonlab number -- throughput, accuracy, wall-clock --
 * needs to pin down the source tree, compiler, platform, runtime
 * environment, and any stochastic seeds the run used.  This module
 * collects all of it into a single @c moonlab_manifest_t that can be
 * emitted as JSON alongside any result.
 *
 * Compile-time facts (git SHA, compiler, enabled feature flags, version)
 * come from the CMake-generated @c moonlab_build_info.h.  Run-time facts
 * (hostname, CPU model, total memory, timestamps, user-provided seeds
 * and labels) are captured when @c moonlab_manifest_capture runs.
 *
 * Typical usage inside a benchmark:
 *
 * @code
 *   moonlab_manifest_t m;
 *   moonlab_manifest_capture(&m, "bench_eshkol_gemm", 42);
 *   // ... run the bench, fill m.metrics_json with result JSON ...
 *   moonlab_manifest_stamp_finish(&m);
 *   moonlab_manifest_write_json(&m, stdout);
 *   moonlab_manifest_release(&m);
 * @endcode
 *
 * The runtime capture is deliberately cheap (no process spawns on the
 * critical path) so benches can write a manifest on every run without
 * inflating wall-clock.  @c uname(2) plus @c sysctlbyname / @c /proc
 * probes supply the host info.
 *
 * JSON is emitted flat, with no vendored dependency.  Keys map 1:1 to
 * the struct fields; strings are escaped per RFC 8259; the user-
 * supplied @c metrics_json field is embedded verbatim so callers can
 * splice their own well-formed JSON sub-document without double
 * quoting.
 *
 * @since v0.2.0
 */

#ifndef MOONLAB_MANIFEST_H
#define MOONLAB_MANIFEST_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    /* Identifier the caller supplies, e.g. "bench_tensor_matmul_eshkol". */
    char* run_label;
    /* Optional user-supplied random seed (0 if none). */
    uint64_t seed;

    /* Build-time facts.  Duplicated here so the JSON dump is
     * self-contained without the caller having to also include
     * moonlab_build_info.h. */
    const char* version;         /* not owned */
    const char* git_sha;
    const char* git_sha_short;
    int         git_dirty;
    const char* git_branch;
    const char* build_timestamp;
    const char* build_type;
    const char* compiler_id;
    const char* compiler_version;
    const char* platform_name;
    const char* arch_name;
    const char* enabled_features;

    /* Run-time facts. */
    char*  hostname;             /* owned */
    char*  os_release;           /* owned; uname "release" */
    char*  cpu_brand;            /* owned; sysctlbyname or /proc/cpuinfo */
    int    cpu_count;
    uint64_t mem_total_bytes;
    char*  run_start_iso;        /* owned */
    char*  run_finish_iso;       /* owned; empty until stamp_finish */
    double run_elapsed_seconds;  /* 0 until stamp_finish */

    /* Caller-provided JSON fragment embedded verbatim as
     * "metrics" : { ... }.  Caller owns the string; we reference it. */
    const char* metrics_json;
} moonlab_manifest_t;

/**
 * @brief Populate build-time fields from moonlab_build_info.h and
 *        run-time fields from uname + sysctl probes.
 *
 * On return, @c run_start_iso is set to the current UTC time; other
 * timing fields remain unset until @c moonlab_manifest_stamp_finish.
 * @c metrics_json is NULL until the caller attaches a fragment.
 *
 * @param m      Manifest, caller-allocated (stack or heap).
 * @param label  Short identifier, copied into run_label.  May be NULL;
 *               stored as "(unnamed)" if so.
 * @param seed   Zero if the run has no stochastic seed.
 * @return 0 on success, nonzero on allocation failure.
 */
int moonlab_manifest_capture(moonlab_manifest_t* m,
                             const char* label,
                             uint64_t seed);

/**
 * @brief Stamp the finish timestamp and elapsed seconds.  Idempotent.
 */
void moonlab_manifest_stamp_finish(moonlab_manifest_t* m);

/**
 * @brief Emit the manifest as a single-line JSON object to @p out.
 *        No trailing newline.
 */
void moonlab_manifest_write_json(const moonlab_manifest_t* m, FILE* out);

/**
 * @brief Same as @c write_json but pretty-printed with two-space
 *        indentation and a trailing newline.  Use for human-readable
 *        sidecars; use the compact form inside structured logs.
 */
void moonlab_manifest_write_json_pretty(const moonlab_manifest_t* m,
                                        FILE* out);

/**
 * @brief Release owned heap allocations (run_label, hostname,
 *        os_release, cpu_brand, run_start_iso, run_finish_iso).
 *        Safe to call on a zero-initialised manifest.
 */
void moonlab_manifest_release(moonlab_manifest_t* m);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_MANIFEST_H */
