/**
 * @file    scheduler.h
 * @brief   Distributed-execution scheduler MVP (v0.7.0).
 *
 * The first piece of moonlab's cloud-platform foundation.  A
 * `moonlab_job_t` carries a circuit description + shot count + worker
 * fan-out; `moonlab_scheduler_run` splits the shots across N workers,
 * dispatches each piece, and merges the outcomes.
 *
 * v0.7.0 transport: OpenMP threads in-process.  v0.7.1+ swaps in MPI
 * (existing `src/distributed/mpi_bridge.{c,h}` scaffolding) and v0.7.2+
 * adds gRPC / HTTP/2 control-plane.  The contract -- gate list +
 * shot count + worker-side execution + result merge -- stays stable.
 *
 * JSON serialisation is supported one-way (`moonlab_job_to_json`) so
 * a remote-worker process can be handed a job description over the
 * wire.  Parsing is v0.7.1+ scope.
 *
 * @since v0.7.0
 */

#ifndef MOONLAB_SCHEDULER_H
#define MOONLAB_SCHEDULER_H

#include <stddef.h>
#include <stdint.h>

#include "../applications/moonlab_api.h"
#include "../applications/moonlab_qgtl_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Status codes for the distributed scheduler. */
#define MOONLAB_SCHED_OK              ( 0)
#define MOONLAB_SCHED_BAD_ARG         (-501)
#define MOONLAB_SCHED_OOM             (-502)
#define MOONLAB_SCHED_INTERNAL        (-503)
#define MOONLAB_SCHED_BUFFER_TOO_SMALL (-504)

/** @brief Opaque job-spec handle. */
typedef struct moonlab_job moonlab_job_t;

/** @brief Output buffer for `moonlab_scheduler_run`.  `outcomes`
 *  has length `total_shots`; entries are bitstrings (uint64 packed).
 *  `worker_seconds[i]` is the wall-clock time worker `i` spent on
 *  its slice. */
typedef struct {
    int       num_qubits;
    int       total_shots;
    uint64_t *outcomes;
    int       num_workers_used;
    double   *worker_seconds;
} moonlab_job_results_t;

/**
 * @brief Allocate a job with `num_qubits` qubits, fan-out 1, shot
 *        count 0.  Call `moonlab_job_add_gate` and
 *        `moonlab_job_set_num_shots` / `_set_num_workers` to flesh
 *        it out before scheduling.
 */
MOONLAB_API moonlab_job_t *
moonlab_job_create(int num_qubits);

/** @brief Release a job handle. */
MOONLAB_API void moonlab_job_free(moonlab_job_t *job);

/** @brief Append a gate.  Same contract as
 *         @ref moonlab_qgtl_add_gate. */
MOONLAB_API int
moonlab_job_add_gate(moonlab_job_t *job,
                     moonlab_qgtl_gate_t type,
                     int target, int control,
                     const double *params);

/** @brief Set the total shot count.  The scheduler splits this
 *         across `num_workers` slices internally. */
MOONLAB_API int
moonlab_job_set_num_shots(moonlab_job_t *job, int num_shots);

/** @brief Set the worker fan-out.  Default 1.  Capped at
 *         `OMP_NUM_THREADS` when OpenMP is enabled; otherwise the
 *         scheduler runs everything serially regardless. */
MOONLAB_API int
moonlab_job_set_num_workers(moonlab_job_t *job, int num_workers);

/** @brief Optional reproducibility seed for the per-worker PRNGs.
 *         Worker `i` derives its seed as
 *         `splitmix64(base_seed XOR i_in_hi32)`. */
MOONLAB_API int
moonlab_job_set_rng_seed(moonlab_job_t *job, uint64_t seed);

/* Introspection. */
MOONLAB_API int moonlab_job_num_qubits(const moonlab_job_t *job);
MOONLAB_API int moonlab_job_num_gates(const moonlab_job_t *job);
MOONLAB_API int moonlab_job_num_shots(const moonlab_job_t *job);
MOONLAB_API int moonlab_job_num_workers(const moonlab_job_t *job);

/**
 * @brief Run the job across the worker fan-out.
 *
 * Splits `total_shots` into `num_workers` contiguous slices; each
 * worker clones the gate list into a `moonlab_qgtl_circuit_t`,
 * executes its slice via `moonlab_qgtl_execute`, and writes its
 * outcomes into the merged buffer.  When built with OpenMP, the
 * worker loop is `#pragma omp parallel for`.
 *
 * `out` is caller-allocated and zero-initialised; the call
 * allocates `outcomes` (length `total_shots`) and
 * `worker_seconds` (length `num_workers_used`).  Release with
 * `moonlab_job_results_free`.
 *
 * @return MOONLAB_SCHED_OK on success or a negative status code.
 */
MOONLAB_API int
moonlab_scheduler_run(moonlab_job_t           *job,
                      moonlab_job_results_t   *out);

/** @brief Release buffers attached to a results record. */
MOONLAB_API void
moonlab_job_results_free(moonlab_job_results_t *r);

/**
 * @brief Serialise a job to JSON.  Format (one example):
 *
 *     {
 *       "schema": "moonlab/job/v0.7.0",
 *       "num_qubits": 2,
 *       "num_shots": 1024,
 *       "num_workers": 4,
 *       "rng_seed": "0xdeadbeef",
 *       "gates": [
 *         { "type": 4, "target": 0 },
 *         { "type": 10, "target": 1, "control": 0 }
 *       ]
 *     }
 *
 * @param[in]  job      Job to serialise.
 * @param[out] buf      Caller-allocated output buffer.
 * @param[in]  bufsize  Capacity of `buf` in bytes (including NUL).
 * @return  Number of bytes that *would* be written if `buf` were
 *          unlimited (excluding the terminating NUL).  When this
 *          exceeds `bufsize`, the output is truncated; pass a
 *          NULL `buf` and `bufsize = 0` to size-probe.
 *          Negative on bad argument.
 */
MOONLAB_API int
moonlab_job_to_json(const moonlab_job_t *job,
                    char *buf, size_t bufsize);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_SCHEDULER_H */
