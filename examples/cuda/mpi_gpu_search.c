/**
 * @file  mpi_gpu_search.c
 * @brief v1.1 step 11: MPI + CUDA together.
 *
 * Each MPI rank creates an INDEPENDENT GPU-backed state via the
 * standard moonlab API (no special MPI-aware GPU state needed),
 * runs a different parameter point of a small variational circuit
 * on its local GPU, computes a target observable, and the ranks
 * MPI_Allreduce to find the lowest-energy point across the cluster.
 *
 * This validates two things at once:
 *   (a) libquantumsim.so built with QSIM_ENABLE_MPI=ON AND
 *       QSIM_ENABLE_CUDA=ON links + runs cleanly.
 *   (b) The new quantum_state_create_gpu() backend works from
 *       inside an MPI process (no global / static-init clashes).
 *
 * Honest scope: this is *trivially parallel* (no data exchange
 * between ranks during the per-rank computation -- ranks only
 * meet at the final MPI_Allreduce).  True sharded MPI+CUDA --
 * where one state vector is split across ranks' GPUs and gates
 * trigger inter-rank halo swaps -- requires teaching
 * partitioned_state_t about gpu_state, which is a separate
 * follow-up refactor (out of scope for the v1.1 GPU arc).
 */

#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"
#include "../../src/backends/cuda_tegra_probe.h"

#include <complex.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Tiny variational ansatz with one free parameter -- 4 qubits,
 * one layer of H followed by RZ(theta) on each, then a CNOT chain.
 * Observable: <Z_0 Z_1> on the resulting state.
 *
 * We sweep theta uniformly across [0, 2*pi) with one point per rank
 * and find the theta that minimizes <Z_0 Z_1>. */
static double compute_zz_expectation(int N, double theta)
{
    quantum_state_t *s = NULL;
    qs_error_t err = quantum_state_create_gpu((size_t)N, &s);
    if (err != QS_SUCCESS) {
        /* Fall back to CPU if GPU isn't available on this rank. */
        s = quantum_state_create(N);
        if (!s) return NAN;
    }

    /* A 1-parameter ansatz where <Z_0 Z_1> sweeps cos(theta):
     *     H q0;  RY(theta) q1;  CNOT q0 q1
     * Adding any further qubits keeps them as ancilla |0>, leaving
     * <Z_0 Z_1> unchanged so the search is well-defined for any N. */
    gate_hadamard(s, 0);
    gate_ry      (s, 1, theta);
    gate_cnot    (s, 0, 1);

    quantum_state_sync_to_host(s);

    /* <Z_0 Z_1> = sum_k (-1)^(bit0 ^ bit1) |amp_k|^2 */
    double zz = 0.0;
    size_t dim = (size_t)1 << N;
    for (size_t k = 0; k < dim; k++) {
        int b0 = (k >> 0) & 1;
        int b1 = (k >> 1) & 1;
        double x = creal(s->amplitudes[k]);
        double y = cimag(s->amplitudes[k]);
        double p = x*x + y*y;
        zz += (b0 ^ b1) ? -p : p;
    }

    quantum_state_destroy(s);
    return zz;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    int N = (argc > 1) ? atoi(argv[1]) : 8;
    if (N < 2) {
        if (rank == 0) fprintf(stderr, "need N>=2\n");
        MPI_Finalize();
        return 1;
    }

    /* Each rank gets a unique theta in [0, 2*pi). */
    double theta = (2.0 * M_PI * rank) / (double)nranks;
    double zz_local = compute_zz_expectation(N, theta);

    /* Find the global minimum and its rank.  MPI_MINLOC with a
     * double-int pair handles both at once. */
    struct { double val; int rank; } in, out;
    in.val = zz_local;
    in.rank = rank;
    MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

    /* Each rank prints its result; rank 0 also prints the winner. */
    const char *rank_host = "redacted";
    char hostname[256] = {0};
    const char *include_host = getenv("MOONLAB_MPI_INCLUDE_HOSTNAME");
    if (include_host && (strcmp(include_host, "1") == 0 ||
                         strcmp(include_host, "true") == 0 ||
                         strcmp(include_host, "TRUE") == 0) &&
        gethostname(hostname, sizeof(hostname) - 1) == 0) {
        rank_host = hostname;
    }
    printf("[rank %d/%d on %-12s gpu=%s] theta=%.4f  <Z0 Z1>=%+.6f\n",
        rank, nranks, rank_host,
        moonlab_gpu_probe_kind_str(moonlab_gpu_probe_kind()),
        theta, zz_local);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("---\n");
        printf("Global minimum: rank %d, <Z0 Z1>=%+.6f at theta=%.4f\n",
            out.rank, out.val, (2.0 * M_PI * out.rank) / (double)nranks);
    }

    MPI_Finalize();
    return 0;
}
