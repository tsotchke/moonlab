/**
 * @file seeded_shots_replay_probe.c
 * @brief Host-independent seeded SHOTS control-plane replay probe.
 *
 * The probe deliberately prints no diagnostics on stdout: its one output line
 * is the artifact compared by the cross-host mesh gate.
 */

#include "../src/applications/moonlab_qgtl_backend.h"
#include "../src/control/control_plane.h"

#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { PROBE_SHOTS = 64 };
static const uint64_t PROBE_SEED = UINT64_C(0x0123456789abcdef);

typedef struct {
  moonlab_control_server_t *server;
  int rc;
} server_thread_args_t;

static void *run_server(void *opaque) {
  server_thread_args_t *args = (server_thread_args_t *)opaque;
  /* The probe submits exactly one request, so let the server exit after
   * that accepted connection instead of relying on a cross-thread wakeup. */
  args->rc = moonlab_control_server_run(args->server, 1);
  return NULL;
}

static char *canonical_bell_text(void) {
  moonlab_qgtl_circuit_t *circuit = moonlab_qgtl_circuit_create(2);
  if (!circuit)
    return NULL;

  if (moonlab_qgtl_add_gate(circuit, MOONLAB_QGTL_GATE_H, 0, 0, NULL) != 0 ||
      moonlab_qgtl_add_gate(circuit, MOONLAB_QGTL_GATE_CNOT, 1, 0, NULL) != 0) {
    moonlab_qgtl_circuit_free(circuit);
    return NULL;
  }

  size_t needed = 0;
  if (moonlab_qgtl_circuit_serialize(circuit, NULL, 0, &needed) != 0 ||
      needed == 0) {
    moonlab_qgtl_circuit_free(circuit);
    return NULL;
  }

  char *text = (char *)malloc(needed + 1);
  if (!text ||
      moonlab_qgtl_circuit_serialize(circuit, text, needed + 1, NULL) != 0) {
    free(text);
    moonlab_qgtl_circuit_free(circuit);
    return NULL;
  }
  text[needed] = '\0';
  moonlab_qgtl_circuit_free(circuit);
  return text;
}

int main(void) {
  moonlab_control_server_t *server = NULL;
  pthread_t server_tid;
  int thread_started = 0;
  int rc = MOONLAB_CONTROL_OK;
  uint16_t port = 0;
  char *circuit_text = NULL;
  uint64_t *outcomes = NULL;
  size_t outcome_count = 0;
  uint64_t effective_seed = 0;
  int exit_code = 1;

  rc = moonlab_control_server_open("127.0.0.1", 0, &server, &port);
  if (rc != MOONLAB_CONTROL_OK || !server || port == 0) {
    fprintf(stderr, "seeded_shots_replay_probe: server open failed (rc=%d)\n",
            rc);
    goto cleanup;
  }

  server_thread_args_t server_args = {server, MOONLAB_CONTROL_IO_ERROR};
  if (pthread_create(&server_tid, NULL, run_server, &server_args) != 0) {
    fprintf(stderr,
            "seeded_shots_replay_probe: server thread creation failed\n");
    goto cleanup;
  }
  thread_started = 1;

  circuit_text = canonical_bell_text();
  if (!circuit_text) {
    fprintf(stderr, "seeded_shots_replay_probe: Bell serialization failed\n");
    goto cleanup;
  }

  rc = moonlab_control_submit_circuit_shots_seeded(
      "127.0.0.1", port, circuit_text, 0, PROBE_SHOTS, PROBE_SEED, &outcomes,
      &outcome_count, &effective_seed);
  if (rc != MOONLAB_CONTROL_OK) {
    fprintf(stderr,
            "seeded_shots_replay_probe: seeded submission failed (rc=%d)\n",
            rc);
    goto cleanup;
  }
  if (effective_seed != PROBE_SEED) {
    fprintf(stderr, "seeded_shots_replay_probe: effective seed mismatch\n");
    goto cleanup;
  }
  if (!outcomes || outcome_count != PROBE_SHOTS) {
    fprintf(stderr,
            "seeded_shots_replay_probe: shot count mismatch (got %zu)\n",
            outcome_count);
    goto cleanup;
  }
  for (size_t i = 0; i < outcome_count; ++i) {
    if (outcomes[i] != UINT64_C(0) && outcomes[i] != UINT64_C(3)) {
      fprintf(stderr,
              "seeded_shots_replay_probe: non-Bell outcome at index %zu\n", i);
      goto cleanup;
    }
  }

  /* This is intentionally the sole stdout write in the program. */
  printf("seed=%016" PRIx64 " shots=%zu outcomes=", effective_seed,
         outcome_count);
  for (size_t i = 0; i < outcome_count; ++i)
    printf("%s%" PRIu64, i == 0 ? "" : ",", outcomes[i]);
  putchar('\n');
  exit_code = 0;

cleanup:
  free(outcomes);
  free(circuit_text);
  if (server)
    moonlab_control_server_shutdown(server);
  if (thread_started) {
    if (pthread_join(server_tid, NULL) != 0) {
      fprintf(stderr, "seeded_shots_replay_probe: server join failed\n");
      exit_code = 1;
    } else if (server_args.rc != MOONLAB_CONTROL_OK) {
      fprintf(stderr, "seeded_shots_replay_probe: server exited (rc=%d)\n",
              server_args.rc);
      exit_code = 1;
    }
  }
  moonlab_control_server_close(server);
  return exit_code;
}
