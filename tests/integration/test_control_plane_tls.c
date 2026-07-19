/**
 * @file  test_control_plane_tls.c
 * @brief End-to-end TLS test for the v0.8.17 control plane.
 *
 * Generates an in-process self-signed RSA-2048 / SHA-256 cert,
 * writes the cert + private key to temp files, starts the server
 * with `moonlab_control_server_use_tls`, then submits a Bell
 * circuit over TLS via `moonlab_control_submit_circuit_tls` and
 * verifies the (0.5, 0, 0, 0.5) signature.
 *
 * Skipped (test PASS, no assertions) when the library was built
 * without QSIM_ENABLE_TLS=ON.
 */

#include "../../src/control/control_plane.h"
#include "../../src/applications/moonlab_qgtl_backend.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifdef MOONLAB_HAVE_TLS
#  include <openssl/pem.h>
#  include <openssl/rsa.h>
#  include <openssl/x509.h>
#  include <openssl/evp.h>
#  include "test_tls_keygen.h"
#endif

static int failures = 0;
#define CHECK(cond, fmt, ...) do {                              \
    if (!(cond)) {                                              \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);    \
        failures++;                                             \
    } else {                                                    \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);    \
    }                                                           \
} while (0)

#ifdef MOONLAB_HAVE_TLS
/* Generate an RSA-2048 / SHA-256 self-signed certificate for
 * "127.0.0.1" valid for 1 day.  Writes the PEM cert + private key
 * to the two paths provided.  Returns 0 on success. */
static int generate_self_signed(const char *cert_path, const char *key_path)
{
    EVP_PKEY *pkey = moonlab_test_generate_rsa_key();
    if (!pkey) return -1;

    X509 *x = X509_new();
    if (!x) { EVP_PKEY_free(pkey); return -1; }
    X509_set_version(x, 2);  /* v3 */
    ASN1_INTEGER_set(X509_get_serialNumber(x), 1);
    X509_gmtime_adj(X509_get_notBefore(x), 0);
    X509_gmtime_adj(X509_get_notAfter(x), 24L * 60 * 60);
    X509_set_pubkey(x, pkey);

    X509_NAME *name = X509_get_subject_name(x);
    X509_NAME_add_entry_by_txt(name, "CN", MBSTRING_ASC,
                               (const unsigned char *)"127.0.0.1", -1, -1, 0);
    X509_set_issuer_name(x, name);

    X509_sign(x, pkey, EVP_sha256());

    FILE *cf = fopen(cert_path, "w");
    FILE *kf = fopen(key_path,  "w");
    int ok = (cf && kf &&
              PEM_write_X509(cf, x) == 1 &&
              PEM_write_PrivateKey(kf, pkey, NULL, NULL, 0, NULL, NULL) == 1);
    if (cf) fclose(cf);
    if (kf) fclose(kf);
    X509_free(x);
    EVP_PKEY_free(pkey);
    return ok ? 0 : -1;
}

typedef struct {
    moonlab_control_server_t *server;
    int                       rc;
} run_args_t;

static void *run_thread(void *arg)
{
    run_args_t *a = (run_args_t *)arg;
    a->rc = moonlab_control_server_run(a->server, 2);
    return NULL;
}

static char *serialize_bell(void)
{
    moonlab_qgtl_circuit_t *c = moonlab_qgtl_circuit_create(2);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_H,    0, 0, NULL);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_CNOT, 1, 0, NULL);
    size_t needed = 0;
    moonlab_qgtl_circuit_serialize(c, NULL, 0, &needed);
    char *buf = (char *)malloc(needed + 1);
    moonlab_qgtl_circuit_serialize(c, buf, needed + 1, NULL);
    moonlab_qgtl_circuit_free(c);
    return buf;
}
#endif

int main(void)
{
    fprintf(stdout, "=== test_control_plane_tls (v0.8.17) ===\n\n");
#ifndef MOONLAB_HAVE_TLS
    fprintf(stdout, "  SKIP  library built without QSIM_ENABLE_TLS=ON\n");
    fprintf(stdout, "=== 0 failure(s) ===\n");
    return 0;
#else
    const char *cert_path = "/tmp/moonlab_tls_cert.pem";
    const char *key_path  = "/tmp/moonlab_tls_key.pem";

    fprintf(stdout, "--- generating self-signed cert ---\n");
    CHECK(generate_self_signed(cert_path, key_path) == 0,
          "self-signed RSA-2048/SHA-256 cert at %s", cert_path);

    moonlab_control_server_t *server = NULL;
    uint16_t port = 0;
    int rc = moonlab_control_server_open("127.0.0.1", 0, &server, &port);
    CHECK(rc == 0, "server_open rc=%d", rc);
    rc = moonlab_control_server_use_tls(server, cert_path, key_path);
    CHECK(rc == 0, "use_tls rc=%d", rc);
    fprintf(stdout, "    server bound to port %u with TLS enabled\n", port);

    run_args_t ra = { server, 0 };
    pthread_t tid;
    pthread_create(&tid, NULL, run_thread, &ra);
    /* Give the server a tick to enter accept(). */
    struct timespec ts = { 0, 30 * 1000 * 1000 };
    nanosleep(&ts, NULL);

    char *text = serialize_bell();

    fprintf(stdout, "\n--- submit Bell circuit over TLS (insecure) ---\n");
    double *probs = NULL; size_t num = 0;
    rc = moonlab_control_submit_circuit_tls(
        "127.0.0.1", port, NULL, /* insecure = */ 1,
        NULL, 0, text, 0, &probs, &num);
    CHECK(rc == 0, "submit_circuit_tls rc=%d", rc);
    CHECK(num == 4, "got 4 probabilities (got %zu)", num);
    if (num == 4) {
        CHECK(fabs(probs[0] - 0.5) < 1e-9, "P[00] = %.6f over TLS", probs[0]);
        CHECK(fabs(probs[1])       < 1e-9, "P[01] = %.6f", probs[1]);
        CHECK(fabs(probs[2])       < 1e-9, "P[10] = %.6f", probs[2]);
        CHECK(fabs(probs[3] - 0.5) < 1e-9, "P[11] = %.6f over TLS", probs[3]);
    }
    free(probs);

    fprintf(stdout, "\n--- second submit to drain server max_iters=2 ---\n");
    probs = NULL; num = 0;
    rc = moonlab_control_submit_circuit_tls(
        "127.0.0.1", port, NULL, 1, NULL, 0, text, 0, &probs, &num);
    CHECK(rc == 0, "second TLS submit rc=%d", rc);
    free(probs);

    pthread_join(tid, NULL);
    CHECK(ra.rc == 0, "server thread exit rc=%d", ra.rc);

    moonlab_control_server_close(server);
    free(text);
    unlink(cert_path);
    unlink(key_path);

    fprintf(stdout, "\n=== %d failure(s) ===\n", failures);
    return failures ? 1 : 0;
#endif
}
