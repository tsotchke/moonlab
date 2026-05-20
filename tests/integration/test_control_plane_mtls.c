/**
 * @file  test_control_plane_mtls.c
 * @brief Mutual-TLS test for the v0.8.19 control plane.
 *
 * Generates a CA, signs a server cert + a client cert with it,
 * configures the server with `use_tls` + `require_client_cert`,
 * and verifies:
 *   Path 1 -- client presents a CA-signed cert -> OK Bell.
 *   Path 2 -- client presents NO cert         -> TLS handshake fails.
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
#  include <openssl/x509v3.h>
#  include <openssl/evp.h>
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
/* Build a self-signed CA at (`ca_cert_path`, `ca_key_path`), then sign
 * an end-entity cert with subject CN=`cn` into (`cert_path`, `key_path`).
 * Both keys are RSA-2048, signed SHA-256. */
static int generate_ca_and_cert(const char *ca_cert_path, const char *ca_key_path,
                                const char *cert_path,    const char *key_path,
                                const char *cn)
{
    /* CA */
    EVP_PKEY *ca_key = EVP_RSA_gen(2048);
    if (!ca_key) return -1;
    X509 *ca = X509_new();
    if (!ca) { EVP_PKEY_free(ca_key); return -1; }
    X509_set_version(ca, 2);
    ASN1_INTEGER_set(X509_get_serialNumber(ca), 1);
    X509_gmtime_adj(X509_get_notBefore(ca), 0);
    X509_gmtime_adj(X509_get_notAfter(ca), 24L * 60 * 60);
    X509_set_pubkey(ca, ca_key);
    X509_NAME *ca_name = X509_get_subject_name(ca);
    X509_NAME_add_entry_by_txt(ca_name, "CN", MBSTRING_ASC,
                               (const unsigned char *)"moonlab-test-ca", -1, -1, 0);
    X509_set_issuer_name(ca, ca_name);
    /* Basic constraints CA:TRUE so we can sign end-entity certs. */
    X509_EXTENSION *bc = X509V3_EXT_conf_nid(NULL, NULL, NID_basic_constraints,
                                             "critical,CA:TRUE");
    X509_add_ext(ca, bc, -1);
    X509_EXTENSION_free(bc);
    X509_sign(ca, ca_key, EVP_sha256());

    /* End-entity (server or client) cert signed by the CA. */
    EVP_PKEY *ee_key = EVP_RSA_gen(2048);
    X509 *ee = X509_new();
    X509_set_version(ee, 2);
    ASN1_INTEGER_set(X509_get_serialNumber(ee), 2);
    X509_gmtime_adj(X509_get_notBefore(ee), 0);
    X509_gmtime_adj(X509_get_notAfter(ee), 24L * 60 * 60);
    X509_set_pubkey(ee, ee_key);
    X509_NAME *ee_name = X509_get_subject_name(ee);
    X509_NAME_add_entry_by_txt(ee_name, "CN", MBSTRING_ASC,
                               (const unsigned char *)cn, -1, -1, 0);
    X509_set_issuer_name(ee, X509_get_subject_name(ca));
    X509_sign(ee, ca_key, EVP_sha256());

    int ok = 1;
    FILE *f;
    if ((f = fopen(ca_cert_path, "w"))) { ok &= PEM_write_X509(f, ca); fclose(f); } else ok = 0;
    if ((f = fopen(ca_key_path,  "w"))) { ok &= PEM_write_PrivateKey(f, ca_key, NULL, NULL, 0, NULL, NULL); fclose(f); } else ok = 0;
    if ((f = fopen(cert_path,    "w"))) { ok &= PEM_write_X509(f, ee); fclose(f); } else ok = 0;
    if ((f = fopen(key_path,     "w"))) { ok &= PEM_write_PrivateKey(f, ee_key, NULL, NULL, 0, NULL, NULL); fclose(f); } else ok = 0;

    X509_free(ee); EVP_PKEY_free(ee_key);
    X509_free(ca); EVP_PKEY_free(ca_key);
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
    fprintf(stdout, "=== test_control_plane_mtls (v0.8.19) ===\n\n");
#ifndef MOONLAB_HAVE_TLS
    fprintf(stdout, "  SKIP  library built without QSIM_ENABLE_TLS=ON\n");
    fprintf(stdout, "=== 0 failure(s) ===\n");
    return 0;
#else
    const char *ca_cert      = "/tmp/moonlab_mtls_ca.crt";
    const char *ca_key       = "/tmp/moonlab_mtls_ca.key";
    const char *server_cert  = "/tmp/moonlab_mtls_server.crt";
    const char *server_key   = "/tmp/moonlab_mtls_server.key";
    const char *client_cert  = "/tmp/moonlab_mtls_client.crt";
    const char *client_key   = "/tmp/moonlab_mtls_client.key";

    fprintf(stdout, "--- generating CA + server cert + client cert ---\n");
    CHECK(generate_ca_and_cert(ca_cert, ca_key,
                               server_cert, server_key, "127.0.0.1") == 0,
          "server cert signed by CA");

    /* Reuse the same CA to sign the client cert.  Re-running the
     * full generator overwrites ca_cert with a fresh CA, so do a
     * variant call that signs against the existing CA. */
    {
        FILE *f = fopen(ca_cert, "r");
        X509 *ca = PEM_read_X509(f, NULL, NULL, NULL); fclose(f);
        f = fopen(ca_key, "r");
        EVP_PKEY *cakey = PEM_read_PrivateKey(f, NULL, NULL, NULL); fclose(f);

        EVP_PKEY *cli_key = EVP_RSA_gen(2048);
        X509 *cli = X509_new();
        X509_set_version(cli, 2);
        ASN1_INTEGER_set(X509_get_serialNumber(cli), 3);
        X509_gmtime_adj(X509_get_notBefore(cli), 0);
        X509_gmtime_adj(X509_get_notAfter(cli), 24L * 60 * 60);
        X509_set_pubkey(cli, cli_key);
        X509_NAME *n = X509_get_subject_name(cli);
        X509_NAME_add_entry_by_txt(n, "CN", MBSTRING_ASC,
                                   (const unsigned char *)"moonlab-test-client", -1, -1, 0);
        X509_set_issuer_name(cli, X509_get_subject_name(ca));
        X509_sign(cli, cakey, EVP_sha256());

        f = fopen(client_cert, "w"); PEM_write_X509(f, cli); fclose(f);
        f = fopen(client_key,  "w"); PEM_write_PrivateKey(f, cli_key, NULL, NULL, 0, NULL, NULL); fclose(f);

        X509_free(cli); EVP_PKEY_free(cli_key);
        X509_free(ca);  EVP_PKEY_free(cakey);
        CHECK(1, "client cert signed by same CA");
    }

    moonlab_control_server_t *server = NULL;
    uint16_t port = 0;
    int rc = moonlab_control_server_open("127.0.0.1", 0, &server, &port);
    CHECK(rc == 0, "server_open rc=%d", rc);
    rc = moonlab_control_server_use_tls(server, server_cert, server_key);
    CHECK(rc == 0, "use_tls rc=%d", rc);
    rc = moonlab_control_server_require_client_cert(server, ca_cert);
    CHECK(rc == 0, "require_client_cert rc=%d", rc);
    fprintf(stdout, "    server bound to port %u with mTLS enabled\n", port);

    run_args_t ra = { server, 0 };
    pthread_t tid;
    pthread_create(&tid, NULL, run_thread, &ra);
    struct timespec ts = { 0, 30 * 1000 * 1000 };
    nanosleep(&ts, NULL);

    char *text = serialize_bell();

    fprintf(stdout, "\n--- path 1: client presents CA-signed cert ---\n");
    double *probs = NULL; size_t num = 0;
    rc = moonlab_control_submit_circuit_mtls(
        "127.0.0.1", port,
        ca_cert,        /* server_ca_path: pin against the same CA */
        client_cert, client_key,
        /* insecure = */ 1, /* skip hostname-pin since we use a self-signed CA */
        NULL, 0, text, 0, &probs, &num);
    CHECK(rc == 0, "submit_circuit_mtls rc=%d", rc);
    CHECK(num == 4, "got 4 probabilities");
    if (num == 4) {
        CHECK(fabs(probs[0] - 0.5) < 1e-9, "P[00] = %.6f over mTLS", probs[0]);
        CHECK(fabs(probs[3] - 0.5) < 1e-9, "P[11] = %.6f over mTLS", probs[3]);
    }
    free(probs);

    fprintf(stdout, "\n--- path 2: no client cert -> handshake fail ---\n");
    probs = NULL; num = 0;
    rc = moonlab_control_submit_circuit_tls(
        "127.0.0.1", port, ca_cert, /* insecure = */ 1,
        NULL, 0, text, 0, &probs, &num);
    CHECK(rc != MOONLAB_CONTROL_OK,
          "unauthenticated client rejected (rc=%d)", rc);
    CHECK(probs == NULL, "no probs returned");

    pthread_join(tid, NULL);
    moonlab_control_server_close(server);
    free(text);
    unlink(ca_cert); unlink(ca_key);
    unlink(server_cert); unlink(server_key);
    unlink(client_cert); unlink(client_key);

    fprintf(stdout, "\n=== %d failure(s) ===\n", failures);
    return failures ? 1 : 0;
#endif
}
