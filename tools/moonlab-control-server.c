/**
 * @file  moonlab-control-server.c
 * @brief Production CLI runner for the Moonlab control plane.
 *
 * Stands up an in-process control-plane server, configures it from
 * command-line flags + environment variables, and runs it until
 * SIGINT / SIGTERM.  Used by the deploy/docker stack as the daemon
 * inside the moonlab-control image.
 */

#include "../src/control/control_plane.h"

#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static moonlab_control_server_t *g_server = NULL;

static void on_signal(int sig) {
    (void)sig;
    if (g_server) moonlab_control_server_shutdown(g_server);
}

static void usage(const char *prog) {
    fprintf(stderr,
        "moonlab-control-server: production daemon for the moonlab line-protocol control plane\n"
        "\n"
        "usage: %s [flags]\n"
        "\n"
        "transport:\n"
        "  --host HOST              bind address (default 0.0.0.0; use :: for dual-stack)\n"
        "  --port PORT              TCP port (default 7070)\n"
        "  --tls-cert PATH          server cert (PEM); enables TLS\n"
        "  --tls-key PATH           server key  (PEM); requires --tls-cert\n"
        "  --client-ca PATH         require client cert chained to this CA (mTLS)\n"
        "\n"
        "auth + defensive layers:\n"
        "  --secret-file PATH       HMAC-SHA3-256 shared secret (raw bytes)\n"
        "  --rate-limit-rps N       per-IP token-bucket refill rate (0 = off)\n"
        "  --rate-limit-burst N     per-IP burst capacity\n"
        "  --request-timeout SECS   per-request socket timeout (0 = off)\n"
        "  --max-concurrent N       max concurrent worker threads (0 = off)\n"
        "\n"
        "observability:\n"
        "  --log-format text|json   structured request log; opt in via MOONLAB_CONTROL_LOG=1 env var\n"
        "\n"
        "  --help                   print this and exit\n"
        "\n",
        prog);
}

int main(int argc, char **argv) {
    const char *host           = "0.0.0.0";
    int         port           = 7070;
    const char *tls_cert       = NULL;
    const char *tls_key        = NULL;
    const char *client_ca      = NULL;
    const char *secret_file    = NULL;
    int         rate_rps       = 0;
    int         rate_burst     = 0;
    int         request_to     = 0;
    int         max_concurrent = 0;
    const char *log_format     = NULL;

    static const struct option opts[] = {
        { "host",            required_argument, NULL, 'H' },
        { "port",            required_argument, NULL, 'p' },
        { "tls-cert",        required_argument, NULL, 'C' },
        { "tls-key",         required_argument, NULL, 'K' },
        { "client-ca",       required_argument, NULL, 'A' },
        { "secret-file",     required_argument, NULL, 'S' },
        { "rate-limit-rps",  required_argument, NULL, 'r' },
        { "rate-limit-burst",required_argument, NULL, 'b' },
        { "request-timeout", required_argument, NULL, 't' },
        { "max-concurrent",  required_argument, NULL, 'm' },
        { "log-format",      required_argument, NULL, 'l' },
        { "help",            no_argument,       NULL, 'h' },
        { NULL, 0, NULL, 0 },
    };

    int opt, idx;
    while ((opt = getopt_long(argc, argv, "h", opts, &idx)) != -1) {
        switch (opt) {
            case 'H': host           = optarg; break;
            case 'p': port           = atoi(optarg); break;
            case 'C': tls_cert       = optarg; break;
            case 'K': tls_key        = optarg; break;
            case 'A': client_ca      = optarg; break;
            case 'S': secret_file    = optarg; break;
            case 'r': rate_rps       = atoi(optarg); break;
            case 'b': rate_burst     = atoi(optarg); break;
            case 't': request_to     = atoi(optarg); break;
            case 'm': max_concurrent = atoi(optarg); break;
            case 'l': log_format     = optarg; break;
            case 'h': default: usage(argv[0]); return opt == 'h' ? 0 : 2;
        }
    }
    if (port <= 0 || port > 65535) {
        fprintf(stderr, "error: --port out of range\n");
        return 2;
    }

    if (log_format) {
        setenv("MOONLAB_CONTROL_LOG", "1", 1);
        setenv("MOONLAB_CONTROL_LOG_FORMAT", log_format, 1);
    }

    uint16_t bound = 0;
    int rc = moonlab_control_server_open(host, (uint16_t)port, &g_server, &bound);
    if (rc != 0) {
        fprintf(stderr, "error: server_open(%s, %d) rc=%d\n", host, port, rc);
        return 1;
    }

    if (secret_file) {
        FILE *f = fopen(secret_file, "rb");
        if (!f) {
            fprintf(stderr, "error: cannot open --secret-file %s: %s\n",
                    secret_file, strerror(errno));
            moonlab_control_server_close(g_server);
            return 1;
        }
        uint8_t buf[256];
        size_t n = fread(buf, 1, sizeof(buf), f);
        fclose(f);
        if (n == 0) {
            fprintf(stderr, "error: --secret-file is empty\n");
            moonlab_control_server_close(g_server);
            return 1;
        }
        rc = moonlab_control_server_set_secret(g_server, buf, n);
        if (rc != 0) {
            fprintf(stderr, "error: set_secret rc=%d\n", rc);
            moonlab_control_server_close(g_server);
            return 1;
        }
    }

    if (tls_cert) {
        if (!tls_key) {
            fprintf(stderr, "error: --tls-cert requires --tls-key\n");
            moonlab_control_server_close(g_server);
            return 2;
        }
        rc = moonlab_control_server_use_tls(g_server, tls_cert, tls_key);
        if (rc != 0) {
            fprintf(stderr, "error: use_tls rc=%d\n", rc);
            moonlab_control_server_close(g_server);
            return 1;
        }
    }
    if (client_ca) {
        rc = moonlab_control_server_require_client_cert(g_server, client_ca);
        if (rc != 0) {
            fprintf(stderr, "error: require_client_cert rc=%d\n", rc);
            moonlab_control_server_close(g_server);
            return 1;
        }
    }
    if (rate_rps > 0 || rate_burst > 0) {
        rc = moonlab_control_server_set_rate_limit(g_server, rate_rps,
                                                    rate_burst > 0
                                                        ? rate_burst
                                                        : rate_rps);
        if (rc != 0) {
            fprintf(stderr, "error: set_rate_limit rc=%d\n", rc);
            moonlab_control_server_close(g_server);
            return 1;
        }
    }
    if (request_to > 0) {
        rc = moonlab_control_server_set_request_timeout(g_server, request_to);
        if (rc != 0) {
            fprintf(stderr, "error: set_request_timeout rc=%d\n", rc);
            moonlab_control_server_close(g_server);
            return 1;
        }
    }
    if (max_concurrent > 0) {
        rc = moonlab_control_server_set_max_concurrent(g_server, max_concurrent);
        if (rc != 0) {
            fprintf(stderr, "error: set_max_concurrent rc=%d\n", rc);
            moonlab_control_server_close(g_server);
            return 1;
        }
    }

    /* Graceful shutdown on SIGINT / SIGTERM. */
    struct sigaction sa = { 0 };
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT,  &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    fprintf(stdout, "moonlab-control-server: listening on %s:%u\n",
            host, (unsigned)bound);
    if (secret_file)   fprintf(stdout, "  auth: HMAC-SHA3-256\n");
    if (tls_cert)      fprintf(stdout, "  tls:  yes\n");
    if (client_ca)     fprintf(stdout, "  mtls: yes (CA=%s)\n", client_ca);
    if (rate_rps)      fprintf(stdout, "  rate: %d req/s burst %d\n",
                               rate_rps, rate_burst);
    if (request_to)    fprintf(stdout, "  request_timeout: %ds\n", request_to);
    if (max_concurrent)fprintf(stdout, "  max_concurrent: %d\n", max_concurrent);
    fflush(stdout);

    rc = moonlab_control_server_run(g_server, INT_MAX);
    moonlab_control_server_close(g_server);
    g_server = NULL;
    return rc == 0 ? 0 : 1;
}
