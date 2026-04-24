/**
 * @file test_hardware_entropy_probe.c
 * @brief Security regression tests for the ARM hardware RNG helper launch path.
 */

#include "../../src/applications/hardware_entropy.h"

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int failures = 0;

#define CHECK(cond, fmt, ...) do {                              \
    if (!(cond)) {                                              \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);    \
        failures++;                                             \
    } else {                                                    \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);    \
    }                                                           \
} while (0)

#if defined(__aarch64__)
static int write_malicious_helper(const char *root, const char *marker_path) {
    char tools_dir[PATH_MAX];
    char helper_path[PATH_MAX];

    if (snprintf(tools_dir, sizeof(tools_dir), "%s/tools", root) >= (int)sizeof(tools_dir)) {
        return -1;
    }
    if (mkdir(tools_dir, 0700) != 0) {
        return -1;
    }

    if (snprintf(helper_path, sizeof(helper_path), "%s/hw_rng_probe", tools_dir) >= (int)sizeof(helper_path)) {
        return -1;
    }

    FILE *fp = fopen(helper_path, "w");
    if (!fp) {
        return -1;
    }

    fprintf(fp,
            "#!/bin/sh\n"
            "printf 'owned' > '%s'\n"
            "printf '0000000000000001\\n'\n",
            marker_path);
    fclose(fp);

    if (chmod(helper_path, 0700) != 0) {
        return -1;
    }

    return 0;
}

static void test_probe_ignores_cwd_helper(void) {
    fprintf(stdout, "\n-- hardware entropy probe ignores cwd helper --\n");

    CHECK(MOONLAB_HW_RNG_PROBE_PATH[0] == '/',
          "compiled helper path is absolute: %s", MOONLAB_HW_RNG_PROBE_PATH);

    char template_dir[] = "/tmp/moonlab-hw-probe-XXXXXX";
    char *temp_dir = mkdtemp(template_dir);
    CHECK(temp_dir != NULL, "mkdtemp succeeded");
    if (!temp_dir) {
        return;
    }

    char marker_path[PATH_MAX];
    if (snprintf(marker_path, sizeof(marker_path), "%s/pwned", temp_dir) >= (int)sizeof(marker_path)) {
        CHECK(0, "marker path fits");
        return;
    }

    CHECK(write_malicious_helper(temp_dir, marker_path) == 0,
          "wrote malicious cwd helper");

    char original_cwd[PATH_MAX];
    char *cwd_result = getcwd(original_cwd, sizeof(original_cwd));
    CHECK(cwd_result != NULL,
          "captured original cwd");
    if (!cwd_result) {
        return;
    }

    int chdir_rc = chdir(temp_dir);
    CHECK(chdir_rc == 0, "changed cwd into temp dir");
    if (chdir_rc != 0) {
        return;
    }

    uint64_t value = 0;
    (void)moonlab_hw_rng_probe_exec("rndr", &value);

    CHECK(chdir(original_cwd) == 0, "restored cwd");
    CHECK(access(marker_path, F_OK) != 0,
          "malicious ./tools/hw_rng_probe was not executed");
}
#endif

int main(void) {
    fprintf(stdout, "=== hardware entropy probe tests ===\n");

#if defined(__aarch64__)
    test_probe_ignores_cwd_helper();
#else
    fprintf(stdout, "\n-- skipped (non-aarch64 target) --\n");
#endif

    fprintf(stdout, "\n%d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
