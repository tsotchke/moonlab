/**
 * @file  cuda_tegra_probe.c
 * @brief Implementation -- see cuda_tegra_probe.h.
 *
 * The Tegra-vs-discrete probe is intentionally lightweight: no
 * CUDA runtime dependency, no driver init.  This file is compiled
 * into the core library (not the CUDA backend lib) so non-GPU
 * code can also branch on platform when allocating audit /
 * communication buffers that need pinned memory on some hosts.
 *
 * On Jetson L4T R35+ the device-tree compatible string contains
 * "nvidia,tegra194" (Xavier), "nvidia,tegra234" (Orin), or similar.
 * Older boards exposed "nvidia,tegra210" / "nvidia,tegra186" --
 * we accept any "nvidia,tegra*" prefix.
 */

#include "cuda_tegra_probe.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static pthread_once_t        s_probe_once = PTHREAD_ONCE_INIT;
static moonlab_gpu_kind_t    s_cached_kind = MOONLAB_GPU_KIND_UNKNOWN;
static moonlab_gpu_kind_t    s_forced_kind = MOONLAB_GPU_KIND_UNKNOWN;
static pthread_mutex_t       s_force_lock = PTHREAD_MUTEX_INITIALIZER;

static int read_compatible_has_tegra(void)
{
    /* /proc/device-tree/compatible is a sequence of NUL-terminated
     * strings.  We scan all of them for the "nvidia,tegra" prefix. */
    FILE *f = fopen("/proc/device-tree/compatible", "rb");
    if (!f) return 0;
    char buf[512];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    if (n == 0) return 0;
    buf[n] = '\0';
    /* Walk NUL-separated entries.  Each ends inside [0, n). */
    size_t pos = 0;
    while (pos < n) {
        const char *entry = &buf[pos];
        size_t len = strnlen(entry, n - pos);
        if (len >= 12 && strncmp(entry, "nvidia,tegra", 12) == 0) {
            return 1;
        }
        pos += len + 1;
    }
    return 0;
}

static void probe_init(void)
{
    if (read_compatible_has_tegra()) {
        s_cached_kind = MOONLAB_GPU_KIND_TEGRA;
        return;
    }
    /* Without the device-tree hint we don't try cudaGetDeviceCount
     * here: this file is compiled without the CUDA runtime, and
     * splitting probe into a separate _cuda.cu file just to detect
     * "is discrete GPU present" adds dependency weight for marginal
     * benefit.  The discrete code path is responsible for its own
     * cudaGetDeviceCount; this probe just answers "is this a
     * Jetson?".  Anything else -> UNKNOWN, and the caller falls
     * back to the discrete-style memory model OR the CPU path. */
    s_cached_kind = MOONLAB_GPU_KIND_UNKNOWN;
}

moonlab_gpu_kind_t moonlab_gpu_probe_kind(void)
{
    pthread_mutex_lock(&s_force_lock);
    moonlab_gpu_kind_t forced = s_forced_kind;
    pthread_mutex_unlock(&s_force_lock);
    if (forced != MOONLAB_GPU_KIND_UNKNOWN) return forced;
    pthread_once(&s_probe_once, probe_init);
    return s_cached_kind;
}

const char *moonlab_gpu_probe_kind_str(moonlab_gpu_kind_t k)
{
    switch (k) {
        case MOONLAB_GPU_KIND_TEGRA:    return "tegra";
        case MOONLAB_GPU_KIND_DISCRETE: return "discrete";
        case MOONLAB_GPU_KIND_NONE:     return "none";
        case MOONLAB_GPU_KIND_UNKNOWN:
        default:                        return "unknown";
    }
}

void moonlab_gpu_probe_force(moonlab_gpu_kind_t k)
{
    pthread_mutex_lock(&s_force_lock);
    s_forced_kind = k;
    pthread_mutex_unlock(&s_force_lock);
}
