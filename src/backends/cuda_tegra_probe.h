/**
 * @file  cuda_tegra_probe.h
 * @brief Runtime detection of Jetson/Tegra integrated-GPU platforms.
 *
 * Jetson SoCs (Tegra194 / Tegra234 / Orin family) ship NVIDIA GPUs
 * that share LPDDR with the CPU through a unified memory
 * controller.  Discrete GPUs (Hopper / Ada / Ampere desktop) have
 * a separate VRAM pool and require explicit cudaMemcpy.
 *
 * Picking the wrong memory model has large perf consequences:
 *   - On Jetson, calling cudaMemcpy(host -> device) is pure waste:
 *     the pages are already DMA-mappable, we're just doing
 *     unnecessary copies inside the same DRAM bank.
 *   - On discrete, calling cudaHostAllocMapped does PCIe-paged
 *     access for every load, ~10x slower than a proper copy.
 *
 * Moonlab dispatches between the two paths based on this probe.
 * Probe is cached on first call; cheap to re-call.
 */

#ifndef MOONLAB_CUDA_TEGRA_PROBE_H
#define MOONLAB_CUDA_TEGRA_PROBE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MOONLAB_GPU_KIND_UNKNOWN  = 0,
    MOONLAB_GPU_KIND_TEGRA    = 1,  /* Jetson / DRIVE / Tegra family.  Unified-memory device. */
    MOONLAB_GPU_KIND_DISCRETE = 2,  /* Discrete PCIe GPU.  Separate VRAM. */
    MOONLAB_GPU_KIND_NONE     = 3,  /* No CUDA-capable device available. */
} moonlab_gpu_kind_t;

/**
 * @brief Determine the GPU architecture class on this host.
 *
 * Probes in this order:
 *   1. /proc/device-tree/compatible — Tegra SoCs declare "nvidia,tegra*".
 *   2. cudaGetDeviceCount + cudaGetDeviceProperties.integrated — runtime
 *      check (only useful if libcuda is linked).
 *   3. Fallback: assume DISCRETE if a CUDA device is present and we
 *      can't tell otherwise.
 *
 * Cached after first call.  Thread-safe (init via pthread_once).
 */
moonlab_gpu_kind_t moonlab_gpu_probe_kind(void);

/**
 * @brief Human-readable name for telemetry / logs.
 */
const char *moonlab_gpu_probe_kind_str(moonlab_gpu_kind_t k);

/**
 * @brief Force a kind (useful for tests + CI lanes that want to
 *        validate the discrete-memory path on a Jetson host).
 *        Pass MOONLAB_GPU_KIND_UNKNOWN to clear and re-probe.
 */
void moonlab_gpu_probe_force(moonlab_gpu_kind_t k);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_CUDA_TEGRA_PROBE_H */
