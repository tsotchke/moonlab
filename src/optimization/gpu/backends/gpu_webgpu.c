/**
 * @file gpu_webgpu.c
 * @brief WebGPU backend scaffold for WASM builds
 *
 * This implementation provides runtime detection, deterministic host-visible
 * fallback, and a first native WebGPU compute slice for key operations.
 *
 * @stability beta
 * @since v0.1.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#include "gpu_webgpu.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>

#if defined(HAS_WEBGPU) && defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#include <stdint.h>

EM_JS(int, moonlab_webgpu_runtime_available, (), {
    try {
        if (typeof navigator === 'undefined' || !navigator.gpu) {
            return 0;
        }
        if (typeof self !== 'undefined' && self.isSecureContext === false) {
            return 0;
        }
        return 1;
    } catch (err) {
        return 0;
    }
});

EM_JS(int, moonlab_webgpu_native_dispatch_supported, (), {
    try {
        if (typeof navigator === 'undefined' || !navigator.gpu) {
            return 0;
        }
        // Deno path is currently unstable with Asyncify during native dispatch.
        // Keep native compute off by default in Deno unless explicitly enabled.
        if (typeof Deno !== 'undefined' &&
            Deno.env &&
            typeof Deno.env.get === 'function') {
            try {
                if (Deno.env.get('MOONLAB_WEBGPU_ENABLE_DENO_NATIVE') !== '1') {
                    return 0;
                }
                if (Deno.env.get('MOONLAB_WEBGPU_DISABLE_DENO_NATIVE') === '1') {
                    return 0;
                }
            } catch (_err) {
                // If env access fails, stay conservative and keep native disabled.
                return 0;
            }
        }
        return 1;
    } catch (err) {
        return 0;
    }
});

EM_JS(int, moonlab_webgpu_tn_native_dispatch_supported, (), {
    try {
        if (!moonlab_webgpu_native_dispatch_supported()) {
            return 0;
        }
        // Tensor-network kernels operate on short-lived intermediate buffers.
        // In Deno, asyncify scheduling can resume after the caller returns,
        // so keep TN native dispatch disabled unless explicitly opted in.
        if (typeof Deno !== 'undefined' &&
            Deno.env &&
            typeof Deno.env.get === 'function') {
            try {
                if (Deno.env.get('MOONLAB_WEBGPU_ENABLE_DENO_NATIVE_TN') !== '1') {
                    return 0;
                }
                if (Deno.env.get('MOONLAB_WEBGPU_DISABLE_DENO_NATIVE_TN') === '1') {
                    return 0;
                }
            } catch (_err) {
                return 0;
            }
        }
        return 1;
    } catch (err) {
        return 0;
    }
});

EM_ASYNC_JS(int, moonlab_webgpu_init_async, (), {
    try {
        if (typeof navigator === 'undefined' || !navigator.gpu) {
            return 0;
        }

        const state = Module.__moonlabWebGPU || (Module.__moonlabWebGPU = {});
        if (state.device &&
            state.hadamardPipeline &&
            state.pauliXPipeline &&
            state.pauliZPipeline &&
            state.cnotPipeline &&
            state.probabilitiesPipeline &&
            state.mpsApplyGateThetaPipeline &&
            state.mpsExpectationZCanonicalPipeline) {
            return 1;
        }
        if (!state.initPromise) {
            state.initPromise = (async () => {
                const adapter = await navigator.gpu.requestAdapter();
                if (!adapter) {
                    return 0;
                }

                const device = await adapter.requestDevice();
                const shaderCode = `
struct HadamardParams {
  qubit: u32,
  state_dim: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> hadamard_src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> hadamard_dst: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> hadamard_params: HadamardParams;

@compute @workgroup_size(256)
fn hadamard_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pair_index = gid.x;
  let pair_count = hadamard_params.state_dim / 2u;
  if (pair_index >= pair_count) {
    return;
  }

  let stride = 1u << hadamard_params.qubit;
  let i0 = (pair_index / stride) * (2u * stride) + (pair_index % stride);
  let i1 = i0 + stride;
  let v0 = hadamard_src[i0];
  let v1 = hadamard_src[i1];
  let inv_sqrt2 = 0.7071067811865476;
  hadamard_dst[i0] = (v0 + v1) * inv_sqrt2;
  hadamard_dst[i1] = (v0 - v1) * inv_sqrt2;
}

struct PauliXParams {
  qubit: u32,
  state_dim: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> pauli_x_src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> pauli_x_dst: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> pauli_x_params: PauliXParams;

@compute @workgroup_size(256)
fn pauli_x_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pair_index = gid.x;
  let pair_count = pauli_x_params.state_dim / 2u;
  if (pair_index >= pair_count) {
    return;
  }

  let stride = 1u << pauli_x_params.qubit;
  let i0 = (pair_index / stride) * (2u * stride) + (pair_index % stride);
  let i1 = i0 + stride;
  pauli_x_dst[i0] = pauli_x_src[i1];
  pauli_x_dst[i1] = pauli_x_src[i0];
}

struct PauliZParams {
  qubit: u32,
  state_dim: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> pauli_z_src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> pauli_z_dst: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> pauli_z_params: PauliZParams;

@compute @workgroup_size(256)
fn pauli_z_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= pauli_z_params.state_dim) {
    return;
  }

  var v = pauli_z_src[i];
  if ((i & (1u << pauli_z_params.qubit)) != 0u) {
    v = -v;
  }
  pauli_z_dst[i] = v;
}

struct CnotParams {
  control: u32,
  target: u32,
  state_dim: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> cnot_src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> cnot_dst: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> cnot_params: CnotParams;

@compute @workgroup_size(256)
fn cnot_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pair_index = gid.x;
  let pair_count = cnot_params.state_dim / 2u;
  if (pair_index >= pair_count) {
    return;
  }

  let target_stride = 1u << cnot_params.target;
  let i0 = (pair_index / target_stride) * (2u * target_stride) + (pair_index % target_stride);
  let i1 = i0 + target_stride;
  if ((i0 & (1u << cnot_params.control)) != 0u) {
    cnot_dst[i0] = cnot_src[i1];
    cnot_dst[i1] = cnot_src[i0];
  } else {
    cnot_dst[i0] = cnot_src[i0];
    cnot_dst[i1] = cnot_src[i1];
  }
}

struct ProbabilityParams {
  state_dim: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> prob_src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> prob_dst: array<f32>;
@group(0) @binding(2) var<uniform> prob_params: ProbabilityParams;

@compute @workgroup_size(256)
fn probabilities_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= prob_params.state_dim) {
    return;
  }
  let amp = prob_src[i];
  prob_dst[i] = dot(amp, amp);
}

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(
    a.x * b.x - a.y * b.y,
    a.x * b.y + a.y * b.x
  );
}

struct MpsGateParams {
  chi_l: u32,
  chi_r: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> mps_theta_src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> mps_theta_dst: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> mps_gate: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> mps_gate_params: MpsGateParams;

@compute @workgroup_size(256)
fn mps_apply_gate_theta_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pair = gid.x;
  let total_pairs = mps_gate_params.chi_l * mps_gate_params.chi_r;
  if (pair >= total_pairs) {
    return;
  }

  let base = pair * 4u;
  let t0 = mps_theta_src[base + 0u];
  let t1 = mps_theta_src[base + 1u];
  let t2 = mps_theta_src[base + 2u];
  let t3 = mps_theta_src[base + 3u];

  let g00 = mps_gate[0u];
  let g01 = mps_gate[1u];
  let g02 = mps_gate[2u];
  let g03 = mps_gate[3u];
  let g10 = mps_gate[4u];
  let g11 = mps_gate[5u];
  let g12 = mps_gate[6u];
  let g13 = mps_gate[7u];
  let g20 = mps_gate[8u];
  let g21 = mps_gate[9u];
  let g22 = mps_gate[10u];
  let g23 = mps_gate[11u];
  let g30 = mps_gate[12u];
  let g31 = mps_gate[13u];
  let g32 = mps_gate[14u];
  let g33 = mps_gate[15u];

  mps_theta_dst[base + 0u] =
      complex_mul(g00, t0) + complex_mul(g01, t1) + complex_mul(g02, t2) + complex_mul(g03, t3);
  mps_theta_dst[base + 1u] =
      complex_mul(g10, t0) + complex_mul(g11, t1) + complex_mul(g12, t2) + complex_mul(g13, t3);
  mps_theta_dst[base + 2u] =
      complex_mul(g20, t0) + complex_mul(g21, t1) + complex_mul(g22, t2) + complex_mul(g23, t3);
  mps_theta_dst[base + 3u] =
      complex_mul(g30, t0) + complex_mul(g31, t1) + complex_mul(g32, t2) + complex_mul(g33, t3);
}

struct MpsExpectationParams {
  chi_l: u32,
  chi_r: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> mps_tensor: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> mps_pair_probs: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> mps_expect_params: MpsExpectationParams;

@compute @workgroup_size(256)
fn mps_expectation_z_canonical_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pair = gid.x;
  let total_pairs = mps_expect_params.chi_l * mps_expect_params.chi_r;
  if (pair >= total_pairs) {
    return;
  }

  let base = pair * 2u;
  let a0 = mps_tensor[base + 0u];
  let a1 = mps_tensor[base + 1u];
  let p0 = dot(a0, a0);
  let p1 = dot(a1, a1);
  mps_pair_probs[pair] = vec2<f32>(p0, p1);
}
`;

                const shaderModule = device.createShaderModule({ code: shaderCode });
                const hadamardPipeline = device.createComputePipeline({
                    layout: 'auto',
                    compute: {
                        module: shaderModule,
                        entryPoint: 'hadamard_kernel',
                    },
                });
                const pauliXPipeline = device.createComputePipeline({
                    layout: 'auto',
                    compute: {
                        module: shaderModule,
                        entryPoint: 'pauli_x_kernel',
                    },
                });
                const pauliZPipeline = device.createComputePipeline({
                    layout: 'auto',
                    compute: {
                        module: shaderModule,
                        entryPoint: 'pauli_z_kernel',
                    },
                });
                const cnotPipeline = device.createComputePipeline({
                    layout: 'auto',
                    compute: {
                        module: shaderModule,
                        entryPoint: 'cnot_kernel',
                    },
                });
                const probabilitiesPipeline = device.createComputePipeline({
                    layout: 'auto',
                    compute: {
                        module: shaderModule,
                        entryPoint: 'probabilities_kernel',
                    },
                });
                const mpsApplyGateThetaPipeline = device.createComputePipeline({
                    layout: 'auto',
                    compute: {
                        module: shaderModule,
                        entryPoint: 'mps_apply_gate_theta_kernel',
                    },
                });
                const mpsExpectationZCanonicalPipeline = device.createComputePipeline({
                    layout: 'auto',
                    compute: {
                        module: shaderModule,
                        entryPoint: 'mps_expectation_z_canonical_kernel',
                    },
                });

                state.adapter = adapter;
                state.device = device;
                state.hadamardPipeline = hadamardPipeline;
                state.pauliXPipeline = pauliXPipeline;
                state.pauliZPipeline = pauliZPipeline;
                state.cnotPipeline = cnotPipeline;
                state.probabilitiesPipeline = probabilitiesPipeline;
                state.mpsApplyGateThetaPipeline = mpsApplyGateThetaPipeline;
                state.mpsExpectationZCanonicalPipeline = mpsExpectationZCanonicalPipeline;
                state.workgroupSize = 256;
                return 1;
            })().catch((_err) => 0);
        }

        const ok = await state.initPromise;
        if (!ok) {
            state.initPromise = null;
        }
        return ok ? 1 : 0;
    } catch (err) {
        return 0;
    }
});

EM_ASYNC_JS(int, moonlab_webgpu_hadamard_dispatch_async,
            (uintptr_t amplitudes_ptr, uint32_t qubit_index, uint32_t state_dim), {
    try {
        const initialized = await moonlab_webgpu_init_async();
        if (!initialized) {
            return 0;
        }

        const state = Module.__moonlabWebGPU;
        const device = state.device;
        const n = state_dim >>> 0;
        const valueCount = n * 2;
        const heapOffset = amplitudes_ptr >>> 3;  // bytes -> f64 index

        const amplitudesF32 = new Float32Array(valueCount);
        for (let i = 0; i < valueCount; i++) {
            amplitudesF32[i] = HEAPF64[heapOffset + i];
        }

        const amplitudesBytes = amplitudesF32.byteLength;
        const src = device.createBuffer({
            size: amplitudesBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const dst = device.createBuffer({
            size: amplitudesBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const readback = device.createBuffer({
            size: amplitudesBytes,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const params = new Uint32Array([qubit_index >>> 0, n, 0, 0]);
        const paramsBuf = device.createBuffer({
            size: params.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        device.queue.writeBuffer(src, 0, amplitudesF32.buffer, amplitudesF32.byteOffset, amplitudesF32.byteLength);
        device.queue.writeBuffer(paramsBuf, 0, params.buffer, params.byteOffset, params.byteLength);

        const bindGroup = device.createBindGroup({
            layout: state.hadamardPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: src } },
                { binding: 1, resource: { buffer: dst } },
                { binding: 2, resource: { buffer: paramsBuf } },
            ],
        });

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(state.hadamardPipeline);
        pass.setBindGroup(0, bindGroup);
        const pairs = n >>> 1;
        const workgroups = Math.max(1, Math.ceil(pairs / state.workgroupSize));
        pass.dispatchWorkgroups(workgroups);
        pass.end();
        encoder.copyBufferToBuffer(dst, 0, readback, 0, amplitudesBytes);
        device.queue.submit([encoder.finish()]);

        await readback.mapAsync(GPUMapMode.READ);
        const mapped = readback.getMappedRange();
        const resultF32 = new Float32Array(mapped.slice(0));
        readback.unmap();

        for (let i = 0; i < valueCount; i++) {
            HEAPF64[heapOffset + i] = resultF32[i];
        }

        if (typeof src.destroy === 'function') src.destroy();
        if (typeof dst.destroy === 'function') dst.destroy();
        if (typeof readback.destroy === 'function') readback.destroy();
        if (typeof paramsBuf.destroy === 'function') paramsBuf.destroy();
        return 1;
    } catch (err) {
        return 0;
    }
});

EM_ASYNC_JS(int, moonlab_webgpu_pauli_x_dispatch_async,
            (uintptr_t amplitudes_ptr, uint32_t qubit_index, uint32_t state_dim), {
    try {
        const initialized = await moonlab_webgpu_init_async();
        if (!initialized) {
            return 0;
        }

        const state = Module.__moonlabWebGPU;
        const device = state.device;
        const n = state_dim >>> 0;
        const valueCount = n * 2;
        const heapOffset = amplitudes_ptr >>> 3;  // bytes -> f64 index

        const amplitudesF32 = new Float32Array(valueCount);
        for (let i = 0; i < valueCount; i++) {
            amplitudesF32[i] = HEAPF64[heapOffset + i];
        }

        const amplitudesBytes = amplitudesF32.byteLength;
        const src = device.createBuffer({
            size: amplitudesBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const dst = device.createBuffer({
            size: amplitudesBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const readback = device.createBuffer({
            size: amplitudesBytes,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const params = new Uint32Array([qubit_index >>> 0, n, 0, 0]);
        const paramsBuf = device.createBuffer({
            size: params.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        device.queue.writeBuffer(src, 0, amplitudesF32.buffer, amplitudesF32.byteOffset, amplitudesF32.byteLength);
        device.queue.writeBuffer(paramsBuf, 0, params.buffer, params.byteOffset, params.byteLength);

        const bindGroup = device.createBindGroup({
            layout: state.pauliXPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: src } },
                { binding: 1, resource: { buffer: dst } },
                { binding: 2, resource: { buffer: paramsBuf } },
            ],
        });

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(state.pauliXPipeline);
        pass.setBindGroup(0, bindGroup);
        const pairs = n >>> 1;
        const workgroups = Math.max(1, Math.ceil(pairs / state.workgroupSize));
        pass.dispatchWorkgroups(workgroups);
        pass.end();
        encoder.copyBufferToBuffer(dst, 0, readback, 0, amplitudesBytes);
        device.queue.submit([encoder.finish()]);

        await readback.mapAsync(GPUMapMode.READ);
        const mapped = readback.getMappedRange();
        const resultF32 = new Float32Array(mapped.slice(0));
        readback.unmap();

        for (let i = 0; i < valueCount; i++) {
            HEAPF64[heapOffset + i] = resultF32[i];
        }

        if (typeof src.destroy === 'function') src.destroy();
        if (typeof dst.destroy === 'function') dst.destroy();
        if (typeof readback.destroy === 'function') readback.destroy();
        if (typeof paramsBuf.destroy === 'function') paramsBuf.destroy();
        return 1;
    } catch (err) {
        return 0;
    }
});

EM_ASYNC_JS(int, moonlab_webgpu_pauli_z_dispatch_async,
            (uintptr_t amplitudes_ptr, uint32_t qubit_index, uint32_t state_dim), {
    try {
        const initialized = await moonlab_webgpu_init_async();
        if (!initialized) {
            return 0;
        }

        const state = Module.__moonlabWebGPU;
        const device = state.device;
        const n = state_dim >>> 0;
        const valueCount = n * 2;
        const heapOffset = amplitudes_ptr >>> 3;  // bytes -> f64 index

        const amplitudesF32 = new Float32Array(valueCount);
        for (let i = 0; i < valueCount; i++) {
            amplitudesF32[i] = HEAPF64[heapOffset + i];
        }

        const amplitudesBytes = amplitudesF32.byteLength;
        const src = device.createBuffer({
            size: amplitudesBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const dst = device.createBuffer({
            size: amplitudesBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const readback = device.createBuffer({
            size: amplitudesBytes,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const params = new Uint32Array([qubit_index >>> 0, n, 0, 0]);
        const paramsBuf = device.createBuffer({
            size: params.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        device.queue.writeBuffer(src, 0, amplitudesF32.buffer, amplitudesF32.byteOffset, amplitudesF32.byteLength);
        device.queue.writeBuffer(paramsBuf, 0, params.buffer, params.byteOffset, params.byteLength);

        const bindGroup = device.createBindGroup({
            layout: state.pauliZPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: src } },
                { binding: 1, resource: { buffer: dst } },
                { binding: 2, resource: { buffer: paramsBuf } },
            ],
        });

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(state.pauliZPipeline);
        pass.setBindGroup(0, bindGroup);
        const workgroups = Math.max(1, Math.ceil(n / state.workgroupSize));
        pass.dispatchWorkgroups(workgroups);
        pass.end();
        encoder.copyBufferToBuffer(dst, 0, readback, 0, amplitudesBytes);
        device.queue.submit([encoder.finish()]);

        await readback.mapAsync(GPUMapMode.READ);
        const mapped = readback.getMappedRange();
        const resultF32 = new Float32Array(mapped.slice(0));
        readback.unmap();

        for (let i = 0; i < valueCount; i++) {
            HEAPF64[heapOffset + i] = resultF32[i];
        }

        if (typeof src.destroy === 'function') src.destroy();
        if (typeof dst.destroy === 'function') dst.destroy();
        if (typeof readback.destroy === 'function') readback.destroy();
        if (typeof paramsBuf.destroy === 'function') paramsBuf.destroy();
        return 1;
    } catch (err) {
        return 0;
    }
});

EM_ASYNC_JS(int, moonlab_webgpu_cnot_dispatch_async,
            (uintptr_t amplitudes_ptr, uint32_t control, uint32_t target, uint32_t state_dim), {
    try {
        const initialized = await moonlab_webgpu_init_async();
        if (!initialized) {
            return 0;
        }

        const state = Module.__moonlabWebGPU;
        const device = state.device;
        const n = state_dim >>> 0;
        const valueCount = n * 2;
        const heapOffset = amplitudes_ptr >>> 3;  // bytes -> f64 index

        const amplitudesF32 = new Float32Array(valueCount);
        for (let i = 0; i < valueCount; i++) {
            amplitudesF32[i] = HEAPF64[heapOffset + i];
        }

        const amplitudesBytes = amplitudesF32.byteLength;
        const src = device.createBuffer({
            size: amplitudesBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const dst = device.createBuffer({
            size: amplitudesBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const readback = device.createBuffer({
            size: amplitudesBytes,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const params = new Uint32Array([control >>> 0, target >>> 0, n, 0]);
        const paramsBuf = device.createBuffer({
            size: params.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        device.queue.writeBuffer(src, 0, amplitudesF32.buffer, amplitudesF32.byteOffset, amplitudesF32.byteLength);
        device.queue.writeBuffer(paramsBuf, 0, params.buffer, params.byteOffset, params.byteLength);

        const bindGroup = device.createBindGroup({
            layout: state.cnotPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: src } },
                { binding: 1, resource: { buffer: dst } },
                { binding: 2, resource: { buffer: paramsBuf } },
            ],
        });

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(state.cnotPipeline);
        pass.setBindGroup(0, bindGroup);
        const pairs = n >>> 1;
        const workgroups = Math.max(1, Math.ceil(pairs / state.workgroupSize));
        pass.dispatchWorkgroups(workgroups);
        pass.end();
        encoder.copyBufferToBuffer(dst, 0, readback, 0, amplitudesBytes);
        device.queue.submit([encoder.finish()]);

        await readback.mapAsync(GPUMapMode.READ);
        const mapped = readback.getMappedRange();
        const resultF32 = new Float32Array(mapped.slice(0));
        readback.unmap();

        for (let i = 0; i < valueCount; i++) {
            HEAPF64[heapOffset + i] = resultF32[i];
        }

        if (typeof src.destroy === 'function') src.destroy();
        if (typeof dst.destroy === 'function') dst.destroy();
        if (typeof readback.destroy === 'function') readback.destroy();
        if (typeof paramsBuf.destroy === 'function') paramsBuf.destroy();
        return 1;
    } catch (err) {
        return 0;
    }
});

EM_ASYNC_JS(int, moonlab_webgpu_probabilities_dispatch_async,
            (uintptr_t amplitudes_ptr, uintptr_t probabilities_ptr, uint32_t state_dim), {
    try {
        const initialized = await moonlab_webgpu_init_async();
        if (!initialized) {
            return 0;
        }

        const state = Module.__moonlabWebGPU;
        const device = state.device;
        const n = state_dim >>> 0;
        const amplitudesCount = n * 2;
        const amplitudesHeapOffset = amplitudes_ptr >>> 3;   // bytes -> f64 index
        const probabilitiesHeapOffset = probabilities_ptr >>> 3;

        const amplitudesF32 = new Float32Array(amplitudesCount);
        for (let i = 0; i < amplitudesCount; i++) {
            amplitudesF32[i] = HEAPF64[amplitudesHeapOffset + i];
        }

        const amplitudesBytes = amplitudesF32.byteLength;
        const probabilitiesBytes = n * 4;
        const src = device.createBuffer({
            size: amplitudesBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const dst = device.createBuffer({
            size: probabilitiesBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const readback = device.createBuffer({
            size: probabilitiesBytes,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const params = new Uint32Array([n, 0, 0, 0]);
        const paramsBuf = device.createBuffer({
            size: params.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        device.queue.writeBuffer(src, 0, amplitudesF32.buffer, amplitudesF32.byteOffset, amplitudesF32.byteLength);
        device.queue.writeBuffer(paramsBuf, 0, params.buffer, params.byteOffset, params.byteLength);

        const bindGroup = device.createBindGroup({
            layout: state.probabilitiesPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: src } },
                { binding: 1, resource: { buffer: dst } },
                { binding: 2, resource: { buffer: paramsBuf } },
            ],
        });

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(state.probabilitiesPipeline);
        pass.setBindGroup(0, bindGroup);
        const workgroups = Math.max(1, Math.ceil(n / state.workgroupSize));
        pass.dispatchWorkgroups(workgroups);
        pass.end();
        encoder.copyBufferToBuffer(dst, 0, readback, 0, probabilitiesBytes);
        device.queue.submit([encoder.finish()]);

        await readback.mapAsync(GPUMapMode.READ);
        const mapped = readback.getMappedRange();
        const resultF32 = new Float32Array(mapped.slice(0));
        readback.unmap();

        for (let i = 0; i < n; i++) {
            HEAPF64[probabilitiesHeapOffset + i] = resultF32[i];
        }

        if (typeof src.destroy === 'function') src.destroy();
        if (typeof dst.destroy === 'function') dst.destroy();
        if (typeof readback.destroy === 'function') readback.destroy();
        if (typeof paramsBuf.destroy === 'function') paramsBuf.destroy();
        return 1;
    } catch (err) {
        return 0;
    }
});

EM_ASYNC_JS(int, moonlab_webgpu_mps_apply_gate_theta_dispatch_async,
            (uintptr_t theta_ptr, uintptr_t gate_ptr, uint32_t chi_l, uint32_t chi_r), {
    try {
        const initialized = await moonlab_webgpu_init_async();
        if (!initialized) {
            return 0;
        }

        const state = Module.__moonlabWebGPU;
        const device = state.device;
        const pairs = (chi_l >>> 0) * (chi_r >>> 0);
        const thetaComplexCount = pairs * 4;
        const thetaValueCount = thetaComplexCount * 2;
        const thetaHeapOffset = theta_ptr >>> 3;
        const gateHeapOffset = gate_ptr >>> 3;

        const thetaF32 = new Float32Array(thetaValueCount);
        for (let i = 0; i < thetaValueCount; i++) {
            thetaF32[i] = HEAPF64[thetaHeapOffset + i];
        }

        const gateF32 = new Float32Array(32);
        for (let i = 0; i < 32; i++) {
            gateF32[i] = HEAPF64[gateHeapOffset + i];
        }

        const thetaBytes = thetaF32.byteLength;
        const src = device.createBuffer({
            size: thetaBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const dst = device.createBuffer({
            size: thetaBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const readback = device.createBuffer({
            size: thetaBytes,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const gateBuf = device.createBuffer({
            size: gateF32.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const params = new Uint32Array([chi_l >>> 0, chi_r >>> 0, 0, 0]);
        const paramsBuf = device.createBuffer({
            size: params.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        device.queue.writeBuffer(src, 0, thetaF32.buffer, thetaF32.byteOffset, thetaF32.byteLength);
        device.queue.writeBuffer(gateBuf, 0, gateF32.buffer, gateF32.byteOffset, gateF32.byteLength);
        device.queue.writeBuffer(paramsBuf, 0, params.buffer, params.byteOffset, params.byteLength);

        const bindGroup = device.createBindGroup({
            layout: state.mpsApplyGateThetaPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: src } },
                { binding: 1, resource: { buffer: dst } },
                { binding: 2, resource: { buffer: gateBuf } },
                { binding: 3, resource: { buffer: paramsBuf } },
            ],
        });

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(state.mpsApplyGateThetaPipeline);
        pass.setBindGroup(0, bindGroup);
        const workgroups = Math.max(1, Math.ceil(pairs / state.workgroupSize));
        pass.dispatchWorkgroups(workgroups);
        pass.end();
        encoder.copyBufferToBuffer(dst, 0, readback, 0, thetaBytes);
        device.queue.submit([encoder.finish()]);

        await readback.mapAsync(GPUMapMode.READ);
        const mapped = readback.getMappedRange();
        const resultF32 = new Float32Array(mapped.slice(0));
        readback.unmap();

        for (let i = 0; i < thetaValueCount; i++) {
            HEAPF64[thetaHeapOffset + i] = resultF32[i];
        }

        if (typeof src.destroy === 'function') src.destroy();
        if (typeof dst.destroy === 'function') dst.destroy();
        if (typeof readback.destroy === 'function') readback.destroy();
        if (typeof gateBuf.destroy === 'function') gateBuf.destroy();
        if (typeof paramsBuf.destroy === 'function') paramsBuf.destroy();
        return 1;
    } catch (err) {
        return 0;
    }
});

EM_ASYNC_JS(int, moonlab_webgpu_mps_expectation_z_canonical_dispatch_async,
            (uintptr_t tensor_ptr, uint32_t chi_l, uint32_t chi_r, uintptr_t expectation_out_ptr), {
    try {
        const initialized = await moonlab_webgpu_init_async();
        if (!initialized) {
            return 0;
        }

        const state = Module.__moonlabWebGPU;
        const device = state.device;
        const pairs = (chi_l >>> 0) * (chi_r >>> 0);
        const tensorComplexCount = pairs * 2;
        const tensorValueCount = tensorComplexCount * 2;
        const tensorHeapOffset = tensor_ptr >>> 3;
        const outHeapOffset = expectation_out_ptr >>> 3;

        const tensorF32 = new Float32Array(tensorValueCount);
        for (let i = 0; i < tensorValueCount; i++) {
            tensorF32[i] = HEAPF64[tensorHeapOffset + i];
        }

        const tensorBytes = tensorF32.byteLength;
        const pairBytes = pairs * 8;
        const src = device.createBuffer({
            size: tensorBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const dst = device.createBuffer({
            size: pairBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const readback = device.createBuffer({
            size: pairBytes,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const params = new Uint32Array([chi_l >>> 0, chi_r >>> 0, 0, 0]);
        const paramsBuf = device.createBuffer({
            size: params.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        device.queue.writeBuffer(src, 0, tensorF32.buffer, tensorF32.byteOffset, tensorF32.byteLength);
        device.queue.writeBuffer(paramsBuf, 0, params.buffer, params.byteOffset, params.byteLength);

        const bindGroup = device.createBindGroup({
            layout: state.mpsExpectationZCanonicalPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: src } },
                { binding: 1, resource: { buffer: dst } },
                { binding: 2, resource: { buffer: paramsBuf } },
            ],
        });

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(state.mpsExpectationZCanonicalPipeline);
        pass.setBindGroup(0, bindGroup);
        const workgroups = Math.max(1, Math.ceil(pairs / state.workgroupSize));
        pass.dispatchWorkgroups(workgroups);
        pass.end();
        encoder.copyBufferToBuffer(dst, 0, readback, 0, pairBytes);
        device.queue.submit([encoder.finish()]);

        await readback.mapAsync(GPUMapMode.READ);
        const mapped = readback.getMappedRange();
        const pairProbs = new Float32Array(mapped.slice(0));
        readback.unmap();

        let numerator = 0.0;
        let denominator = 0.0;
        for (let i = 0; i < pairs; i++) {
            const p0 = pairProbs[i * 2];
            const p1 = pairProbs[i * 2 + 1];
            numerator += p0 - p1;
            denominator += p0 + p1;
        }
        HEAPF64[outHeapOffset] = denominator > 1e-30 ? (numerator / denominator) : 0.0;

        if (typeof src.destroy === 'function') src.destroy();
        if (typeof dst.destroy === 'function') dst.destroy();
        if (typeof readback.destroy === 'function') readback.destroy();
        if (typeof paramsBuf.destroy === 'function') paramsBuf.destroy();
        return 1;
    } catch (err) {
        return 0;
    }
});

static double webgpu_now_seconds(void) {
    return emscripten_get_now() / 1000.0;
}

#else

static int moonlab_webgpu_runtime_available(void) {
    return 0;
}

static int moonlab_webgpu_native_dispatch_supported(void) {
    return 0;
}

static int moonlab_webgpu_tn_native_dispatch_supported(void) {
    return 0;
}

static double webgpu_now_seconds(void) {
    return 0.0;
}

static int moonlab_webgpu_init_async(void) {
    return 0;
}

static int moonlab_webgpu_hadamard_dispatch_async(uintptr_t amplitudes_ptr,
                                                  uint32_t qubit_index,
                                                  uint32_t state_dim) {
    (void)amplitudes_ptr;
    (void)qubit_index;
    (void)state_dim;
    return 0;
}

static int moonlab_webgpu_pauli_x_dispatch_async(uintptr_t amplitudes_ptr,
                                                 uint32_t qubit_index,
                                                 uint32_t state_dim) {
    (void)amplitudes_ptr;
    (void)qubit_index;
    (void)state_dim;
    return 0;
}

static int moonlab_webgpu_pauli_z_dispatch_async(uintptr_t amplitudes_ptr,
                                                 uint32_t qubit_index,
                                                 uint32_t state_dim) {
    (void)amplitudes_ptr;
    (void)qubit_index;
    (void)state_dim;
    return 0;
}

static int moonlab_webgpu_cnot_dispatch_async(uintptr_t amplitudes_ptr,
                                              uint32_t control,
                                              uint32_t target,
                                              uint32_t state_dim) {
    (void)amplitudes_ptr;
    (void)control;
    (void)target;
    (void)state_dim;
    return 0;
}

static int moonlab_webgpu_probabilities_dispatch_async(uintptr_t amplitudes_ptr,
                                                       uintptr_t probabilities_ptr,
                                                       uint32_t state_dim) {
    (void)amplitudes_ptr;
    (void)probabilities_ptr;
    (void)state_dim;
    return 0;
}

static int moonlab_webgpu_mps_apply_gate_theta_dispatch_async(uintptr_t theta_ptr,
                                                              uintptr_t gate_ptr,
                                                              uint32_t chi_l,
                                                              uint32_t chi_r) {
    (void)theta_ptr;
    (void)gate_ptr;
    (void)chi_l;
    (void)chi_r;
    return 0;
}

static int moonlab_webgpu_mps_expectation_z_canonical_dispatch_async(uintptr_t tensor_ptr,
                                                                     uint32_t chi_l,
                                                                     uint32_t chi_r,
                                                                     uintptr_t expectation_out_ptr) {
    (void)tensor_ptr;
    (void)chi_l;
    (void)chi_r;
    (void)expectation_out_ptr;
    return 0;
}

#endif

struct webgpu_compute_ctx {
    int available;
    int native_webgpu_ready;
    int perf_monitoring;
    double last_exec_time;
    char device_name[128];
    char last_error[256];
};

struct webgpu_buffer {
    webgpu_compute_ctx_t* ctx;
    void* host_ptr;
    size_t size;
};

static int set_error(webgpu_compute_ctx_t* ctx, const char* msg) {
    if (ctx && msg) {
        snprintf(ctx->last_error, sizeof(ctx->last_error), "%s", msg);
    }
    return -1;
}

static int validate_state_buffer(webgpu_compute_ctx_t* ctx,
                                 webgpu_buffer_t* amplitudes,
                                 uint64_t state_dim) {
    if (!ctx || !amplitudes || !amplitudes->host_ptr || state_dim == 0) {
        return set_error(ctx, "Invalid WebGPU state buffer");
    }
    if (amplitudes->size < state_dim * sizeof(double complex)) {
        return set_error(ctx, "State buffer too small");
    }
    return 0;
}

static void mark_exec_time(webgpu_compute_ctx_t* ctx, double start_time) {
    if (!ctx) return;
    ctx->last_exec_time = webgpu_now_seconds() - start_time;
}

static void mark_native_ready(webgpu_compute_ctx_t* ctx) {
    if (!ctx) return;
    ctx->native_webgpu_ready = 1;
    if (strcmp(ctx->device_name, "WebGPU (WGSL compute)") != 0) {
        snprintf(ctx->device_name, sizeof(ctx->device_name), "WebGPU (WGSL compute)");
    }
}

static int webgpu_hadamard_cpu(webgpu_compute_ctx_t* ctx,
                               webgpu_buffer_t* amplitudes,
                               uint32_t qubit_index,
                               uint64_t state_dim) {
    const double inv_sqrt2 = 0.70710678118654752440;
    const uint64_t stride = 1ULL << qubit_index;
    const uint64_t pairs = state_dim >> 1;
    double complex* a = (double complex*)amplitudes->host_ptr;

    for (uint64_t idx = 0; idx < pairs; idx++) {
        const uint64_t i0 = (idx / stride) * (2 * stride) + (idx % stride);
        const uint64_t i1 = i0 + stride;

        const double complex v0 = a[i0];
        const double complex v1 = a[i1];
        a[i0] = inv_sqrt2 * (v0 + v1);
        a[i1] = inv_sqrt2 * (v0 - v1);
    }

    (void)ctx;
    return 0;
}

static int webgpu_compute_probabilities_cpu(webgpu_compute_ctx_t* ctx,
                                            webgpu_buffer_t* amplitudes,
                                            webgpu_buffer_t* probabilities,
                                            uint64_t state_dim) {
    double complex* a = (double complex*)amplitudes->host_ptr;
    double* p = (double*)probabilities->host_ptr;

    for (uint64_t i = 0; i < state_dim; i++) {
        const double re = creal(a[i]);
        const double im = cimag(a[i]);
        p[i] = re * re + im * im;
    }

    (void)ctx;
    return 0;
}

int webgpu_is_available(void) {
    return moonlab_webgpu_runtime_available();
}

webgpu_compute_ctx_t* webgpu_compute_init(void) {
    if (!webgpu_is_available()) {
        return NULL;
    }

    webgpu_compute_ctx_t* ctx = (webgpu_compute_ctx_t*)calloc(1, sizeof(webgpu_compute_ctx_t));
    if (!ctx) {
        return NULL;
    }

    ctx->available = 1;
    ctx->native_webgpu_ready = 0;
    ctx->perf_monitoring = 0;
    ctx->last_exec_time = 0.0;
    snprintf(ctx->device_name, sizeof(ctx->device_name), "WebGPU (WASM)");
    ctx->last_error[0] = '\0';
    return ctx;
}

void webgpu_compute_free(webgpu_compute_ctx_t* ctx) {
    if (!ctx) return;
    free(ctx);
}

void webgpu_get_device_info(webgpu_compute_ctx_t* ctx,
                            char* name,
                            uint32_t* max_work_group_size,
                            uint32_t* compute_units) {
    if (name) {
        if (ctx && ctx->device_name[0]) {
            snprintf(name, 256, "%s", ctx->device_name);
        } else {
            snprintf(name, 256, "WebGPU (Unavailable)");
        }
    }
    if (max_work_group_size) *max_work_group_size = 256;
    if (compute_units) *compute_units = 1;
}

const char* webgpu_last_error(const webgpu_compute_ctx_t* ctx) {
    static const char* kNoContext = "WebGPU context is NULL";
    if (!ctx) return kNoContext;
    if (ctx->last_error[0] == '\0') return "";
    return ctx->last_error;
}

webgpu_buffer_t* webgpu_buffer_create(webgpu_compute_ctx_t* ctx, size_t size) {
    if (!ctx || !ctx->available || size == 0) {
        return NULL;
    }

    webgpu_buffer_t* buffer = (webgpu_buffer_t*)calloc(1, sizeof(webgpu_buffer_t));
    if (!buffer) {
        return NULL;
    }

    buffer->host_ptr = calloc(1, size);
    if (!buffer->host_ptr) {
        free(buffer);
        return NULL;
    }

    buffer->ctx = ctx;
    buffer->size = size;
    return buffer;
}

webgpu_buffer_t* webgpu_buffer_create_from_data(webgpu_compute_ctx_t* ctx,
                                                const void* data,
                                                size_t size) {
    if (!data || size == 0) {
        return NULL;
    }

    webgpu_buffer_t* buffer = webgpu_buffer_create(ctx, size);
    if (!buffer) {
        return NULL;
    }

    memcpy(buffer->host_ptr, data, size);
    return buffer;
}

void* webgpu_buffer_contents(webgpu_buffer_t* buffer) {
    if (!buffer) return NULL;
    return buffer->host_ptr;
}

int webgpu_buffer_write(webgpu_compute_ctx_t* ctx,
                        webgpu_buffer_t* buffer,
                        const void* src,
                        size_t size) {
    if (!buffer || !src || !buffer->host_ptr || size > buffer->size) {
        return set_error(ctx, "Invalid WebGPU buffer write");
    }

    memcpy(buffer->host_ptr, src, size);
    return 0;
}

int webgpu_buffer_read(webgpu_compute_ctx_t* ctx,
                       webgpu_buffer_t* buffer,
                       void* dst,
                       size_t size) {
    if (!buffer || !dst || !buffer->host_ptr || size > buffer->size) {
        return set_error(ctx, "Invalid WebGPU buffer read");
    }

    memcpy(dst, buffer->host_ptr, size);
    return 0;
}

void webgpu_buffer_free(webgpu_buffer_t* buffer) {
    if (!buffer) return;
    free(buffer->host_ptr);
    free(buffer);
}

int webgpu_hadamard(webgpu_compute_ctx_t* ctx,
                    webgpu_buffer_t* amplitudes,
                    uint32_t qubit_index,
                    uint64_t state_dim) {
    if (validate_state_buffer(ctx, amplitudes, state_dim) != 0) return -1;
    if (qubit_index >= 63) return set_error(ctx, "Invalid qubit index");

    const double start = webgpu_now_seconds();
    if (moonlab_webgpu_native_dispatch_supported() &&
        state_dim <= UINT32_MAX &&
        qubit_index < 32) {
        const int dispatched = moonlab_webgpu_hadamard_dispatch_async(
            (uintptr_t)amplitudes->host_ptr,
            qubit_index,
            (uint32_t)state_dim
        );
        if (dispatched == 1) {
            mark_native_ready(ctx);
            mark_exec_time(ctx, start);
            return 0;
        }
    }

    webgpu_hadamard_cpu(ctx, amplitudes, qubit_index, state_dim);
    mark_exec_time(ctx, start);
    return 0;
}

int webgpu_hadamard_all(webgpu_compute_ctx_t* ctx,
                        webgpu_buffer_t* amplitudes,
                        uint32_t num_qubits,
                        uint64_t state_dim) {
    if (validate_state_buffer(ctx, amplitudes, state_dim) != 0) return -1;

    const double start = webgpu_now_seconds();
    for (uint32_t q = 0; q < num_qubits; q++) {
        if (webgpu_hadamard(ctx, amplitudes, q, state_dim) != 0) {
            return -1;
        }
    }
    mark_exec_time(ctx, start);
    return 0;
}

int webgpu_pauli_x(webgpu_compute_ctx_t* ctx,
                   webgpu_buffer_t* amplitudes,
                   uint32_t qubit_index,
                   uint64_t state_dim) {
    if (validate_state_buffer(ctx, amplitudes, state_dim) != 0) return -1;
    if (qubit_index >= 63) return set_error(ctx, "Invalid qubit index");

    const double start = webgpu_now_seconds();
    if (moonlab_webgpu_native_dispatch_supported() &&
        state_dim <= UINT32_MAX &&
        qubit_index < 32) {
        const int dispatched = moonlab_webgpu_pauli_x_dispatch_async(
            (uintptr_t)amplitudes->host_ptr,
            qubit_index,
            (uint32_t)state_dim
        );
        if (dispatched == 1) {
            mark_native_ready(ctx);
            mark_exec_time(ctx, start);
            return 0;
        }
    }

    const uint64_t stride = 1ULL << qubit_index;
    const uint64_t pairs = state_dim >> 1;
    double complex* a = (double complex*)amplitudes->host_ptr;

    for (uint64_t idx = 0; idx < pairs; idx++) {
        const uint64_t i0 = (idx / stride) * (2 * stride) + (idx % stride);
        const uint64_t i1 = i0 + stride;
        const double complex tmp = a[i0];
        a[i0] = a[i1];
        a[i1] = tmp;
    }

    mark_exec_time(ctx, start);
    return 0;
}

int webgpu_pauli_z(webgpu_compute_ctx_t* ctx,
                   webgpu_buffer_t* amplitudes,
                   uint32_t qubit_index,
                   uint64_t state_dim) {
    if (validate_state_buffer(ctx, amplitudes, state_dim) != 0) return -1;
    if (qubit_index >= 63) return set_error(ctx, "Invalid qubit index");

    const double start = webgpu_now_seconds();
    if (moonlab_webgpu_native_dispatch_supported() &&
        state_dim <= UINT32_MAX &&
        qubit_index < 32) {
        const int dispatched = moonlab_webgpu_pauli_z_dispatch_async(
            (uintptr_t)amplitudes->host_ptr,
            qubit_index,
            (uint32_t)state_dim
        );
        if (dispatched == 1) {
            mark_native_ready(ctx);
            mark_exec_time(ctx, start);
            return 0;
        }
    }

    const uint64_t mask = 1ULL << qubit_index;
    double complex* a = (double complex*)amplitudes->host_ptr;

    for (uint64_t i = 0; i < state_dim; i++) {
        if (i & mask) {
            a[i] = -a[i];
        }
    }

    mark_exec_time(ctx, start);
    return 0;
}

int webgpu_phase(webgpu_compute_ctx_t* ctx,
                 webgpu_buffer_t* amplitudes,
                 uint32_t qubit_index,
                 double phase,
                 uint64_t state_dim) {
    if (validate_state_buffer(ctx, amplitudes, state_dim) != 0) return -1;
    if (qubit_index >= 63) return set_error(ctx, "Invalid qubit index");

    const double start = webgpu_now_seconds();
    const uint64_t mask = 1ULL << qubit_index;
    const double complex phase_factor = cos(phase) + I * sin(phase);
    double complex* a = (double complex*)amplitudes->host_ptr;

    for (uint64_t i = 0; i < state_dim; i++) {
        if (i & mask) {
            a[i] *= phase_factor;
        }
    }

    mark_exec_time(ctx, start);
    return 0;
}

int webgpu_cnot(webgpu_compute_ctx_t* ctx,
                webgpu_buffer_t* amplitudes,
                uint32_t control,
                uint32_t target,
                uint64_t state_dim) {
    if (validate_state_buffer(ctx, amplitudes, state_dim) != 0) return -1;
    if (control >= 63 || target >= 63 || control == target) {
        return set_error(ctx, "Invalid control/target qubits");
    }

    const double start = webgpu_now_seconds();
    if (moonlab_webgpu_native_dispatch_supported() &&
        state_dim <= UINT32_MAX &&
        control < 32 &&
        target < 32) {
        const int dispatched = moonlab_webgpu_cnot_dispatch_async(
            (uintptr_t)amplitudes->host_ptr,
            control,
            target,
            (uint32_t)state_dim
        );
        if (dispatched == 1) {
            mark_native_ready(ctx);
            mark_exec_time(ctx, start);
            return 0;
        }
    }

    const uint64_t target_stride = 1ULL << target;
    const uint64_t control_mask = 1ULL << control;
    const uint64_t pairs = state_dim >> 1;
    double complex* a = (double complex*)amplitudes->host_ptr;

    for (uint64_t idx = 0; idx < pairs; idx++) {
        const uint64_t i0 = (idx / target_stride) * (2 * target_stride) + (idx % target_stride);
        const uint64_t i1 = i0 + target_stride;
        if (i0 & control_mask) {
            const double complex tmp = a[i0];
            a[i0] = a[i1];
            a[i1] = tmp;
        }
    }

    mark_exec_time(ctx, start);
    return 0;
}

int webgpu_mps_apply_gate_theta(webgpu_compute_ctx_t* ctx,
                                void* theta_host_ptr,
                                const double complex* gate_4x4,
                                uint32_t chi_l,
                                uint32_t chi_r,
                                int* used_native_out) {
    if (!ctx || !theta_host_ptr || !gate_4x4 || chi_l == 0 || chi_r == 0) {
        return set_error(ctx, "Invalid MPS gate-theta parameters");
    }
    if (used_native_out) {
        *used_native_out = 0;
    }

    const uint64_t pairs = (uint64_t)chi_l * (uint64_t)chi_r;
    const double start = webgpu_now_seconds();

    if (moonlab_webgpu_tn_native_dispatch_supported()) {
        const int dispatched = moonlab_webgpu_mps_apply_gate_theta_dispatch_async(
            (uintptr_t)theta_host_ptr,
            (uintptr_t)gate_4x4,
            chi_l,
            chi_r
        );
        if (dispatched == 1) {
            mark_native_ready(ctx);
            if (used_native_out) {
                *used_native_out = 1;
            }
            mark_exec_time(ctx, start);
            return 0;
        }
    }

    double complex* theta = (double complex*)theta_host_ptr;
    for (uint64_t pair = 0; pair < pairs; pair++) {
        const uint64_t base = pair * 4;
        const double complex t0 = theta[base + 0];
        const double complex t1 = theta[base + 1];
        const double complex t2 = theta[base + 2];
        const double complex t3 = theta[base + 3];

        theta[base + 0] = gate_4x4[0] * t0 + gate_4x4[1] * t1 + gate_4x4[2] * t2 + gate_4x4[3] * t3;
        theta[base + 1] = gate_4x4[4] * t0 + gate_4x4[5] * t1 + gate_4x4[6] * t2 + gate_4x4[7] * t3;
        theta[base + 2] = gate_4x4[8] * t0 + gate_4x4[9] * t1 + gate_4x4[10] * t2 + gate_4x4[11] * t3;
        theta[base + 3] = gate_4x4[12] * t0 + gate_4x4[13] * t1 + gate_4x4[14] * t2 + gate_4x4[15] * t3;
    }

    mark_exec_time(ctx, start);
    return 0;
}

int webgpu_mps_expectation_z_canonical(webgpu_compute_ctx_t* ctx,
                                       const void* tensor_host_ptr,
                                       uint32_t chi_l,
                                       uint32_t chi_r,
                                       double* expectation_out,
                                       int* used_native_out) {
    if (!ctx || !tensor_host_ptr || !expectation_out || chi_l == 0 || chi_r == 0) {
        return set_error(ctx, "Invalid MPS canonical expectation parameters");
    }
    if (used_native_out) {
        *used_native_out = 0;
    }

    const uint64_t pairs = (uint64_t)chi_l * (uint64_t)chi_r;
    const double start = webgpu_now_seconds();

    if (moonlab_webgpu_tn_native_dispatch_supported()) {
        const int dispatched = moonlab_webgpu_mps_expectation_z_canonical_dispatch_async(
            (uintptr_t)tensor_host_ptr,
            chi_l,
            chi_r,
            (uintptr_t)expectation_out
        );
        if (dispatched == 1) {
            mark_native_ready(ctx);
            if (used_native_out) {
                *used_native_out = 1;
            }
            mark_exec_time(ctx, start);
            return 0;
        }
    }

    const double complex* tensor = (const double complex*)tensor_host_ptr;
    double numerator = 0.0;
    double denominator = 0.0;
    for (uint64_t pair = 0; pair < pairs; pair++) {
        const uint64_t base = pair * 2;
        const double complex a0 = tensor[base + 0];
        const double complex a1 = tensor[base + 1];
        const double p0 = creal(a0 * conj(a0));
        const double p1 = creal(a1 * conj(a1));
        numerator += p0 - p1;
        denominator += p0 + p1;
    }

    *expectation_out = (denominator > 1e-30) ? (numerator / denominator) : 0.0;
    mark_exec_time(ctx, start);
    return 0;
}

int webgpu_oracle(webgpu_compute_ctx_t* ctx,
                  webgpu_buffer_t* amplitudes,
                  uint64_t target,
                  uint64_t state_dim) {
    if (validate_state_buffer(ctx, amplitudes, state_dim) != 0) return -1;
    if (target >= state_dim) return set_error(ctx, "Oracle target out of range");

    const double start = webgpu_now_seconds();
    double complex* a = (double complex*)amplitudes->host_ptr;
    a[target] = -a[target];
    mark_exec_time(ctx, start);
    return 0;
}

int webgpu_oracle_multi(webgpu_compute_ctx_t* ctx,
                        webgpu_buffer_t* amplitudes,
                        const uint64_t* targets,
                        uint32_t num_targets,
                        uint64_t state_dim) {
    if (!targets) return set_error(ctx, "Targets pointer is NULL");
    if (validate_state_buffer(ctx, amplitudes, state_dim) != 0) return -1;

    const double start = webgpu_now_seconds();
    double complex* a = (double complex*)amplitudes->host_ptr;
    for (uint32_t i = 0; i < num_targets; i++) {
        if (targets[i] >= state_dim) {
            return set_error(ctx, "Oracle target out of range");
        }
        a[targets[i]] = -a[targets[i]];
    }
    mark_exec_time(ctx, start);
    return 0;
}

int webgpu_grover_diffusion(webgpu_compute_ctx_t* ctx,
                            webgpu_buffer_t* amplitudes,
                            uint32_t num_qubits,
                            uint64_t state_dim) {
    (void)num_qubits;
    if (validate_state_buffer(ctx, amplitudes, state_dim) != 0) return -1;

    const double start = webgpu_now_seconds();
    double complex* a = (double complex*)amplitudes->host_ptr;
    double complex sum = 0.0;

    for (uint64_t i = 0; i < state_dim; i++) {
        sum += a[i];
    }

    const double complex mean = sum / (double)state_dim;
    for (uint64_t i = 0; i < state_dim; i++) {
        a[i] = 2.0 * mean - a[i];
    }

    mark_exec_time(ctx, start);
    return 0;
}

int webgpu_grover_iteration(webgpu_compute_ctx_t* ctx,
                            webgpu_buffer_t* amplitudes,
                            uint64_t target,
                            uint32_t num_qubits,
                            uint64_t state_dim) {
    const double start = webgpu_now_seconds();
    if (webgpu_oracle(ctx, amplitudes, target, state_dim) != 0) {
        return -1;
    }
    if (webgpu_grover_diffusion(ctx, amplitudes, num_qubits, state_dim) != 0) {
        return -1;
    }
    mark_exec_time(ctx, start);
    return 0;
}

int webgpu_compute_probabilities(webgpu_compute_ctx_t* ctx,
                                 webgpu_buffer_t* amplitudes,
                                 webgpu_buffer_t* probabilities,
                                 uint64_t state_dim) {
    if (validate_state_buffer(ctx, amplitudes, state_dim) != 0) return -1;
    if (!probabilities || !probabilities->host_ptr ||
        probabilities->size < state_dim * sizeof(double)) {
        return set_error(ctx, "Invalid probabilities buffer");
    }

    const double start = webgpu_now_seconds();
    if (moonlab_webgpu_native_dispatch_supported() &&
        state_dim <= UINT32_MAX) {
        const int dispatched = moonlab_webgpu_probabilities_dispatch_async(
            (uintptr_t)amplitudes->host_ptr,
            (uintptr_t)probabilities->host_ptr,
            (uint32_t)state_dim
        );
        if (dispatched == 1) {
            mark_native_ready(ctx);
            mark_exec_time(ctx, start);
            return 0;
        }
    }

    webgpu_compute_probabilities_cpu(ctx, amplitudes, probabilities, state_dim);
    mark_exec_time(ctx, start);
    return 0;
}

int webgpu_normalize(webgpu_compute_ctx_t* ctx,
                     webgpu_buffer_t* amplitudes,
                     double norm,
                     uint64_t state_dim) {
    if (validate_state_buffer(ctx, amplitudes, state_dim) != 0) return -1;
    if (fabs(norm) < 1e-18) {
        return set_error(ctx, "Normalization factor too small");
    }

    const double start = webgpu_now_seconds();
    double complex* a = (double complex*)amplitudes->host_ptr;
    const double inv_norm = 1.0 / norm;

    for (uint64_t i = 0; i < state_dim; i++) {
        a[i] *= inv_norm;
    }

    mark_exec_time(ctx, start);
    return 0;
}

int webgpu_sum_squared_magnitudes(webgpu_compute_ctx_t* ctx,
                                  webgpu_buffer_t* amplitudes,
                                  uint64_t state_dim,
                                  double* result) {
    if (!result) return set_error(ctx, "Result pointer is NULL");
    if (validate_state_buffer(ctx, amplitudes, state_dim) != 0) return -1;

    const double start = webgpu_now_seconds();
    double complex* a = (double complex*)amplitudes->host_ptr;
    double sum = 0.0;

    for (uint64_t i = 0; i < state_dim; i++) {
        const double re = creal(a[i]);
        const double im = cimag(a[i]);
        sum += re * re + im * im;
    }

    *result = sum;
    mark_exec_time(ctx, start);
    return 0;
}

void webgpu_wait_completion(webgpu_compute_ctx_t* ctx) {
    (void)ctx;
}

double webgpu_get_last_execution_time(webgpu_compute_ctx_t* ctx) {
    if (!ctx) return 0.0;
    return ctx->last_exec_time;
}

void webgpu_set_performance_monitoring(webgpu_compute_ctx_t* ctx, int enable) {
    if (!ctx) return;
    ctx->perf_monitoring = enable;
}

int webgpu_native_compute_ready(const webgpu_compute_ctx_t* ctx) {
    if (!ctx) return 0;
    return ctx->native_webgpu_ready != 0;
}
