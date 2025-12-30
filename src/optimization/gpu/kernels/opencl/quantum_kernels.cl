/**
 * @file quantum_kernels.cl
 * @brief OpenCL GPU kernels for quantum gate operations
 *
 * Cross-platform quantum simulation kernels
 * OpenCL 1.2+ compatible
 *
 * Performance targets:
 * - 10-50x speedup over CPU per operation
 * - Optimized memory access patterns
 * - Work-group local memory utilization
 */

// ============================================================================
// COMPLEX NUMBER ARITHMETIC
// ============================================================================

typedef float2 complex_t;  // (real, imaginary)

/**
 * Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
 */
inline complex_t cmul(complex_t a, complex_t b) {
    return (complex_t)(
        a.x * b.x - a.y * b.y,  // real part
        a.x * b.y + a.y * b.x   // imaginary part
    );
}

/**
 * Complex addition
 */
inline complex_t cadd(complex_t a, complex_t b) {
    return a + b;
}

/**
 * Complex subtraction
 */
inline complex_t csub(complex_t a, complex_t b) {
    return a - b;
}

/**
 * Complex magnitude squared: |a+bi|^2 = a^2 + b^2
 */
inline float cabs2(complex_t z) {
    return z.x * z.x + z.y * z.y;
}

/**
 * Scalar multiplication of complex number
 */
inline complex_t cscale(complex_t z, float s) {
    return z * s;
}

/**
 * Complex negation
 */
inline complex_t cneg(complex_t z) {
    return -z;
}

/**
 * Complex conjugate
 */
inline complex_t cconj(complex_t z) {
    return (complex_t)(z.x, -z.y);
}

/**
 * Complex from polar: r * e^(i*theta) = r*(cos(theta) + i*sin(theta))
 */
inline complex_t cpolar(float r, float theta) {
    return (complex_t)(r * cos(theta), r * sin(theta));
}

// ============================================================================
// HADAMARD TRANSFORM - Single Qubit
// ============================================================================

/**
 * Apply Hadamard gate to single qubit
 *
 * H = 1/sqrt(2) * [1   1]
 *                 [1  -1]
 *
 * For qubit i: |psi> -> H_i|psi>
 */
__kernel void hadamard_transform(
    __global complex_t* amplitudes,
    const uint qubit_index,
    const uint state_dim
) {
    uint tid = get_global_id(0);
    uint stride = 1u << qubit_index;
    uint num_pairs = state_dim / 2;

    if (tid >= num_pairs) return;

    // Calculate indices for amplitude pair
    uint mask = stride - 1;
    uint base = (tid >> qubit_index) << (qubit_index + 1);
    uint offset = tid & mask;
    uint idx0 = base + offset;
    uint idx1 = idx0 + stride;

    // Load amplitudes
    complex_t amp0 = amplitudes[idx0];
    complex_t amp1 = amplitudes[idx1];

    // Hadamard transformation
    const float inv_sqrt2 = 0.70710678118f;  // 1/sqrt(2)
    complex_t new_amp0 = cscale(cadd(amp0, amp1), inv_sqrt2);
    complex_t new_amp1 = cscale(csub(amp0, amp1), inv_sqrt2);

    // Write back
    amplitudes[idx0] = new_amp0;
    amplitudes[idx1] = new_amp1;
}

// ============================================================================
// HADAMARD ALL QUBITS
// ============================================================================

/**
 * Apply Hadamard to all qubits simultaneously
 *
 * Creates uniform superposition: |0...0> -> |+>^n
 */
__kernel void hadamard_all_qubits(
    __global complex_t* amplitudes,
    const uint num_qubits,
    const uint state_dim
) {
    uint tid = get_global_id(0);

    if (tid >= state_dim) return;

    complex_t amp = amplitudes[tid];

    // Count number of 1s in binary representation
    uint hamming = popcount(tid);

    // Sign: (-1)^hamming for proper Hadamard
    float sign = (hamming & 1) ? -1.0f : 1.0f;

    // Scale factor: 1/sqrt(2^n)
    float scale = 1.0f / sqrt((float)(1u << num_qubits));

    amplitudes[tid] = cscale(amp, sign * scale);
}

// ============================================================================
// ORACLE - Single Target (Phase Flip)
// ============================================================================

/**
 * Apply oracle for single marked state
 *
 * O|x> = -|x> if x = target, else |x>
 */
__kernel void oracle_single_target(
    __global complex_t* amplitudes,
    const uint target
) {
    uint tid = get_global_id(0);

    if (tid == target) {
        amplitudes[tid] = cneg(amplitudes[tid]);
    }
}

// ============================================================================
// ORACLE - Multiple Targets (Sparse)
// ============================================================================

/**
 * Apply oracle for multiple marked states
 *
 * O|x> = -|x> if x in marked_states, else |x>
 */
__kernel void sparse_oracle(
    __global complex_t* amplitudes,
    __global const uint* marked_states,
    const uint num_marked,
    const uint state_dim
) {
    uint tid = get_global_id(0);

    if (tid >= state_dim) return;

    // Check if current amplitude index is marked
    for (uint i = 0; i < num_marked; i++) {
        if (tid == marked_states[i]) {
            amplitudes[tid] = cneg(amplitudes[tid]);
            break;
        }
    }
}

// ============================================================================
// GROVER DIFFUSION - Phase 1: Compute Sum
// ============================================================================

/**
 * Compute partial sums for diffusion operator
 * Uses work-group reduction
 */
__kernel void diffusion_sum(
    __global const complex_t* amplitudes,
    __global complex_t* partial_sums,
    __local complex_t* local_sums,
    const uint state_dim
) {
    uint tid = get_global_id(0);
    uint lid = get_local_id(0);
    uint group_id = get_group_id(0);
    uint local_size = get_local_size(0);

    // Each thread computes partial sum
    complex_t sum = (complex_t)(0.0f, 0.0f);
    uint grid_size = get_global_size(0);

    for (uint i = tid; i < state_dim; i += grid_size) {
        sum = cadd(sum, amplitudes[i]);
    }

    local_sums[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction within work-group
    for (uint s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            local_sums[lid] = cadd(local_sums[lid], local_sums[lid + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // First thread writes partial sum
    if (lid == 0) {
        partial_sums[group_id] = local_sums[0];
    }
}

// ============================================================================
// GROVER DIFFUSION - Phase 2: Apply Inversion
// ============================================================================

/**
 * Apply inversion about average
 * alpha -> 2*avg - alpha
 */
__kernel void diffusion_apply(
    __global complex_t* amplitudes,
    __global const complex_t* avg_buffer,
    const uint state_dim
) {
    uint tid = get_global_id(0);

    if (tid >= state_dim) return;

    complex_t avg = avg_buffer[0];
    complex_t two_avg = cscale(avg, 2.0f);

    amplitudes[tid] = csub(two_avg, amplitudes[tid]);
}

// ============================================================================
// GROVER DIFFUSION - Fused (for small states)
// ============================================================================

/**
 * Fused Grover diffusion for small state dimensions
 * Fits entirely in local memory
 */
__kernel void grover_diffusion_fused(
    __global complex_t* amplitudes,
    __local complex_t* local_amps,
    const uint num_qubits,
    const uint state_dim
) {
    uint tid = get_global_id(0);
    uint lid = get_local_id(0);
    uint local_size = get_local_size(0);

    // Load to local memory
    if (tid < state_dim) {
        local_amps[lid] = amplitudes[tid];
    } else {
        local_amps[lid] = (complex_t)(0.0f, 0.0f);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute sum using reduction
    __local complex_t shared_sum[256];  // Reduction workspace

    complex_t local_sum = (complex_t)(0.0f, 0.0f);
    for (uint i = lid; i < state_dim; i += local_size) {
        local_sum = cadd(local_sum, local_amps[i]);
    }
    shared_sum[lid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction
    for (uint s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s && lid + s < local_size) {
            shared_sum[lid] = cadd(shared_sum[lid], shared_sum[lid + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Compute average and apply inversion
    complex_t avg = cscale(shared_sum[0], 1.0f / (float)state_dim);
    complex_t two_avg = cscale(avg, 2.0f);

    if (tid < state_dim) {
        amplitudes[tid] = csub(two_avg, local_amps[lid]);
    }
}

// ============================================================================
// COMPUTE PROBABILITIES
// ============================================================================

/**
 * Compute measurement probabilities from amplitudes
 *
 * P(x) = |psi_x|^2 = Re(psi_x)^2 + Im(psi_x)^2
 */
__kernel void compute_probabilities(
    __global const complex_t* amplitudes,
    __global float* probabilities,
    const uint state_dim
) {
    uint tid = get_global_id(0);

    if (tid >= state_dim) return;

    complex_t amp = amplitudes[tid];
    probabilities[tid] = cabs2(amp);
}

// ============================================================================
// NORMALIZE STATE
// ============================================================================

/**
 * Normalize quantum state by given norm
 *
 * |psi> -> |psi> / norm
 */
__kernel void normalize_state(
    __global complex_t* amplitudes,
    const float norm,
    const uint state_dim
) {
    uint tid = get_global_id(0);

    if (tid >= state_dim) return;

    float inv_norm = 1.0f / norm;
    amplitudes[tid] = cscale(amplitudes[tid], inv_norm);
}

// ============================================================================
// PAULI X GATE
// ============================================================================

/**
 * Apply Pauli X gate (bit flip)
 *
 * X = [0 1]
 *     [1 0]
 *
 * Swaps amplitudes: |0> <-> |1>
 */
__kernel void pauli_x(
    __global complex_t* amplitudes,
    const uint qubit_index,
    const uint state_dim
) {
    uint tid = get_global_id(0);
    uint stride = 1u << qubit_index;
    uint num_pairs = state_dim / 2;

    if (tid >= num_pairs) return;

    // Calculate pair indices
    uint mask = stride - 1;
    uint base = (tid >> qubit_index) << (qubit_index + 1);
    uint offset = tid & mask;
    uint idx0 = base + offset;
    uint idx1 = idx0 + stride;

    // Swap amplitudes
    complex_t temp = amplitudes[idx0];
    amplitudes[idx0] = amplitudes[idx1];
    amplitudes[idx1] = temp;
}

// ============================================================================
// PAULI Y GATE
// ============================================================================

/**
 * Apply Pauli Y gate
 *
 * Y = [0 -i]
 *     [i  0]
 */
__kernel void pauli_y(
    __global complex_t* amplitudes,
    const uint qubit_index,
    const uint state_dim
) {
    uint tid = get_global_id(0);
    uint stride = 1u << qubit_index;
    uint num_pairs = state_dim / 2;

    if (tid >= num_pairs) return;

    // Calculate pair indices
    uint mask = stride - 1;
    uint base = (tid >> qubit_index) << (qubit_index + 1);
    uint offset = tid & mask;
    uint idx0 = base + offset;
    uint idx1 = idx0 + stride;

    // Load amplitudes
    complex_t amp0 = amplitudes[idx0];
    complex_t amp1 = amplitudes[idx1];

    // Y gate: |0> -> i|1>, |1> -> -i|0>
    // multiply by i = (0, 1): (a, b) * i = (-b, a)
    // multiply by -i = (0, -1): (a, b) * -i = (b, -a)
    amplitudes[idx0] = (complex_t)(amp1.y, -amp1.x);   // -i * amp1
    amplitudes[idx1] = (complex_t)(-amp0.y, amp0.x);   // i * amp0
}

// ============================================================================
// PAULI Z GATE
// ============================================================================

/**
 * Apply Pauli Z gate (phase flip)
 *
 * Z = [1  0]
 *     [0 -1]
 *
 * Flips phase: |1> -> -|1>
 */
__kernel void pauli_z(
    __global complex_t* amplitudes,
    const uint qubit_index,
    const uint state_dim
) {
    uint tid = get_global_id(0);

    if (tid >= state_dim) return;

    // Check if qubit_index bit is set in tid
    uint mask = 1u << qubit_index;
    if (tid & mask) {
        // Bit is 1, flip phase
        amplitudes[tid] = cneg(amplitudes[tid]);
    }
}

// ============================================================================
// PHASE GATE
// ============================================================================

/**
 * Apply phase rotation gate
 *
 * P(phi) = [1       0    ]
 *          [0  e^(i*phi) ]
 */
__kernel void phase_gate(
    __global complex_t* amplitudes,
    const uint qubit_index,
    const float phase,
    const uint state_dim
) {
    uint tid = get_global_id(0);

    if (tid >= state_dim) return;

    // Check if qubit_index bit is set in tid
    uint mask = 1u << qubit_index;
    if (tid & mask) {
        // Bit is 1, apply phase rotation
        complex_t phase_factor = cpolar(1.0f, phase);
        amplitudes[tid] = cmul(amplitudes[tid], phase_factor);
    }
}

// ============================================================================
// CNOT GATE
// ============================================================================

/**
 * Apply CNOT (controlled-X) gate
 *
 * Flips target if control is |1>
 */
__kernel void cnot_gate(
    __global complex_t* amplitudes,
    const uint control_qubit,
    const uint target_qubit,
    const uint state_dim
) {
    uint tid = get_global_id(0);

    if (tid >= state_dim) return;

    uint control_mask = 1u << control_qubit;
    uint target_mask = 1u << target_qubit;

    // Only swap if control is 1 and we're the lower index
    if ((tid & control_mask) && !(tid & target_mask)) {
        uint partner = tid | target_mask;  // Set target bit

        // Swap amplitudes
        complex_t temp = amplitudes[tid];
        amplitudes[tid] = amplitudes[partner];
        amplitudes[partner] = temp;
    }
}

// ============================================================================
// BATCH GROVER SEARCH
// ============================================================================

/**
 * Execute multiple independent Grover searches in parallel
 * Each work-group handles one complete search
 */
__kernel void grover_batch_search(
    __global complex_t* batch_states,  // All state vectors concatenated
    __global const uint* targets,       // Target for each search
    __global uint* results,             // Output results
    const uint num_searches,
    const uint num_qubits,
    const uint num_iterations
) {
    uint search_id = get_group_id(0);
    uint lid = get_local_id(0);
    uint local_size = get_local_size(0);

    if (search_id >= num_searches) return;

    uint state_dim = 1u << num_qubits;
    uint target = targets[search_id];

    // Pointer to this search's state vector
    __global complex_t* amplitudes = batch_states + (search_id * state_dim);

    // Local memory for reductions
    __local complex_t local_sum[256];

    // Run Grover iterations
    for (uint iter = 0; iter < num_iterations; iter++) {
        // Oracle: flip sign of target
        if (lid == 0) {
            amplitudes[target] = cneg(amplitudes[target]);
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        // Diffusion: compute sum
        complex_t sum = (complex_t)(0.0f, 0.0f);
        for (uint i = lid; i < state_dim; i += local_size) {
            sum = cadd(sum, amplitudes[i]);
        }
        local_sum[lid] = sum;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Reduce sum
        for (uint s = local_size / 2; s > 0; s >>= 1) {
            if (lid < s) {
                local_sum[lid] = cadd(local_sum[lid], local_sum[lid + s]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Apply diffusion
        complex_t avg = cscale(local_sum[0], 1.0f / (float)state_dim);
        complex_t two_avg = cscale(avg, 2.0f);

        for (uint i = lid; i < state_dim; i += local_size) {
            amplitudes[i] = csub(two_avg, amplitudes[i]);
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    // Find maximum probability state
    if (lid == 0) {
        float max_prob = 0.0f;
        uint max_state = 0;

        for (uint i = 0; i < state_dim; i++) {
            float prob = cabs2(amplitudes[i]);
            if (prob > max_prob) {
                max_prob = prob;
                max_state = i;
            }
        }

        results[search_id] = max_state;
    }
}
