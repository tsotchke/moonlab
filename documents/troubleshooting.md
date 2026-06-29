# Archived Moonlab Documentation: Troubleshooting

This local Moonlab document is retained as archived vendor text for the QGTL integration audit; current supported claims are measured by `scripts/moonlab_doc_claim_audit.py` and grounded against `external/moonlab/README.md`, `external/moonlab/CMakeLists.txt`, and `docs/MOONLAB_OPEN_CORE_INTEGRATION.md`.

The historical text below is preserved as an archival snapshot, not as current release documentation.

```text
# Troubleshooting

Solutions to common issues when using Moonlab.

## Quick Diagnostics

To verify a working install, run the test gauntlet:

[archived fence delimiter: ```bash]
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
[archived fence delimiter: ```]

A clean install exits with `100% tests passed`.  For the Python and
Rust bindings:

[archived fence delimiter: ```bash]
MOONLAB_LIB_DIR="$(pwd)/build" \
  PYTHONPATH="$(pwd)/bindings/python" \
  python3 -m pytest bindings/python/tests -q

cd bindings/rust/moonlab
MOONLAB_LIB_DIR="$(realpath ../../../build)" cargo test
[archived fence delimiter: ```]

## Installation Issues

### Import Error: Module Not Found

**Symptom:**
[archived fence delimiter: ```python]
>>> import moonlab
ModuleNotFoundError: No module named 'moonlab'
[archived fence delimiter: ```]

**Solutions:**

1. Verify installation:
[archived fence delimiter:    ```bash]
   pip show moonlab
[archived fence delimiter:    ```]

2. Check Python version (requires 3.8+):
[archived fence delimiter:    ```bash]
   python --version
[archived fence delimiter:    ```]

3. Reinstall in correct environment:
[archived fence delimiter:    ```bash]
   pip uninstall moonlab
   pip install moonlab
[archived fence delimiter:    ```]

4. For development install:
[archived fence delimiter:    ```bash]
   pip install -e ./bindings/python
[archived fence delimiter:    ```]

### C Library Not Found

**Symptom:**
[archived fence delimiter: ```]
OSError: libquantum_sim.dylib not found
[archived fence delimiter: ```]

**Solutions:**

1. Set library path:
[archived fence delimiter:    ```bash]
   export DYLD_LIBRARY_PATH=/path/to/moonlab/lib:$DYLD_LIBRARY_PATH
[archived fence delimiter:    ```]

2. Rebuild with correct RPATH:
[archived fence delimiter:    ```bash]
   cmake -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON ..
   make install
[archived fence delimiter:    ```]

### Build Fails: Missing Dependencies

**Symptom:**
[archived fence delimiter: ```]
CMake Error: Could not find OpenBLAS
[archived fence delimiter: ```]

**Solutions:**

macOS:
[archived fence delimiter: ```bash]
brew install openblas
cmake -DBLAS_DIR=$(brew --prefix openblas) ..
[archived fence delimiter: ```]

Ubuntu:
[archived fence delimiter: ```bash]
sudo apt-get install libopenblas-dev
cmake ..
[archived fence delimiter: ```]

## GPU Issues

### GPU Not Detected

**Symptom:**
[archived fence delimiter: ```python]
>>> from moonlab import gpu_available
>>> gpu_available()
False
[archived fence delimiter: ```]

**Diagnosis:**
Probe each backend explicitly via the C API
(`qsim_backend_available`, `src/utils/config.h:438`); the bindings
forward to the same function.  On macOS the canonical check is

[archived fence delimiter: ```bash]
system_profiler SPDisplaysDataType | grep -i metal
[archived fence delimiter: ```]

If Metal is available but Moonlab cannot bind it, force the backend
selection with the `QSIM_BACKEND=gpu_metal` environment variable and
rerun.  A failure trace then identifies whether the issue is build
gating (`-DQSIM_ENABLE_GPU=OFF`), driver, or capability detection.

**Common Causes:**

1. **macOS version too old** (requires 12.0+):
[archived fence delimiter:    ```bash]
   sw_vers -productVersion
[archived fence delimiter:    ```]

2. **Running in VM** without GPU passthrough

3. **Built without GPU support**:
[archived fence delimiter:    ```bash]
   cmake -DENABLE_GPU=ON ..
   make clean && make
[archived fence delimiter:    ```]

4. **Intel Mac with unsupported GPU**:
   - Check: `system_profiler SPDisplaysDataType | grep Metal`
   - Requires Metal-capable GPU

### GPU Out of Memory

**Symptom:**
[archived fence delimiter: ```]
RuntimeError: GPU memory allocation failed (requested 8.0 GB, available 4.2 GB)
[archived fence delimiter: ```]

**Solutions:**

1. Reduce qubit count:
[archived fence delimiter:    ```python]
   # Memory doubles per qubit
   state = QuantumState(26)  # Instead of 28
[archived fence delimiter:    ```]

2. Use CPU for large states by forcing the backend at process start:
[archived fence delimiter:    ```bash]
   QSIM_BACKEND=cpu python3 your_script.py
[archived fence delimiter:    ```]

3. Close other GPU applications

4. Use tensor network methods:
[archived fence delimiter:    ```python]
   from moonlab.tensor_network import MPS
   mps = MPS(50, max_bond_dim=100)  # Much less memory
[archived fence delimiter:    ```]

### GPU Performance Worse Than CPU

**Symptom:** GPU is slower for small circuits

**Explanation:** GPU has overhead for kernel launch and memory transfer. This is expected for small systems.

**Solutions:**

1. Raise the GPU dispatch threshold for tensor-network kernels:
[archived fence delimiter:    ```bash]
   # Multiplier on the default GPU-vs-CPU crossover; values > 1 keep
   # work on the CPU for longer.  Read by tn_gates.c at runtime.
   export MOONLAB_TENSOR_GPU_THRESHOLD_MUL=2.0
[archived fence delimiter:    ```]

2. Use the auto-selecting backend (default):
[archived fence delimiter:    ```bash]
   QSIM_BACKEND=auto python3 your_script.py
[archived fence delimiter:    ```]

3. Batch operations into a single kernel pipeline via the gate-fusion
   DAG in `src/optimization/fusion/`; `fuse_circuit_*` + `fuse_compile`
   collapse adjacent gates so the GPU pays one launch instead of many.

## Numerical Issues

### NaN or Infinity in Results

**Symptom:**
[archived fence delimiter: ```python]
>>> state.amplitudes
array([nan+nanj, nan+nanj, ...])
[archived fence delimiter: ```]

**Common Causes:**

1. **Numerical overflow** in rotation angles:
[archived fence delimiter:    ```python]
   # Bad: angle too large
   state.rz(0, 1e15)

   # Good: normalize angles
   import numpy as np
   angle = 1e15 % (2 * np.pi)
   state.rz(0, angle)
[archived fence delimiter:    ```]

2. **Division by zero** in normalization:
[archived fence delimiter:    ```python]
   # After measurement, if all probability concentrated
   # Check state is valid before operations
   if state.norm() < 1e-10:
       print("Warning: near-zero state")
[archived fence delimiter:    ```]

3. **Corrupted state** from previous error:
[archived fence delimiter:    ```python]
   # Create fresh state
   state = QuantumState(n_qubits)
[archived fence delimiter:    ```]

### State Not Normalized

**Symptom:**
[archived fence delimiter: ```python]
>>> sum(state.probabilities)
0.9987  # Should be 1.0
[archived fence delimiter: ```]

**Solutions:**

1. Manual normalization:
[archived fence delimiter:    ```python]
   state.normalize()
[archived fence delimiter:    ```]

2. Moonlab always runs in double precision; there is no
   single-precision Python configuration knob.  Numerical drift in
   long simulations is normally a sign of a missing
   ``state.normalize()`` call after a non-unitary operation (noise
   channel, post-selection, imag-time evolution).  See
   ``src/quantum/state.h::quantum_state_normalize``.

### Incorrect Measurement Statistics

**Symptom:** Measured probabilities don't match expected values

**Diagnosis:**
[archived fence delimiter: ```python]
# Compare theoretical vs measured
theoretical = state.probabilities
measured = state.sample_distribution(shots=100000)

for i, (t, m) in enumerate(zip(theoretical, measured)):
    if abs(t - m) > 0.01:
        print(f"Basis {i:04b}: expected {t:.4f}, got {m:.4f}")
[archived fence delimiter: ```]

**Solutions:**

1. Increase shot count for better statistics:
[archived fence delimiter:    ```python]
   results = state.sample(shots=100000)  # More shots
[archived fence delimiter:    ```]

2. Pin the random seed for reproducibility (env var read by the
   C config layer at `src/utils/config.c::qsim_config_from_env`):
[archived fence delimiter:    ```bash]
   QSIM_SEED=42 python3 your_script.py
[archived fence delimiter:    ```]

## Performance Issues

### Simulation Too Slow

**Diagnosis:** the C library ships a per-host throughput harness at
``benchmarks/harness/`` and a per-step profiler in
``tools/profiler/profiler.c``.  The Python bindings do not expose a
``Profiler`` wrapper; instead time individual operations with
``time.perf_counter()`` around them, or run a known benchmark
(``cmake --build build --target bench_state_vector`` etc.) and
compare against the ``benchmarks/results/`` baselines.

**Solutions:**

1. **Pick the GPU backend** (typical break-even is 18+ qubits):
[archived fence delimiter:    ```bash]
   QSIM_BACKEND=gpu_metal python3 your_script.py
[archived fence delimiter:    ```]

2. **Force a specific SIMD level** (auto-detect is the default;
   override only when measuring):
[archived fence delimiter:    ```bash]
   QSIM_SIMD=avx512   # or neon, sve, avx2, sse2, none
[archived fence delimiter:    ```]

3. **Set the OpenMP / thread count**:
[archived fence delimiter:    ```bash]
   QSIM_THREADS=8 python3 your_script.py
   # or
   OMP_NUM_THREADS=8 python3 your_script.py
[archived fence delimiter:    ```]

4. **Use the gate-fusion DAG** at ``src/optimization/fusion/`` to
   collapse adjacent gates before dispatch.

5. **Consider tensor networks** for low-entanglement states; see
   ``moonlab.tdvp`` and ``moonlab.mpdo``.

### Memory Usage Too High

**Diagnosis:** dense state-vector memory is `16 * 2^N` bytes
(double-precision complex), so estimation is trivial in head: `N=24
-> 256 MB`, `N=28 -> 4 GB`, `N=32 -> 64 GB`.  No `estimate_memory()`
helper is exposed in the Python bindings -- compute directly:

[archived fence delimiter: ```python]
import math
print(f"{16 * 2**28 / 2**30:.1f} GB")
[archived fence delimiter: ```]

**Solutions:**

1. There is no runtime "single precision" toggle in v0.4.2; the C
   state vector is always `double _Complex`.  To halve memory you
   must drop to a tensor-network representation:
[archived fence delimiter:    ```python]
   from moonlab.tdvp import random_mps, mpo_heisenberg, TdvpEngine
   mps = random_mps(num_sites=50, chi_init=8, max_bond_dim=100)
[archived fence delimiter:    ```]

2. The TDVP and MPDO engines scale polynomially in bond dimension,
   not exponentially in qubit count.  For a 50-qubit chain with
   moderate entanglement they replace a 16 EB state vector with a
   buffer measured in MB.

## Algorithm-Specific Issues

### VQE Not Converging

**Symptoms:** Energy oscillates or converges to wrong value

**Solutions:**

1. **Adjust learning rate**:
[archived fence delimiter:    ```python]
   vqe = VQE(hamiltonian, learning_rate=0.01)  # Smaller
[archived fence delimiter:    ```]

2. **Try different optimizer**:
[archived fence delimiter:    ```python]
   vqe = VQE(hamiltonian, optimizer='COBYLA')  # Gradient-free
[archived fence delimiter:    ```]

3. **Increase shots** for better gradient estimates:
[archived fence delimiter:    ```python]
   vqe = VQE(hamiltonian, shots=10000)
[archived fence delimiter:    ```]

4. **Check ansatz expressibility**:
[archived fence delimiter:    ```python]
   vqe = VQE(hamiltonian, ansatz='UCCSD', layers=2)
[archived fence delimiter:    ```]

5. **Multiple random initializations**:
[archived fence delimiter:    ```python]
   best_energy = float('inf')
   for seed in range(10):
       result = vqe.optimize(seed=seed)
       if result.energy < best_energy:
           best_energy = result.energy
[archived fence delimiter:    ```]

### Grover's Search Returns Wrong Answer

**Symptoms:** Measured result isn't the target

**Common Causes:**

1. **Wrong number of iterations**:
[archived fence delimiter:    ```python]
   import numpy as np
   N = 2 ** n_qubits
   optimal_iter = int(np.pi / 4 * np.sqrt(N))
[archived fence delimiter:    ```]

2. **Oracle implementation bug**:
[archived fence delimiter:    ```python]
   # Test oracle marks correct state
   state = QuantumState(n)
   state.set_basis_state(target)
   oracle(state)
   # Phase should be -1
   assert np.isclose(state.amplitudes[target], -1)
[archived fence delimiter:    ```]

3. **Multiple solutions** (need fewer iterations):
[archived fence delimiter:    ```python]
   M = count_solutions(oracle)
   optimal_iter = int(np.pi / 4 * np.sqrt(N / M))
[archived fence delimiter:    ```]

### DMRG Not Converging

**Symptoms:** Energy not decreasing or bond dimension exploding

**Solutions:**

1. **Increase sweeps**:
[archived fence delimiter:    ```python]
   dmrg = DMRG(max_sweeps=20)
[archived fence delimiter:    ```]

2. **Gradual bond dimension increase**:
[archived fence delimiter:    ```python]
   dmrg = DMRG(bond_dim_schedule=[20, 50, 100, 200])
[archived fence delimiter:    ```]

3. **Check Hamiltonian**:
[archived fence delimiter:    ```python]
   # Verify Hermiticity
   H_matrix = hamiltonian.to_matrix()
   assert np.allclose(H_matrix, H_matrix.conj().T)
[archived fence delimiter:    ```]

4. **Try different initial state**:
[archived fence delimiter:    ```python]
   dmrg = DMRG(initial_state='random')
[archived fence delimiter:    ```]

## Python Binding Issues

### Segmentation Fault

**Symptom:** Python crashes without error message

**Common Causes:**

1. **State already destroyed**:
[archived fence delimiter:    ```python]
   # Bad: using destroyed state
   state = QuantumState(10)
   del state
   # state.h(0)  # Would crash!

   # Good: don't manually delete
[archived fence delimiter:    ```]

2. **Thread safety violation**:
[archived fence delimiter:    ```python]
   # Bad: sharing state across threads without locks
   # Good: use thread-local states or locks
[archived fence delimiter:    ```]

3. **Memory corruption**:
[archived fence delimiter:    ```bash]
   # Debug build
   cmake -DCMAKE_BUILD_TYPE=Debug -DSANITIZE_ADDRESS=ON ..
   make
   python your_script.py
[archived fence delimiter:    ```]

### Memory Leak

**Diagnosis:**
[archived fence delimiter: ```python]
import tracemalloc
tracemalloc.start()

# Your code
for i in range(1000):
    state = QuantumState(20)
    state.h(0)

current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1e6:.1f} MB, Peak: {peak / 1e6:.1f} MB")
[archived fence delimiter: ```]

**Solutions:**

1. States should be garbage collected automatically
2. If leak persists, explicitly delete:
[archived fence delimiter:    ```python]
   del state
   import gc
   gc.collect()
[archived fence delimiter:    ```]

## Getting Help

### Reporting Issues

When reporting bugs, include:

1. **System information**:
[archived fence delimiter:    ```python]
   from moonlab import system_info
   print(system_info())
[archived fence delimiter:    ```]

2. **Minimal reproducible example**

3. **Full error traceback**

4. **Expected vs actual behavior**

### Community Resources

- **GitHub Issues**: [github.com/tsotchke/moonlab/issues](https://github.com/tsotchke/moonlab/issues)
- **Discussions**: [github.com/tsotchke/moonlab/discussions](https://github.com/tsotchke/moonlab/discussions)

### Debug Mode

Enable verbose logging via the environment variable parsed by
`qsim_config_from_env`:

[archived fence delimiter: ```bash]
export MOONLAB_LOG_LEVEL=DEBUG
python your_script.py
[archived fence delimiter: ```]

## See Also

- [FAQ](faq.md) - Frequently asked questions
- [Installation](installation.md) - Setup instructions
- [Performance Tuning](guides/performance-tuning.md) - Optimization
- [GPU Acceleration](guides/gpu-acceleration.md) - GPU setup

```
