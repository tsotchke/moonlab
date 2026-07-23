# Troubleshooting

Solutions to common issues when using Moonlab.

## Quick Diagnostics

To verify a working install, run the test gauntlet:

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

A clean install exits with `100% tests passed`.  For the Python and
Rust bindings:

```bash
MOONLAB_LIB_DIR="$(pwd)/build" \
  PYTHONPATH="$(pwd)/bindings/python" \
  python3 -m pytest bindings/python/tests -q

cd bindings/rust/moonlab
MOONLAB_LIB_DIR="$(realpath ../../../build)" cargo test
```

## Installation Issues

### Import Error: Module Not Found

**Symptom:**
```python
>>> import moonlab
ModuleNotFoundError: No module named 'moonlab'
```

**Solutions:**

1. Verify installation:
   ```bash
   pip show moonlab
   ```

2. Check Python version (requires 3.8+):
   ```bash
   python --version
   ```

3. Reinstall in correct environment:
   ```bash
   pip uninstall moonlab
   pip install moonlab
   ```

4. For development install:
   ```bash
   pip install -e ./bindings/python
   ```

### C Library Not Found

**Symptom:**
```
OSError: libquantum_sim.dylib not found
```

**Solutions:**

1. Set library path:
   ```bash
   export DYLD_LIBRARY_PATH=/path/to/moonlab/lib:$DYLD_LIBRARY_PATH
   ```

2. Rebuild with correct RPATH:
   ```bash
   cmake -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON ..
   make install
   ```

### Build Fails: Missing Dependencies

**Symptom:**
```
CMake Error: Could not find OpenBLAS
```

**Solutions:**

macOS:
```bash
brew install openblas
cmake -DBLAS_DIR=$(brew --prefix openblas) ..
```

Ubuntu:
```bash
sudo apt-get install libopenblas-dev
cmake ..
```

## GPU Issues

### GPU Not Detected

**Symptom:**
```python
>>> from moonlab import gpu_available
>>> gpu_available()
False
```

**Diagnosis:**
Probe each backend explicitly via the C API
(`qsim_backend_available`, `src/utils/config.h:438`); the bindings
forward to the same function.  On macOS the canonical check is

```bash
system_profiler SPDisplaysDataType | grep -i metal
```

If Metal is available but Moonlab cannot bind it, force the backend
selection with the `QSIM_BACKEND=gpu_metal` environment variable and
rerun.  A failure trace then identifies whether the issue is build
gating (`-DQSIM_ENABLE_GPU=OFF`), driver, or capability detection.

**Common Causes:**

1. **macOS version too old** (requires 12.0+):
   ```bash
   sw_vers -productVersion
   ```

2. **Running in VM** without GPU passthrough

3. **Built without GPU support**:
   ```bash
   cmake -DENABLE_GPU=ON ..
   make clean && make
   ```

4. **Intel Mac with unsupported GPU**:
   - Check: `system_profiler SPDisplaysDataType | grep Metal`
   - Requires Metal-capable GPU

### GPU Out of Memory

**Symptom:**
```
RuntimeError: GPU memory allocation failed (requested 8.0 GB, available 4.2 GB)
```

**Solutions:**

1. Reduce qubit count:
   ```python
   # Memory doubles per qubit
   state = QuantumState(26)  # Instead of 28
   ```

2. Use CPU for large states by forcing the backend at process start:
   ```bash
   QSIM_BACKEND=cpu python3 your_script.py
   ```

3. Close other GPU applications

4. Use tensor network methods:
   ```python
   from moonlab.tensor_network import MPS
   mps = MPS(50, max_bond_dim=100)  # Much less memory
   ```

### GPU Performance Worse Than CPU

**Symptom:** GPU is slower for small circuits

**Explanation:** GPU has overhead for kernel launch and memory transfer. This is expected for small systems.

**Solutions:**

1. Raise the GPU dispatch threshold for tensor-network kernels:
   ```bash
   # Multiplier on the default GPU-vs-CPU crossover; values > 1 keep
   # work on the CPU for longer.  Read by tn_gates.c at runtime.
   export MOONLAB_TENSOR_GPU_THRESHOLD_MUL=2.0
   ```

2. Use the auto-selecting backend (default):
   ```bash
   QSIM_BACKEND=auto python3 your_script.py
   ```

3. Batch operations into a single kernel pipeline via the gate-fusion
   DAG in `src/optimization/fusion/`; `fuse_circuit_*` + `fuse_compile`
   collapse adjacent gates so the GPU pays one launch instead of many.

## Numerical Issues

### NaN or Infinity in Results

**Symptom:**
```python
>>> state.amplitudes
array([nan+nanj, nan+nanj, ...])
```

**Common Causes:**

1. **Numerical overflow** in rotation angles:
   ```python
   # Bad: angle too large
   state.rz(0, 1e15)

   # Good: normalize angles
   import numpy as np
   angle = 1e15 % (2 * np.pi)
   state.rz(0, angle)
   ```

2. **Division by zero** in normalization:
   ```python
   # After measurement, if all probability concentrated
   # Check state is valid before operations
   if state.norm() < 1e-10:
       print("Warning: near-zero state")
   ```

3. **Corrupted state** from previous error:
   ```python
   # Create fresh state
   state = QuantumState(n_qubits)
   ```

### State Not Normalized

**Symptom:**
```python
>>> sum(state.probabilities)
0.9987  # Should be 1.0
```

**Solutions:**

1. Manual normalization:
   ```python
   state.normalize()
   ```

2. Moonlab always runs in double precision; there is no
   single-precision Python configuration knob.  Numerical drift in
   long simulations is normally a sign of a missing
   ``state.normalize()`` call after a non-unitary operation (noise
   channel, post-selection, imag-time evolution).  See
   ``src/quantum/state.h::quantum_state_normalize``.

### Incorrect Measurement Statistics

**Symptom:** Measured probabilities don't match expected values

**Diagnosis:**
```python
# Compare theoretical vs measured
theoretical = state.probabilities
measured = state.sample_distribution(shots=100000)

for i, (t, m) in enumerate(zip(theoretical, measured)):
    if abs(t - m) > 0.01:
        print(f"Basis {i:04b}: expected {t:.4f}, got {m:.4f}")
```

**Solutions:**

1. Increase shot count for better statistics:
   ```python
   results = state.sample(shots=100000)  # More shots
   ```

2. Pin the random seed for reproducibility (env var read by the
   C config layer at `src/utils/config.c::qsim_config_from_env`):
   ```bash
   QSIM_SEED=42 python3 your_script.py
   ```

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
   ```bash
   QSIM_BACKEND=gpu_metal python3 your_script.py
   ```

2. **Force a specific SIMD level** (auto-detect is the default;
   override only when measuring):
   ```bash
   QSIM_SIMD=avx512   # or neon, sve, avx2, sse2, none
   ```

3. **Set the OpenMP / thread count**:
   ```bash
   QSIM_THREADS=8 python3 your_script.py
   # or
   OMP_NUM_THREADS=8 python3 your_script.py
   ```

4. **Use the gate-fusion DAG** at ``src/optimization/fusion/`` to
   collapse adjacent gates before dispatch.

5. **Consider tensor networks** for low-entanglement states; see
   ``moonlab.tdvp`` and ``moonlab.mpdo``.

### Memory Usage Too High

**Diagnosis:** dense state-vector memory is `16 * 2^N` bytes
(double-precision complex), so estimation is trivial in head: `N=24
-> 256 MB`, `N=28 -> 4 GB`, `N=32 -> 64 GB`.  No `estimate_memory()`
helper is exposed in the Python bindings -- compute directly:

```python
import math
print(f"{16 * 2**28 / 2**30:.1f} GB")
```

**Solutions:**

1. There is no runtime "single precision" toggle in v1.2.0; the C
   state vector is always `double _Complex`.  To halve memory you
   must drop to a tensor-network representation:
   ```python
   from moonlab.tdvp import random_mps, mpo_heisenberg, TdvpEngine
   mps = random_mps(num_sites=50, chi_init=8, max_bond_dim=100)
   ```

2. The TDVP and MPDO engines scale polynomially in bond dimension,
   not exponentially in qubit count.  For a 50-qubit chain with
   moderate entanglement they replace a 16 EB state vector with a
   buffer measured in MB.

## Algorithm-Specific Issues

### VQE Not Converging

**Symptoms:** Energy oscillates or converges to wrong value

**Solutions:**

1. **Adjust learning rate**:
   ```python
   vqe = VQE(hamiltonian, learning_rate=0.01)  # Smaller
   ```

2. **Try different optimizer**:
   ```python
   vqe = VQE(hamiltonian, optimizer='COBYLA')  # Gradient-free
   ```

3. **Increase shots** for better gradient estimates:
   ```python
   vqe = VQE(hamiltonian, shots=10000)
   ```

4. **Check ansatz expressibility**:
   ```python
   vqe = VQE(hamiltonian, ansatz='UCCSD', layers=2)
   ```

5. **Multiple random initializations**:
   ```python
   best_energy = float('inf')
   for seed in range(10):
       result = vqe.optimize(seed=seed)
       if result.energy < best_energy:
           best_energy = result.energy
   ```

### Grover's Search Returns Wrong Answer

**Symptoms:** Measured result isn't the target

**Common Causes:**

1. **Wrong number of iterations**:
   ```python
   import numpy as np
   N = 2 ** n_qubits
   optimal_iter = int(np.pi / 4 * np.sqrt(N))
   ```

2. **Oracle implementation bug**:
   ```python
   # Test oracle marks correct state
   state = QuantumState(n)
   state.set_basis_state(target)
   oracle(state)
   # Phase should be -1
   assert np.isclose(state.amplitudes[target], -1)
   ```

3. **Multiple solutions** (need fewer iterations):
   ```python
   M = count_solutions(oracle)
   optimal_iter = int(np.pi / 4 * np.sqrt(N / M))
   ```

### DMRG Not Converging

**Symptoms:** Energy not decreasing or bond dimension exploding

**Solutions:**

1. **Increase sweeps**:
   ```python
   dmrg = DMRG(max_sweeps=20)
   ```

2. **Gradual bond dimension increase**:
   ```python
   dmrg = DMRG(bond_dim_schedule=[20, 50, 100, 200])
   ```

3. **Check Hamiltonian**:
   ```python
   # Verify Hermiticity
   H_matrix = hamiltonian.to_matrix()
   assert np.allclose(H_matrix, H_matrix.conj().T)
   ```

4. **Try different initial state**:
   ```python
   dmrg = DMRG(initial_state='random')
   ```

## Python Binding Issues

### Segmentation Fault

**Symptom:** Python crashes without error message

**Common Causes:**

1. **State already destroyed**:
   ```python
   # Bad: using destroyed state
   state = QuantumState(10)
   del state
   # state.h(0)  # Would crash!

   # Good: don't manually delete
   ```

2. **Thread safety violation**:
   ```python
   # Bad: sharing state across threads without locks
   # Good: use thread-local states or locks
   ```

3. **Memory corruption**:
   ```bash
   # Debug build
   cmake -DCMAKE_BUILD_TYPE=Debug -DSANITIZE_ADDRESS=ON ..
   make
   python your_script.py
   ```

### Memory Leak

**Diagnosis:**
```python
import tracemalloc
tracemalloc.start()

# Your code
for i in range(1000):
    state = QuantumState(20)
    state.h(0)

current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1e6:.1f} MB, Peak: {peak / 1e6:.1f} MB")
```

**Solutions:**

1. States should be garbage collected automatically
2. If leak persists, explicitly delete:
   ```python
   del state
   import gc
   gc.collect()
   ```

## Getting Help

### Reporting Issues

When reporting bugs, include:

1. **System information**:
   ```python
   from moonlab import system_info
   print(system_info())
   ```

2. **Minimal reproducible example**

3. **Full error traceback**

4. **Expected vs actual behavior**

### Community Resources

- **GitHub Issues**: [github.com/tsotchke/moonlab/issues](https://github.com/tsotchke/moonlab/issues)
- **Discussions**: [github.com/tsotchke/moonlab/discussions](https://github.com/tsotchke/moonlab/discussions)

### Debug Mode

Enable verbose logging via the environment variable parsed by
`qsim_config_from_env`:

```bash
export MOONLAB_LOG_LEVEL=DEBUG
python your_script.py
```

## See Also

- [FAQ](faq.md) - Frequently asked questions
- [Installation](installation.md) - Setup instructions
- [Performance Tuning](guides/performance-tuning.md) - Optimization
- [GPU Acceleration](guides/gpu-acceleration.md) - GPU setup

