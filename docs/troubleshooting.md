# Troubleshooting

Solutions to common issues when using Moonlab.

## Quick Diagnostics

Run the diagnostic tool to identify common issues:

```bash
moonlab-diagnose
```

Or in Python:

```python
from moonlab import diagnose
diagnose()
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
```python
from moonlab import gpu_diagnose
result = gpu_diagnose()
print(result['issue'])
print(result['solution'])
```

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

2. Use CPU for large states:
   ```python
   from moonlab import set_backend
   set_backend('cpu')
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

1. Increase GPU threshold:
   ```python
   from moonlab import configure
   configure(gpu_threshold=20)  # Only use GPU for 20+ qubits
   ```

2. Use automatic selection:
   ```python
   set_backend('auto')
   ```

3. Batch operations to amortize overhead:
   ```python
   with gpu_context() as ctx:
       state = ctx.create_state(24)
       # All operations batched on GPU
   ```

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

2. Enable automatic renormalization:
   ```python
   configure(auto_normalize=True)
   ```

3. Check for numerical precision issues with single precision:
   ```python
   configure(precision='double')  # More accurate
   ```

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

2. Check random seed for reproducibility:
   ```python
   from moonlab import set_seed
   set_seed(42)
   ```

## Performance Issues

### Simulation Too Slow

**Diagnosis:**
```python
from moonlab import Profiler

with Profiler(detailed=True) as p:
    # Your simulation
    pass

p.print_summary()
# Shows where time is spent
```

**Solutions:**

1. **Enable GPU** (for 18+ qubits):
   ```python
   set_backend('metal')
   ```

2. **Enable SIMD**:
   ```python
   configure(simd_level='auto')
   ```

3. **Increase threads**:
   ```python
   configure(num_threads=8)
   ```

4. **Use batching**:
   ```python
   # Instead of individual measurements
   results = state.sample(1000)  # Batch sample
   ```

5. **Consider tensor networks** for low-entanglement states

### Memory Usage Too High

**Diagnosis:**
```python
from moonlab import estimate_memory, MemoryProfiler

print(f"Estimated: {estimate_memory(28):.1f} GB")

with MemoryProfiler() as mp:
    state = QuantumState(28)
print(f"Actual: {mp.peak_mb:.1f} MB")
```

**Solutions:**

1. Use single precision:
   ```python
   configure(precision='single')  # Half memory
   ```

2. Enable memory-efficient mode:
   ```python
   configure(memory_efficient=True)
   ```

3. Use tensor networks:
   ```python
   from moonlab.tensor_network import MPS
   mps = MPS(50, max_bond_dim=100)
   ```

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

Enable verbose logging:

```python
from moonlab import configure
configure(log_level='DEBUG')
```

Or via environment variable:

```bash
export MOONLAB_LOG_LEVEL=DEBUG
python your_script.py
```

## See Also

- [FAQ](faq.md) - Frequently asked questions
- [Installation](installation.md) - Setup instructions
- [Performance Tuning](guides/performance-tuning.md) - Optimization
- [GPU Acceleration](guides/gpu-acceleration.md) - GPU setup

