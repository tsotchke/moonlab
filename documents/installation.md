# Installation

Complete installation instructions for Moonlab Quantum Simulator across supported platforms.

## System Requirements

### Minimum Requirements
- 64-bit processor (x86-64 or ARM64)
- 4 GB RAM (8 GB recommended for 20+ qubit simulations)
- C11-compatible compiler (GCC 9+ or Clang 11+)
- Make build system

### Recommended Requirements
- Apple Silicon Mac (M1/M2/M3/M4) for Metal GPU acceleration
- 16+ GB RAM for 25+ qubit simulations
- OpenMP-capable compiler for multi-core parallelization

### Memory Requirements by Qubit Count

| Qubits | State Vector Size | Recommended RAM |
|--------|------------------|-----------------|
| 20     | 16 MB            | 4 GB            |
| 25     | 512 MB           | 8 GB            |
| 28     | 4 GB             | 16 GB           |
| 30     | 16 GB            | 32 GB           |
| 32     | 64 GB            | 128 GB          |

## macOS Installation

### Prerequisites

Install Xcode Command Line Tools:
```bash
xcode-select --install
```

For OpenMP support (optional but recommended):
```bash
brew install libomp
```

### Build from Source

```bash
git clone https://github.com/tsotchke/moonlab.git
cd moonlab
make
```

The build system automatically detects Apple Silicon and enables:
- Metal GPU acceleration
- Accelerate framework for BLAS/LAPACK
- NEON SIMD optimizations
- OpenMP parallelization (if libomp installed)

### Verify Installation

```bash
make test
```

Expected output includes:
```
╔═══════════════════════════════════════════════════════════╗
║         ALL TESTS COMPLETED                               ║
╚═══════════════════════════════════════════════════════════╝
```

### Apple Silicon Optimizations

The Makefile automatically detects M-series chips:

```
[BUILD] Detected Apple Silicon: M4
[BUILD] Optimization: -march=native (auto-detects M-series features)
[BUILD] Using 10 performance cores for OpenMP
```

No manual configuration required—the build adapts to your specific chip.

## Linux Installation

### Ubuntu/Debian

```bash
# Install build dependencies
sudo apt update
sudo apt install build-essential gcc make libgomp1

# Clone and build
git clone https://github.com/tsotchke/moonlab.git
cd moonlab
make
```

### Fedora/RHEL

```bash
sudo dnf install gcc make libgomp
git clone https://github.com/tsotchke/moonlab.git
cd moonlab
make
```

### Arch Linux

```bash
sudo pacman -S base-devel gcc openmp
git clone https://github.com/tsotchke/moonlab.git
cd moonlab
make
```

## Python Bindings

After building the C library:

```bash
cd bindings/python
pip install -e .
```

Or install dependencies and use directly:

```bash
pip install numpy
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/quantum-simulator
python -c "from moonlab import QuantumState; print('OK')"
```

### Requirements
- Python 3.8+
- NumPy
- (Optional) PyTorch for quantum ML layers

## Rust Bindings

Add to your `Cargo.toml`:

```toml
[dependencies]
moonlab = { path = "/path/to/quantum-simulator/bindings/rust/moonlab" }
```

Or build the static library for FFI:

```bash
make rust-lib  # Creates libquantumsim.a
```

The `moonlab-sys` crate provides raw FFI bindings; `moonlab` provides safe Rust wrappers.

## JavaScript/WebAssembly

### Using npm

```bash
cd bindings/javascript
pnpm install
pnpm build
```

### In a Web Project

```bash
npm install @moonlab/quantum-core @moonlab/quantum-viz
```

```javascript
import { QuantumState } from '@moonlab/quantum-core';
const state = await QuantumState.create({ numQubits: 4 });
```

## Build Options

### Compiler Selection

```bash
# Use specific compiler
CC=gcc-12 make

# Use Clang
CC=clang make
```

### Optimization Levels

The default build uses aggressive optimization (`-Ofast -march=native`). For debugging:

```bash
# Debug build
make CFLAGS="-g -O0 -DDEBUG"
```

### Disable Specific Features

```bash
# Disable OpenMP (single-threaded)
make OPENMP_FLAGS="" OPENMP_LIBS=""

# Disable Metal GPU (macOS)
make METAL_FLAGS=""
```

### Static Library

```bash
make libquantumsim.a
```

## GPU Acceleration

### Metal (macOS)

Metal GPU acceleration is enabled by default on macOS. Verify:

```bash
make metal_gpu
./examples/quantum/metal_gpu_benchmark
```

### CUDA (Linux/NVIDIA)

CUDA support requires manual configuration:

```bash
# Set CUDA paths
export CUDA_PATH=/usr/local/cuda
make CUDA_FLAGS="-I$CUDA_PATH/include" CUDA_LIBS="-L$CUDA_PATH/lib64 -lcudart -lcublas"
```

### Vulkan (Cross-platform)

```bash
# Requires Vulkan SDK
make VULKAN_FLAGS="-I$VULKAN_SDK/include" VULKAN_LIBS="-L$VULKAN_SDK/lib -lvulkan"
```

## Distributed Computing (MPI)

For multi-node cluster simulation:

```bash
# Install MPI
sudo apt install libopenmpi-dev  # Ubuntu
brew install open-mpi            # macOS

# Build with MPI support
make MPI_FLAGS="-I/usr/include/mpi" MPI_LIBS="-lmpi"
```

## Verifying the Installation

### Run Unit Tests

```bash
make test_unit
```

### Run Integration Tests

```bash
make test_integration
```

### Run All Examples

```bash
make examples
./examples/applications/vqe_h2_molecule
```

### Check GPU Acceleration

```bash
# Metal (macOS)
./examples/quantum/metal_batch_benchmark

# Expected output includes:
# Metal GPU batch processing: 40x speedup over CPU
```

## Troubleshooting

### "libquantumsim.so not found"

Add the library path:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)
```

Or on macOS:
```bash
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$(pwd)
```

### "OpenMP not found" on macOS

Install libomp via Homebrew:
```bash
brew install libomp
```

The Makefile automatically detects the installation path.

### Metal Errors

- Ensure running on native hardware (not VM)
- Check macOS version (12.0+ required)
- Verify GPU availability: `system_profiler SPDisplaysDataType`

### Compilation Errors with GCC

Some GCC versions have issues with aggressive optimization:
```bash
# Use Clang instead
CC=clang make
```

### Python Import Errors

Ensure the shared library is built and accessible:
```bash
make
ls -la libquantumsim.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)
```

## Uninstallation

```bash
make clean
rm -rf build/
```

To remove Python bindings:
```bash
pip uninstall moonlab
```

## Next Steps

- [Quick Start](quickstart.md) - Build your first quantum circuit
- [Building from Source](guides/building-from-source.md) - Advanced build options
- [GPU Acceleration Guide](guides/gpu-acceleration.md) - Configure hardware acceleration
