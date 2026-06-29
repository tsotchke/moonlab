# Archived Moonlab Documentation: Installation

This local Moonlab document is retained as archived vendor text for the QGTL integration audit; current supported claims are measured by `scripts/moonlab_doc_claim_audit.py` and grounded against `external/moonlab/README.md`, `external/moonlab/CMakeLists.txt`, and `docs/MOONLAB_OPEN_CORE_INTEGRATION.md`.

The historical text below is preserved as an archival snapshot, not as current release documentation.

```text
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
[archived fence delimiter: ```bash]
xcode-select --install
[archived fence delimiter: ```]

For OpenMP support (optional but recommended):
[archived fence delimiter: ```bash]
brew install libomp
[archived fence delimiter: ```]

### Build from Source

[archived fence delimiter: ```bash]
git clone https://github.com/tsotchke/moonlab.git
cd moonlab
make
[archived fence delimiter: ```]

The build system automatically detects Apple Silicon and enables:
- Metal GPU acceleration
- Accelerate framework for BLAS/LAPACK
- NEON SIMD optimizations
- OpenMP parallelization (if libomp installed)

### Verify Installation

[archived fence delimiter: ```bash]
make test
[archived fence delimiter: ```]

Expected output includes:
[archived fence delimiter: ```]
╔═══════════════════════════════════════════════════════════╗
║         ALL TESTS COMPLETED                               ║
╚═══════════════════════════════════════════════════════════╝
[archived fence delimiter: ```]

### Apple Silicon Optimizations

The Makefile automatically detects M-series chips:

[archived fence delimiter: ```]
[BUILD] Detected Apple Silicon: M4
[BUILD] Optimization: -march=native (auto-detects M-series features)
[BUILD] Using 10 performance cores for OpenMP
[archived fence delimiter: ```]

No manual configuration required—the build adapts to your specific chip.

## Linux Installation

### Ubuntu/Debian

[archived fence delimiter: ```bash]
# Install build dependencies
sudo apt update
sudo apt install build-essential gcc make libgomp1

# Clone and build
git clone https://github.com/tsotchke/moonlab.git
cd moonlab
make
[archived fence delimiter: ```]

### Fedora/RHEL

[archived fence delimiter: ```bash]
sudo dnf install gcc make libgomp
git clone https://github.com/tsotchke/moonlab.git
cd moonlab
make
[archived fence delimiter: ```]

### Arch Linux

[archived fence delimiter: ```bash]
sudo pacman -S base-devel gcc openmp
git clone https://github.com/tsotchke/moonlab.git
cd moonlab
make
[archived fence delimiter: ```]

## Python Bindings

After building the C library:

[archived fence delimiter: ```bash]
cd bindings/python
pip install -e .
[archived fence delimiter: ```]

Or install dependencies and use directly:

[archived fence delimiter: ```bash]
pip install numpy
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/moonlab
python -c "from moonlab import QuantumState; print('OK')"
[archived fence delimiter: ```]

### Requirements
- Python 3.8+
- NumPy
- (Optional) PyTorch for quantum ML layers

## Rust Bindings

Add to your `Cargo.toml`:

[archived fence delimiter: ```toml]
[dependencies]
moonlab = { path = "/path/to/moonlab/bindings/rust/moonlab" }
[archived fence delimiter: ```]

Or build the static library for FFI:

[archived fence delimiter: ```bash]
make rust-lib  # Creates libquantumsim.a
[archived fence delimiter: ```]

The `moonlab-sys` crate provides raw FFI bindings; `moonlab` provides safe Rust wrappers.

## JavaScript/WebAssembly

### Using npm

[archived fence delimiter: ```bash]
cd bindings/javascript
pnpm install
pnpm build
[archived fence delimiter: ```]

### In a Web Project

[archived fence delimiter: ```bash]
npm install @moonlab/quantum-core @moonlab/quantum-viz
[archived fence delimiter: ```]

[archived fence delimiter: ```javascript]
import { QuantumState } from '@moonlab/quantum-core';
const state = await QuantumState.create({ numQubits: 4 });
[archived fence delimiter: ```]

## Build Options

### Compiler Selection

[archived fence delimiter: ```bash]
# Use specific compiler
CC=gcc-12 make

# Use Clang
CC=clang make
[archived fence delimiter: ```]

### Optimization Levels

The default build uses aggressive optimization (`-Ofast -march=native`). For debugging:

[archived fence delimiter: ```bash]
# Debug build
make CFLAGS="-g -O0 -DDEBUG"
[archived fence delimiter: ```]

### Disable Specific Features

[archived fence delimiter: ```bash]
# Disable OpenMP (single-threaded)
make OPENMP_FLAGS="" OPENMP_LIBS=""

# Disable Metal GPU (macOS)
make METAL_FLAGS=""
[archived fence delimiter: ```]

### Static Library

[archived fence delimiter: ```bash]
make libquantumsim.a
[archived fence delimiter: ```]

## GPU Acceleration

### Metal (macOS)

Metal GPU acceleration is enabled by default on macOS. Verify:

[archived fence delimiter: ```bash]
make metal_gpu
./examples/quantum/metal_gpu_benchmark
[archived fence delimiter: ```]

### CUDA (Linux/NVIDIA)

CUDA support requires manual configuration:

[archived fence delimiter: ```bash]
# Set CUDA paths
export CUDA_PATH=/usr/local/cuda
make CUDA_FLAGS="-I$CUDA_PATH/include" CUDA_LIBS="-L$CUDA_PATH/lib64 -lcudart -lcublas"
[archived fence delimiter: ```]

### Vulkan (Cross-platform)

[archived fence delimiter: ```bash]
# Requires Vulkan SDK
make VULKAN_FLAGS="-I$VULKAN_SDK/include" VULKAN_LIBS="-L$VULKAN_SDK/lib -lvulkan"
[archived fence delimiter: ```]

## Distributed Computing (MPI)

For multi-node cluster simulation:

[archived fence delimiter: ```bash]
# Install MPI
sudo apt install libopenmpi-dev  # Ubuntu
brew install open-mpi            # macOS

# Build with MPI support
make MPI_FLAGS="-I/usr/include/mpi" MPI_LIBS="-lmpi"
[archived fence delimiter: ```]

## Verifying the Installation

### Run Unit Tests

[archived fence delimiter: ```bash]
make test_unit
[archived fence delimiter: ```]

### Run Integration Tests

[archived fence delimiter: ```bash]
make test_integration
[archived fence delimiter: ```]

### Run All Examples

[archived fence delimiter: ```bash]
make examples
./examples/applications/vqe_h2_molecule
[archived fence delimiter: ```]

### Check GPU Acceleration

[archived fence delimiter: ```bash]
# Metal (macOS)
./examples/quantum/metal_batch_benchmark

# Expected output includes:
# Metal GPU batch processing: 40x speedup over CPU
[archived fence delimiter: ```]

## Troubleshooting

### "libquantumsim.so not found"

Add the library path:
[archived fence delimiter: ```bash]
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)
[archived fence delimiter: ```]

Or on macOS:
[archived fence delimiter: ```bash]
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$(pwd)
[archived fence delimiter: ```]

### "OpenMP not found" on macOS

Install libomp via Homebrew:
[archived fence delimiter: ```bash]
brew install libomp
[archived fence delimiter: ```]

The Makefile automatically detects the installation path.

### Metal Errors

- Ensure running on native hardware (not VM)
- Check macOS version (12.0+ required)
- Verify GPU availability: `system_profiler SPDisplaysDataType`

### Compilation Errors with GCC

Some GCC versions have issues with aggressive optimization:
[archived fence delimiter: ```bash]
# Use Clang instead
CC=clang make
[archived fence delimiter: ```]

### Python Import Errors

Ensure the shared library is built and accessible:
[archived fence delimiter: ```bash]
make
ls -la libquantumsim.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)
[archived fence delimiter: ```]

## Uninstallation

[archived fence delimiter: ```bash]
make clean
rm -rf build/
[archived fence delimiter: ```]

To remove Python bindings:
[archived fence delimiter: ```bash]
pip uninstall moonlab
[archived fence delimiter: ```]

## Next Steps

- [Quick Start](quickstart.md) - Build your first quantum circuit
- [Building from Source](guides/building-from-source.md) - Advanced build options
- [GPU Acceleration Guide](guides/gpu-acceleration.md) - Configure hardware acceleration
```
