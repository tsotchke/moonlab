# Building from Source

Compile Moonlab with custom options and optimizations.

## Prerequisites

### Required Tools

| Tool | Minimum Version | Purpose |
|------|-----------------|---------|
| C Compiler | GCC 9+ / Clang 12+ | Core library |
| CMake | 3.16+ | Build system |
| Make or Ninja | Any | Build execution |

### Optional Tools

| Tool | Purpose |
|------|---------|
| Python 3.8+ | Python bindings |
| Rust 1.65+ | Rust bindings |
| Node.js 18+ | JavaScript bindings |
| MPI | Distributed computing |
| Doxygen | API documentation |

## Basic Build

### Clone and Build

```bash
# Clone repository
git clone https://github.com/tsotchke/moonlab.git
cd moonlab

# Create build directory
mkdir build && cd build

# Configure
cmake ..

# Build
make -j$(nproc)

# Test
make test

# Install (optional)
sudo make install
```

### Verify Installation

```bash
# Check version
./bin/moonlab --version

# Run quick test
./bin/moonlab-test
```

## Build Options

### CMake Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | Release | Debug, Release, RelWithDebInfo |
| `ENABLE_GPU` | AUTO | Enable Metal GPU acceleration |
| `ENABLE_MPI` | OFF | Enable distributed computing |
| `ENABLE_OPENMP` | AUTO | Enable OpenMP threading |
| `BUILD_PYTHON` | ON | Build Python bindings |
| `BUILD_RUST` | ON | Build Rust bindings |
| `BUILD_JAVASCRIPT` | OFF | Build WebAssembly bindings |
| `BUILD_TESTS` | ON | Build test suite |
| `BUILD_EXAMPLES` | ON | Build example programs |
| `BUILD_DOCS` | OFF | Build documentation |
| `DOUBLE_PRECISION` | ON | Use double precision |

### Example Configurations

#### Maximum Performance

```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_GPU=ON \
      -DENABLE_OPENMP=ON \
      -DCMAKE_C_FLAGS="-march=native" \
      ..
```

#### Debug Build

```bash
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DSANITIZE_ADDRESS=ON \
      -DSANITIZE_UNDEFINED=ON \
      ..
```

#### Minimal Build

```bash
cmake -DBUILD_PYTHON=OFF \
      -DBUILD_RUST=OFF \
      -DBUILD_TESTS=OFF \
      -DBUILD_EXAMPLES=OFF \
      ..
```

#### Distributed Computing

```bash
cmake -DENABLE_MPI=ON \
      -DMPI_C_COMPILER=/usr/local/bin/mpicc \
      ..
```

## Platform-Specific Instructions

### macOS (Apple Silicon)

```bash
# Install Xcode command line tools
xcode-select --install

# Install dependencies via Homebrew
brew install cmake

# Build with Metal support (automatic on Apple Silicon)
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(sysctl -n hw.ncpu)
```

### macOS (Intel)

```bash
# Metal is available but CPU may be preferred
cmake -DENABLE_GPU=OFF \
      -DCMAKE_C_FLAGS="-march=native" \
      ..
```

### Linux (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install build-essential cmake libopenblas-dev

# Optional: OpenMP
sudo apt-get install libomp-dev

# Optional: MPI
sudo apt-get install libopenmpi-dev

# Build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### Linux (RHEL/CentOS)

```bash
# Install dependencies
sudo yum groupinstall "Development Tools"
sudo yum install cmake3 openblas-devel

# Build
mkdir build && cd build
cmake3 -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

## Building Bindings

### Python Bindings

```bash
# Method 1: CMake
cmake -DBUILD_PYTHON=ON ..
make python_bindings

# Method 2: pip (from source)
pip install ./bindings/python

# Method 3: Development install
pip install -e ./bindings/python
```

Verify:
```python
import moonlab
print(moonlab.__version__)
```

### Rust Bindings

```bash
# Build Rust crate
cd bindings/rust/moonlab
cargo build --release

# Run tests
cargo test
```

### JavaScript/WebAssembly

```bash
# Requires Emscripten SDK
source /path/to/emsdk/emsdk_env.sh

# Configure for WebAssembly
mkdir build-wasm && cd build-wasm
emcmake cmake -DBUILD_JAVASCRIPT=ON ..
emmake make

# Output in bindings/javascript/packages/core/wasm/
```

## Optimization Flags

### Compiler Optimization

```bash
# Aggressive optimization
cmake -DCMAKE_C_FLAGS="-O3 -march=native -ffast-math" ..

# Link-time optimization
cmake -DCMAKE_C_FLAGS="-flto" \
      -DCMAKE_EXE_LINKER_FLAGS="-flto" ..
```

### SIMD Selection

```bash
# Force specific SIMD level
cmake -DSIMD_LEVEL=AVX2 ..    # Intel AVX2
cmake -DSIMD_LEVEL=AVX512 ..  # Intel AVX-512
cmake -DSIMD_LEVEL=NEON ..    # ARM NEON
cmake -DSIMD_LEVEL=NONE ..    # Disable SIMD
```

## Troubleshooting Build Issues

### CMake Can't Find Dependencies

```bash
# Specify paths explicitly
cmake -DBLAS_DIR=/opt/openblas \
      -DPYTHON_EXECUTABLE=/usr/bin/python3 \
      ..
```

### Compiler Errors

```bash
# Check compiler version
gcc --version
clang --version

# Use specific compiler
cmake -DCMAKE_C_COMPILER=/usr/bin/clang ..
```

### Link Errors

```bash
# Check library paths
ldd ./bin/moonlab

# Add library paths
cmake -DCMAKE_LIBRARY_PATH="/usr/local/lib" ..
```

### GPU Not Detected

```bash
# Verify Metal support (macOS)
system_profiler SPDisplaysDataType | grep Metal

# Check GPU build
cmake -DENABLE_GPU=ON -DCMAKE_VERBOSE_MAKEFILE=ON ..
```

## Cross-Compilation

### ARM64 from x86_64

```bash
cmake -DCMAKE_SYSTEM_NAME=Linux \
      -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
      -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
      ..
```

### iOS

```bash
cmake -DCMAKE_SYSTEM_NAME=iOS \
      -DCMAKE_OSX_ARCHITECTURES=arm64 \
      ..
```

## Packaging

### Create Distribution Package

```bash
# CPack packaging
cmake -DCPACK_GENERATOR="TGZ;DEB;RPM" ..
make package
```

### Create Python Wheel

```bash
cd bindings/python
python -m build
# Output: dist/moonlab-*.whl
```

## Verification

### Run Test Suite

```bash
# All tests
make test

# Specific test
./bin/test_quantum_gates

# With verbose output
ctest --verbose
```

### Benchmark

```bash
# Run benchmarks
./bin/benchmark_gates

# Compare with baseline
./bin/benchmark_gates --compare baseline.json
```

## See Also

- [GPU Acceleration Guide](gpu-acceleration.md) - Configure GPU support
- [Performance Tuning](performance-tuning.md) - Optimize for your hardware
- [Troubleshooting](../troubleshooting.md) - Common issues

