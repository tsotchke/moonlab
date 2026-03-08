# Development Setup

Configure your environment for Moonlab development.

## Prerequisites

### Required Tools

| Tool | Minimum Version | Check Command |
|------|-----------------|---------------|
| Git | 2.30+ | `git --version` |
| C Compiler | GCC 9+ / Clang 12+ | `gcc --version` |
| CMake | 3.16+ | `cmake --version` |
| Make or Ninja | Any | `make --version` |

### Optional Tools

| Tool | Purpose | Install |
|------|---------|---------|
| Python 3.8+ | Python bindings | `python3 --version` |
| Rust 1.65+ | Rust bindings | `rustc --version` |
| Doxygen | API docs | `doxygen --version` |
| Valgrind | Memory debugging | `valgrind --version` |
| clang-format | Code formatting | `clang-format --version` |

## Platform Setup

### macOS

```bash
# Install Xcode command line tools
xcode-select --install

# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake
brew install llvm  # For clang-format
brew install doxygen  # Optional: docs

# Verify
cmake --version
clang --version
```

### Ubuntu/Debian

```bash
# Update package list
sudo apt-get update

# Install build essentials
sudo apt-get install build-essential cmake git

# Install optional dependencies
sudo apt-get install clang-format doxygen
sudo apt-get install python3-dev python3-pip

# For MPI support
sudo apt-get install libopenmpi-dev

# Verify
cmake --version
gcc --version
```

### Fedora/RHEL

```bash
# Install development tools
sudo dnf groupinstall "Development Tools"
sudo dnf install cmake git

# Optional
sudo dnf install clang-tools-extra doxygen
sudo dnf install python3-devel

# Verify
cmake --version
```

## Clone and Build

### Fork and Clone

```bash
# Fork via GitHub UI first, then:
git clone https://github.com/YOUR_USERNAME/moonlab.git
cd moonlab

# Add upstream remote
git remote add upstream https://github.com/tsotchke/moonlab.git

# Verify remotes
git remote -v
```

### Debug Build

```bash
mkdir build-debug && cd build-debug

cmake -DCMAKE_BUILD_TYPE=Debug \
      -DBUILD_TESTS=ON \
      -DSANITIZE_ADDRESS=ON \
      ..

make -j$(nproc)
```

### Release Build

```bash
mkdir build-release && cd build-release

cmake -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTS=ON \
      -DBUILD_EXAMPLES=ON \
      ..

make -j$(nproc)
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | Release | Debug, Release, RelWithDebInfo |
| `BUILD_TESTS` | ON | Build test suite |
| `BUILD_EXAMPLES` | ON | Build examples |
| `BUILD_DOCS` | OFF | Build documentation |
| `ENABLE_GPU` | AUTO | Metal GPU support |
| `ENABLE_MPI` | OFF | Distributed computing |
| `SANITIZE_ADDRESS` | OFF | Address sanitizer |
| `SANITIZE_UNDEFINED` | OFF | UB sanitizer |
| `BUILD_PYTHON` | ON | Python bindings |
| `BUILD_RUST` | ON | Rust bindings |

## IDE Setup

### VS Code

1. Install extensions:
   - C/C++ (Microsoft)
   - CMake Tools
   - Python
   - clangd (optional, better IntelliSense)

2. Create `.vscode/settings.json`:

```json
{
    "cmake.buildDirectory": "${workspaceFolder}/build-debug",
    "cmake.configureSettings": {
        "CMAKE_BUILD_TYPE": "Debug",
        "BUILD_TESTS": "ON"
    },
    "C_Cpp.default.configurationProvider": "ms-vscode.cmake-tools",
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "editor.formatOnSave": true,
    "files.associations": {
        "*.h": "c"
    }
}
```

3. Create `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug Tests",
            "type": "lldb",
            "request": "launch",
            "program": "${workspaceFolder}/build-debug/bin/test_quantum_gates",
            "args": [],
            "cwd": "${workspaceFolder}"
        },
        {
            "name": "Debug Example",
            "type": "lldb",
            "request": "launch",
            "program": "${workspaceFolder}/build-debug/bin/examples/hello_quantum",
            "args": [],
            "cwd": "${workspaceFolder}"
        }
    ]
}
```

### CLion

1. Open the project folder (CMake detected automatically)
2. Configure CMake profiles in Settings → Build, Execution, Deployment → CMake:

   **Debug Profile:**
   - Build type: Debug
   - CMake options: `-DBUILD_TESTS=ON -DSANITIZE_ADDRESS=ON`

   **Release Profile:**
   - Build type: Release
   - CMake options: `-DBUILD_TESTS=ON`

3. Mark source directories:
   - Right-click `src/` → Mark as → Sources Root
   - Right-click `tests/` → Mark as → Test Sources Root

### Xcode (macOS)

```bash
# Generate Xcode project
mkdir build-xcode && cd build-xcode
cmake -G Xcode -DBUILD_TESTS=ON ..
open moonlab.xcodeproj
```

## Python Development

### Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install development dependencies
pip install -e "./bindings/python[dev]"

# Or manually
pip install pytest pytest-cov mypy black numpy
```

### Running Python Tests

```bash
# All tests
pytest bindings/python/tests/

# With coverage
pytest --cov=moonlab bindings/python/tests/

# Specific test
pytest bindings/python/tests/test_state.py -v
```

## Rust Development

### Setup

```bash
cd bindings/rust/moonlab

# Build
cargo build

# Test
cargo test

# Build with release optimizations
cargo build --release
```

### Linking to C Library

The Rust bindings require the C library. Set the library path:

```bash
export LIBRARY_PATH=/path/to/moonlab/build-release/lib
export LD_LIBRARY_PATH=/path/to/moonlab/build-release/lib
```

## Pre-Commit Hooks

### Setup

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install
```

### Configuration (`.pre-commit-config.yaml`)

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: local
    hooks:
      - id: clang-format
        name: clang-format
        entry: clang-format -i
        language: system
        types: [c]
        files: ^src/

      - id: tests
        name: tests
        entry: bash -c 'cd build-debug && make test'
        language: system
        pass_filenames: false
```

## Debugging

### GDB/LLDB

```bash
# Build with debug symbols
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Debug with LLDB (macOS)
lldb ./bin/test_quantum_gates
(lldb) breakpoint set --name quantum_state_create
(lldb) run

# Debug with GDB (Linux)
gdb ./bin/test_quantum_gates
(gdb) break quantum_state_create
(gdb) run
```

### Address Sanitizer

```bash
cmake -DCMAKE_BUILD_TYPE=Debug -DSANITIZE_ADDRESS=ON ..
make

# Run tests - ASan will report memory errors
./bin/test_quantum_gates
```

### Valgrind (Linux)

```bash
valgrind --leak-check=full --show-leak-kinds=all \
         ./bin/test_quantum_gates
```

## Keeping Up-to-Date

```bash
# Fetch upstream changes
git fetch upstream

# Merge into your branch
git checkout main
git merge upstream/main

# Rebase feature branch
git checkout feature/your-feature
git rebase main
```

## Troubleshooting

### CMake Cache Issues

```bash
# Clear cache and rebuild
rm -rf build-debug
mkdir build-debug && cd build-debug
cmake ..
```

### Library Path Issues

```bash
# macOS
export DYLD_LIBRARY_PATH=/path/to/build/lib:$DYLD_LIBRARY_PATH

# Linux
export LD_LIBRARY_PATH=/path/to/build/lib:$LD_LIBRARY_PATH
```

### Python Import Issues

```bash
# Ensure development install
pip install -e ./bindings/python

# Check installation
python -c "import moonlab; print(moonlab.__file__)"
```

## See Also

- [Code Style](code-style.md) - Formatting guidelines
- [Testing Guide](testing-guide.md) - Testing practices
- [Building from Source](../guides/building-from-source.md) - Detailed build options

