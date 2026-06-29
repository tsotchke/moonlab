# Archived Moonlab Documentation: Development Setup

This local Moonlab document is retained as archived vendor text for the QGTL integration audit; current supported claims are measured by `scripts/moonlab_doc_claim_audit.py` and grounded against `external/moonlab/README.md`, `external/moonlab/CMakeLists.txt`, and `docs/MOONLAB_OPEN_CORE_INTEGRATION.md`.

The historical text below is preserved as an archival snapshot, not as current release documentation.

```text
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

[archived fence delimiter: ```bash]
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
[archived fence delimiter: ```]

### Ubuntu/Debian

[archived fence delimiter: ```bash]
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
[archived fence delimiter: ```]

### Fedora/RHEL

[archived fence delimiter: ```bash]
# Install development tools
sudo dnf groupinstall "Development Tools"
sudo dnf install cmake git

# Optional
sudo dnf install clang-tools-extra doxygen
sudo dnf install python3-devel

# Verify
cmake --version
[archived fence delimiter: ```]

## Clone and Build

### Fork and Clone

[archived fence delimiter: ```bash]
# Fork via GitHub UI first, then:
git clone https://github.com/YOUR_USERNAME/moonlab.git
cd moonlab

# Add upstream remote
git remote add upstream https://github.com/tsotchke/moonlab.git

# Verify remotes
git remote -v
[archived fence delimiter: ```]

### Debug Build

[archived fence delimiter: ```bash]
mkdir build-debug && cd build-debug

cmake -DCMAKE_BUILD_TYPE=Debug \
      -DBUILD_TESTS=ON \
      -DSANITIZE_ADDRESS=ON \
      ..

make -j$(nproc)
[archived fence delimiter: ```]

### Release Build

[archived fence delimiter: ```bash]
mkdir build-release && cd build-release

cmake -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTS=ON \
      -DBUILD_EXAMPLES=ON \
      ..

make -j$(nproc)
[archived fence delimiter: ```]

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

[archived fence delimiter: ```json]
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
[archived fence delimiter: ```]

3. Create `.vscode/launch.json`:

[archived fence delimiter: ```json]
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
[archived fence delimiter: ```]

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

[archived fence delimiter: ```bash]
# Generate Xcode project
mkdir build-xcode && cd build-xcode
cmake -G Xcode -DBUILD_TESTS=ON ..
open moonlab.xcodeproj
[archived fence delimiter: ```]

## Python Development

### Virtual Environment

[archived fence delimiter: ```bash]
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install development dependencies
pip install -e "./bindings/python[dev]"

# Or manually
pip install pytest pytest-cov mypy black numpy
[archived fence delimiter: ```]

### Running Python Tests

[archived fence delimiter: ```bash]
# All tests
pytest bindings/python/tests/

# With coverage
pytest --cov=moonlab bindings/python/tests/

# Specific test
pytest bindings/python/tests/test_state.py -v
[archived fence delimiter: ```]

## Rust Development

### Setup

[archived fence delimiter: ```bash]
cd bindings/rust/moonlab

# Build
cargo build

# Test
cargo test

# Build with release optimizations
cargo build --release
[archived fence delimiter: ```]

### Linking to C Library

The Rust bindings require the C library. Set the library path:

[archived fence delimiter: ```bash]
export LIBRARY_PATH=/path/to/moonlab/build-release/lib
export LD_LIBRARY_PATH=/path/to/moonlab/build-release/lib
[archived fence delimiter: ```]

## Pre-Commit Hooks

### Setup

[archived fence delimiter: ```bash]
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install
[archived fence delimiter: ```]

### Configuration (`.pre-commit-config.yaml`)

[archived fence delimiter: ```yaml]
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
[archived fence delimiter: ```]

## Debugging

### GDB/LLDB

[archived fence delimiter: ```bash]
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
[archived fence delimiter: ```]

### Address Sanitizer

[archived fence delimiter: ```bash]
cmake -DCMAKE_BUILD_TYPE=Debug -DSANITIZE_ADDRESS=ON ..
make

# Run tests - ASan will report memory errors
./bin/test_quantum_gates
[archived fence delimiter: ```]

### Valgrind (Linux)

[archived fence delimiter: ```bash]
valgrind --leak-check=full --show-leak-kinds=all \
         ./bin/test_quantum_gates
[archived fence delimiter: ```]

## Keeping Up-to-Date

[archived fence delimiter: ```bash]
# Fetch upstream changes
git fetch upstream

# Merge into your branch
git checkout main
git merge upstream/main

# Rebase feature branch
git checkout feature/your-feature
git rebase main
[archived fence delimiter: ```]

## Troubleshooting

### CMake Cache Issues

[archived fence delimiter: ```bash]
# Clear cache and rebuild
rm -rf build-debug
mkdir build-debug && cd build-debug
cmake ..
[archived fence delimiter: ```]

### Library Path Issues

[archived fence delimiter: ```bash]
# macOS
export DYLD_LIBRARY_PATH=/path/to/build/lib:$DYLD_LIBRARY_PATH

# Linux
export LD_LIBRARY_PATH=/path/to/build/lib:$LD_LIBRARY_PATH
[archived fence delimiter: ```]

### Python Import Issues

[archived fence delimiter: ```bash]
# Ensure development install
pip install -e ./bindings/python

# Check installation
python -c "import moonlab; print(moonlab.__file__)"
[archived fence delimiter: ```]

## See Also

- [Code Style](code-style.md) - Formatting guidelines
- [Testing Guide](testing-guide.md) - Testing practices
- [Building from Source](../guides/building-from-source.md) - Detailed build options

```
