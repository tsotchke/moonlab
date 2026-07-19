#!/usr/bin/env bash
# Build, test, and package Moonlab from source on mainstream Linux families.
#
# The build itself is distro-independent. This driver detects the package
# family from /etc/os-release, installs the corresponding build dependencies,
# then performs the same portable release configuration used by tagged builds.
#
# Supported families:
#   debian  Debian, Ubuntu, Mint, LMDE, Pop!_OS, elementary, Kali, ...
#   fedora  Fedora, RHEL, CentOS Stream, Rocky, Alma, ...
#   arch    Arch, Manjaro, EndeavourOS, CachyOS, ...
#   suse    openSUSE Tumbleweed/Leap and SLES
#   alpine  Alpine Linux (musl)
#   nixos   NixOS (dependencies are entered through a temporary nix-shell)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build-linux"
INSTALL_DEPS=1
RUN_CTEST=0
FULL_CTEST=0
VERIFY_PACKAGE=0
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '2')"

usage() {
    cat <<'EOF'
Usage: scripts/build-linux.sh [options]

Options:
  --build-dir DIR       Build directory (default: build-linux)
  --jobs N              Parallel build/test jobs (default: host CPU count)
  --no-install-deps     Use dependencies already installed on the host
  --ctest               Run the bounded release test labels after building
  --full-ctest          Run every registered CTest (physical-host gate)
  --verify-package      Package the install tree and test an external consumer
  -h, --help            Show this help
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --build-dir)
            [ "$#" -ge 2 ] || { echo "--build-dir requires a value" >&2; exit 2; }
            BUILD_DIR="$2"
            shift 2
            ;;
        --jobs)
            [ "$#" -ge 2 ] || { echo "--jobs requires a value" >&2; exit 2; }
            JOBS="$2"
            shift 2
            ;;
        --no-install-deps)
            INSTALL_DEPS=0
            shift
            ;;
        --ctest)
            RUN_CTEST=1
            shift
            ;;
        --full-ctest)
            RUN_CTEST=1
            FULL_CTEST=1
            shift
            ;;
        --verify-package)
            VERIFY_PACKAGE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$JOBS" in
    ''|*[!0-9]*) echo "--jobs must be a positive integer" >&2; exit 2 ;;
    0) echo "--jobs must be a positive integer" >&2; exit 2 ;;
esac

if [ ! -r /etc/os-release ]; then
    echo "build-linux.sh: /etc/os-release not found; this driver requires Linux" >&2
    exit 4
fi

# shellcheck disable=SC1091
. /etc/os-release

family=""
for candidate in "${ID:-}" ${ID_LIKE:-}; do
    case "$candidate" in
        debian|ubuntu|linuxmint) family=debian; break ;;
        fedora|rhel|centos|rocky|almalinux) family=fedora; break ;;
        arch) family=arch; break ;;
        suse|opensuse|opensuse-tumbleweed|opensuse-leap|sles) family=suse; break ;;
        alpine) family=alpine; break ;;
        nixos) family=nixos; break ;;
    esac
done
if [ -z "$family" ] && [ -r /etc/linuxmint/info ]; then
    family=debian
fi

if [ -z "$family" ]; then
    cat >&2 <<EOF
build-linux.sh: unsupported distribution ID='${ID:-unknown}' ID_LIKE='${ID_LIKE:-}'.
Install CMake 3.20+, Ninja, a C11/C++17 toolchain, OpenBLAS, LAPACKE,
OpenSSL, pkg-config, Git, and Python 3, then rerun with --no-install-deps.
EOF
    exit 4
fi

sudo_cmd() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        echo "build-linux.sh: dependency installation requires root or sudo" >&2
        exit 3
    fi
}

install_debian() {
    sudo_cmd apt-get update -o Acquire::Retries=5
    sudo_cmd env DEBIAN_FRONTEND=noninteractive apt-get install -y \
        ca-certificates build-essential cmake ninja-build git \
        libopenblas-dev liblapacke-dev libssl-dev pkg-config python3
}

install_fedora() {
    # RHEL-family rebuilds keep scientific development packages in an
    # opt-in repository: PowerTools on major 8, CRB on major 9+. Fedora
    # itself has neither identifier and simply continues with its defaults.
    sudo_cmd dnf install -y dnf-plugins-core
    if dnf repolist --all | grep -Eq '^powertools[[:space:]]'; then
        sudo_cmd dnf config-manager --set-enabled powertools
    fi
    if dnf repolist --all | grep -Eq '^crb[[:space:]]'; then
        sudo_cmd dnf config-manager --set-enabled crb
    fi

    local compiler_packages=(gcc gcc-c++)
    if [ "${VERSION_ID%%.*}" = "8" ] &&
       dnf -q list --available gcc-toolset-12-gcc >/dev/null 2>&1; then
        # EL8's system GCC 8 predates Moonlab's GCC 9 floor and lacks the
        # RDSEED intrinsic used by the entropy provider. Use the supported
        # parallel toolset without replacing the system compiler.
        compiler_packages=(gcc-toolset-12-gcc gcc-toolset-12-gcc-c++)
        export CC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
        export CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
    fi
    sudo_cmd dnf install -y \
        ca-certificates "${compiler_packages[@]}" cmake ninja-build git \
        openblas-devel lapack-devel openssl-devel pkgconf-pkg-config python3
}

install_arch() {
    sudo_cmd pacman -Syu --needed --noconfirm \
        base-devel ca-certificates cmake ninja git \
        openblas lapacke openssl pkgconf python
}

install_suse() {
    sudo_cmd zypper --non-interactive install \
        ca-certificates gcc gcc-c++ cmake ninja git \
        openblas-devel lapack-devel lapacke-devel libopenssl-devel \
        pkg-config python3
}

install_alpine() {
    sudo_cmd apk add --no-cache \
        bash build-base ca-certificates cmake ninja git linux-headers \
        openblas-dev openssl-dev pkgconf python3
}

install_nixos() {
    command -v nix-shell >/dev/null 2>&1 || {
        echo "build-linux.sh: NixOS requires nix-shell" >&2
        exit 5
    }
    if [ "${MOONLAB_NIX_SHELL:-0}" = "1" ]; then
        echo "build-linux.sh: required build dependencies are missing inside nix-shell" >&2
        exit 5
    fi

    local rerun=(env MOONLAB_NIX_SHELL=1 bash "$SCRIPT_DIR/build-linux.sh"
        --build-dir "$BUILD_DIR" --jobs "$JOBS" --no-install-deps)
    if [ "$RUN_CTEST" -eq 1 ]; then
        if [ "$FULL_CTEST" -eq 1 ]; then
            rerun+=(--full-ctest)
        else
            rerun+=(--ctest)
        fi
    fi
    if [ "$VERIFY_PACKAGE" -eq 1 ]; then
        rerun+=(--verify-package)
    fi
    local rerun_command
    printf -v rerun_command '%q ' "${rerun[@]}"
    exec nix-shell -p bash cmake ninja gcc git openblas lapack openssl \
        pkg-config python3 --run "$rerun_command"
}

echo "build-linux.sh: ${PRETTY_NAME:-${ID:-linux}} -> family '${family}'"
if [ "$INSTALL_DEPS" -eq 1 ]; then
    case "$family" in
        debian) install_debian ;;
        fedora) install_fedora ;;
        arch) install_arch ;;
        suse) install_suse ;;
        alpine) install_alpine ;;
        nixos) install_nixos ;;
    esac
fi

for tool in cmake ninja git; do
    case "$tool" in
        cmake|ninja|git) ;;
        *) echo "build-linux.sh: internal invalid tool probe '$tool'" >&2; exit 5 ;;
    esac
    tool_path="$(command -v -- "$tool" 2>/dev/null)" || {
        echo "build-linux.sh: required tool '$tool' is unavailable" >&2
        exit 5
    }
    if [ ! -x "$tool_path" ] || [ -d "$tool_path" ]; then
        echo "build-linux.sh: required tool '$tool' resolved to a non-executable path" >&2
        exit 5
    fi
done

cmake_version="$(cmake --version | sed -n '1s/[^0-9]*\([0-9][0-9.]*\).*/\1/p')"
cmake_major="${cmake_version%%.*}"
cmake_minor="${cmake_version#*.}"
cmake_minor="${cmake_minor%%.*}"
if [ -z "$cmake_major" ] || [ -z "$cmake_minor" ] ||
   [ "$cmake_major" -lt 3 ] || { [ "$cmake_major" -eq 3 ] && [ "$cmake_minor" -lt 20 ]; }; then
    echo "build-linux.sh: CMake 3.20+ is required; found ${cmake_version:-unknown}" >&2
    exit 5
fi

cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DQSIM_BUILD_EXAMPLES=OFF \
    -DQSIM_BUILD_BENCHMARKS=OFF \
    -DQSIM_NATIVE_ARCH=OFF \
    -DQSIM_FAST_MATH=OFF \
    -DQSIM_WERROR=ON

cmake --build "$BUILD_DIR" --parallel "$JOBS"

if [ "$RUN_CTEST" -eq 1 ]; then
    if [ "$FULL_CTEST" -eq 1 ]; then
        ctest --test-dir "$BUILD_DIR" --output-on-failure --timeout 600 \
            --parallel "$JOBS"
    else
        ctest --test-dir "$BUILD_DIR" --output-on-failure --timeout 180 \
            -L "(core|abi|topology|health)" -LE "long|memory_heavy" \
            --parallel "$JOBS"
    fi
fi

if [ "$VERIFY_PACKAGE" -eq 1 ]; then
    package_path="$BUILD_DIR/moonlab-linux-smoke.tar.gz"
    "$SCRIPT_DIR/package_release_artifact.sh" \
        --build-dir "$BUILD_DIR" --output "$package_path"
    "$SCRIPT_DIR/verify_release_package.sh" --package "$package_path"
fi

echo "build-linux.sh: PASS (${PRETTY_NAME:-${ID:-linux}}, ${family})"
