# Moonlab Debian Release Build
# Used for ARM64 builds via QEMU
FROM debian:12-slim

ARG VERSION=0.1.0
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    gcc \
    g++ \
    libomp-dev \
    libopenblas-dev \
    liblapacke-dev \
    libssl-dev \
    dpkg-dev \
    file \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy source
WORKDIR /app
COPY . .

# Build library with CMake (the Makefile path is legacy and doesn't
# generate moonlab_build_info.h that src/utils/manifest.c needs).
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j --target quantumsim moonlab-control-server

# Stage the same install tree that the hosted release job packages.
RUN cmake --install build --prefix pkg && \
    cp README.md LICENSE pkg/

# Verify the ARM64 package inside the ARM64 image so the external consumer
# compile/link/run path is checked before the workflow extracts pkg/.
RUN tar -czf /tmp/moonlab-arm64-verify.tar.gz -C pkg . && \
    scripts/verify_release_package.sh --package /tmp/moonlab-arm64-verify.tar.gz

# Keep container running for file extraction
CMD ["sleep", "infinity"]
