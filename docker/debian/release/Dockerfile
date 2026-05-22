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
    cmake --build build -j --target quantumsim

# Stage artefacts under /app/lib for the release.yml extractor.
# Keep both libquantumsim.so symlinks and the static, where built.
RUN mkdir -p lib && \
    cp build/libquantumsim.* lib/ 2>/dev/null || true

# Keep container running for file extraction
CMD ["sleep", "infinity"]
