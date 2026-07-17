# Moonlab Ubuntu Release Build
# Matches GitHub Actions ubuntu-22.04 environment
FROM ubuntu:22.04

ARG VERSION=1.1.0
LABEL org.opencontainers.image.version="${VERSION}"
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies - matching GitHub Actions CI
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
    lcov \
    valgrind \
    dpkg-dev \
    file \
    curl \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy source
WORKDIR /app
COPY . .

# Build with CMake (the Makefile path is legacy and doesn't generate
# moonlab_build_info.h that src/utils/manifest.c needs -- see
# docker/debian/release/Dockerfile, which this mirrors).
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DQSIM_NATIVE_ARCH=OFF \
        -DQSIM_BUILD_TESTS=ON -DQSIM_BUILD_EXAMPLES=OFF -DQSIM_BUILD_BENCHMARKS=OFF && \
    cmake --build build -j

# Run the test suite via ctest -- matching what CI enforces on
# ubuntu-22.04. A failing test must fail the image build rather than
# being masked with `|| echo`.
RUN cd build && ctest --output-on-failure

# Stage the same install tree the hosted release job packages.
RUN cmake --install build --prefix pkg && \
    cp README.md LICENSE pkg/

# Verify artifacts
RUN ls -la pkg/lib/

# Keep container running for file extraction
CMD ["sleep", "infinity"]
