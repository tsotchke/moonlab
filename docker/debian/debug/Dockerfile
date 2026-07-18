# Moonlab Debian Debug Build
FROM debian:12-slim

ARG VERSION=1.2.0
LABEL org.opencontainers.image.version="${VERSION}"
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies including debug tools
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
    gdb \
    valgrind \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy source
WORKDIR /app
COPY . .

# Build with debug symbols via CMake (the Makefile path is legacy and
# doesn't generate moonlab_build_info.h that src/utils/manifest.c
# needs -- see docker/debian/release/Dockerfile, which this mirrors).
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DQSIM_NATIVE_ARCH=OFF \
        -DQSIM_BUILD_TESTS=ON -DQSIM_BUILD_EXAMPLES=OFF -DQSIM_BUILD_BENCHMARKS=OFF && \
    cmake --build build -j

# Run the test suite via ctest. This is a debug image built for
# interactive gdb/valgrind sessions against a known-good binary --
# masking a test failure with `|| echo` would ship exactly the
# untrustworthy debug target gdb/valgrind sessions are meant to rule
# out, so a failure here fails the image build.
RUN cd build && ctest --output-on-failure

# Keep container running for debugging
CMD ["sleep", "infinity"]
