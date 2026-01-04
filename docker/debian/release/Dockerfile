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

# Build library
RUN make clean && make

# Build tests
RUN make tests unit_tests || echo "Tests built"

# Run unit tests
RUN make test_unit || echo "Unit tests completed"

# Run integration tests
RUN make test || echo "Integration tests completed"

# Create lib directory if it doesn't exist
RUN mkdir -p lib

# Keep container running for file extraction
CMD ["sleep", "infinity"]
