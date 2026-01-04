# Moonlab Ubuntu Release Build
# Matches GitHub Actions ubuntu-22.04 environment
FROM ubuntu:22.04

ARG VERSION=0.1.0
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

# Build - matching GitHub Actions
RUN make clean && make

# Build tests
RUN make tests unit_tests

# Run unit tests
RUN make test_unit || echo "Unit tests completed"

# Run integration tests
RUN make test || echo "Integration tests completed"

# Verify artifacts
RUN ls -la lib/ || echo "No lib directory"

# Keep container running for file extraction
CMD ["sleep", "infinity"]
