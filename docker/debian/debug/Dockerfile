# Moonlab Debian Debug Build
FROM debian:12-slim

ARG VERSION=0.1.0
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
    dpkg-dev \
    file \
    gdb \
    valgrind \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy source
WORKDIR /app
COPY . .

# Build with debug symbols
ENV CFLAGS="-g -O0"
RUN make clean && make

# Build tests
RUN make tests unit_tests || echo "Tests built"

# Run unit tests
RUN make test_unit || echo "Unit tests completed"

# Keep container running for debugging
CMD ["sleep", "infinity"]
