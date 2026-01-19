# Quantum Random Number Generator

Generate cryptographically secure random numbers using quantum measurements.

## Overview

Quantum Random Number Generators (QRNGs) exploit the fundamental randomness of quantum mechanics to produce truly random numbers. Unlike classical pseudo-random number generators (PRNGs), quantum randomness is:

- **Truly unpredictable**: Based on quantum measurement outcomes
- **Provably random**: Grounded in physical principles
- **Device-independent**: Can be certified via Bell tests

Moonlab provides a multi-source entropy system that can incorporate quantum randomness.

## The Physics

### Quantum Randomness

When measuring a qubit in superposition:

$$|\psi\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$$

The outcome is fundamentally random:
- $P(0) = P(1) = 0.5$
- No hidden variable determines the result
- Each measurement is independent

### Von Neumann Extraction

To remove bias from imperfect measurements, use von Neumann extraction:

1. Generate pairs of random bits
2. If pair is `01` → output `0`
3. If pair is `10` → output `1`
4. If pair is `00` or `11` → discard

This produces unbiased output even from biased sources.

## Moonlab Entropy System

### Source Hierarchy

Moonlab's entropy system uses multiple sources:

```
┌──────────────────────────────────────────────────────────┐
│                     Entropy Pool                         │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Hardware   │  │      OS      │  │    Jitter    │    │
│  │     RNG      │  │   Entropy    │  │   Entropy    │    │
│  │  (RDRAND/    │  │  (/dev/      │  │  (timing     │    │
│  │   RNDR)      │  │  urandom)    │  │  variations) │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                 │                 │            │
│         └─────────────────┼─────────────────┘            │
│                           │                              │
│                    ┌──────┴───────┐                      │
│                    │   Mixing     │                      │
│                    │   Function   │                      │
│                    │  (XorShift+  │                      │
│                    │  SplitMix64) │                      │
│                    └──────┬───────┘                      │
│                           │                              │
│                    ┌──────┴───────┐                      │
│                    │   Output     │                      │
│                    │   Pool       │                      │
│                    └──────────────┘                      │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Source Types

| Source | Platform | Description |
|--------|----------|-------------|
| Hardware RNG | x86 (RDRAND/RDSEED) | Intel/AMD hardware random |
| Hardware RNG | ARM64 (RNDR) | Apple Silicon hardware random |
| OS Entropy | macOS (SecRandomCopyBytes) | System entropy pool |
| OS Entropy | Linux (/dev/urandom) | Kernel entropy pool |
| OS Entropy | Windows (BCryptGenRandom) | CNG entropy |
| Jitter | All | CPU timing variations |
| Quantum | Moonlab | Quantum measurement outcomes |

## C Implementation

### Basic Usage

```c
#include "utils/entropy.h"
#include <stdio.h>

int main(void) {
    // Create entropy context
    entropy_ctx_t* ctx = entropy_create();

    // Generate random bytes
    uint8_t buffer[32];
    entropy_bytes(ctx, buffer, sizeof(buffer));

    printf("Random bytes: ");
    for (int i = 0; i < 32; i++) {
        printf("%02x", buffer[i]);
    }
    printf("\n");

    // Generate random numbers
    uint64_t random_u64 = entropy_uint64(ctx);
    double random_double = entropy_double(ctx);  // [0, 1)

    printf("Random uint64: %llu\n", (unsigned long long)random_u64);
    printf("Random double: %.15f\n", random_double);

    entropy_destroy(ctx);
    return 0;
}
```

### Quantum-Enhanced Entropy

```c
#include "quantum/state.h"
#include "quantum/measurement.h"
#include "utils/entropy.h"
#include <stdio.h>

/**
 * @brief Generate random bits from quantum measurements
 *
 * Uses Hadamard gates to create superposition, then measures.
 * Each measurement produces one truly random bit.
 */
void quantum_random_bits(entropy_ctx_t* classical_entropy,
                         uint8_t* output, size_t num_bytes) {
    quantum_state_t state;
    quantum_entropy_ctx_t qe;
    quantum_entropy_init(&qe, entropy_callback, classical_entropy);

    size_t bits_needed = num_bytes * 8;
    size_t bits_collected = 0;

    while (bits_collected < bits_needed) {
        // Use 8 qubits per batch
        int num_qubits = 8;
        quantum_state_init(&state, num_qubits);

        // Create superposition on all qubits
        for (int q = 0; q < num_qubits; q++) {
            gate_hadamard(&state, q);
        }

        // Measure all qubits
        uint64_t measurement = quantum_measure_all(&state, &qe);

        // Extract bits
        for (int b = 0; b < num_qubits && bits_collected < bits_needed; b++) {
            size_t byte_idx = bits_collected / 8;
            size_t bit_idx = bits_collected % 8;

            if (bit_idx == 0) {
                output[byte_idx] = 0;
            }

            if (measurement & (1ULL << b)) {
                output[byte_idx] |= (1 << bit_idx);
            }

            bits_collected++;
        }

        quantum_state_free(&state);
    }
}

int main(void) {
    printf("=== Quantum Random Number Generator ===\n\n");

    entropy_ctx_t* entropy = entropy_create();

    // Generate 256 bits (32 bytes) of quantum randomness
    uint8_t quantum_random[32];
    quantum_random_bits(entropy, quantum_random, 32);

    printf("Quantum random bytes:\n");
    for (int i = 0; i < 32; i++) {
        printf("%02x", quantum_random[i]);
        if ((i + 1) % 16 == 0) printf("\n");
    }

    entropy_destroy(entropy);
    return 0;
}
```

### Quality Assessment

```c
#include "utils/entropy.h"
#include <stdio.h>

void assess_entropy_quality(void) {
    entropy_ctx_t* ctx = entropy_create();

    // Generate sample data
    size_t sample_size = 10000;
    uint8_t* sample = malloc(sample_size);
    entropy_bytes(ctx, sample, sample_size);

    // Assess quality
    entropy_quality_t quality;
    int passed = entropy_test_data(sample, sample_size, &quality);

    printf("Entropy Quality Assessment:\n");
    printf("  Shannon entropy: %.4f bits/byte (max 8.0)\n",
           quality.estimated_entropy);
    printf("  Chi-squared: %.2f (expected: 210-302)\n",
           quality.chi_squared);
    printf("  Compression ratio: %.4f\n", quality.compression_ratio);
    printf("  Hardware RNG: %s\n",
           quality.hardware_available ? "available" : "not available");
    printf("  Basic tests: %s\n", quality.passed_basic_tests ? "PASSED" : "FAILED");
    printf("  Overall: %s\n", passed ? "PASS" : "FAIL");

    free(sample);
    entropy_destroy(ctx);
}
```

## Python Implementation

```python
import moonlab as ml
import numpy as np
from collections import Counter

class QuantumRNG:
    """Quantum Random Number Generator using Moonlab."""

    def __init__(self, qubits_per_sample: int = 8):
        """Initialize QRNG with specified qubits per measurement."""
        self.qubits = qubits_per_sample
        self.bits_generated = 0

    def random_bits(self, num_bits: int) -> list:
        """Generate random bits from quantum measurements."""
        bits = []

        while len(bits) < num_bits:
            # Create superposition state
            state = ml.QuantumState(self.qubits)
            for q in range(self.qubits):
                state.h(q)

            # Measure all qubits
            result = state.measure_all()

            # Extract bits
            for q in range(self.qubits):
                if len(bits) < num_bits:
                    bits.append((result >> q) & 1)

        self.bits_generated += num_bits
        return bits

    def random_bytes(self, num_bytes: int) -> bytes:
        """Generate random bytes."""
        bits = self.random_bits(num_bytes * 8)
        result = bytearray(num_bytes)

        for i, bit in enumerate(bits):
            byte_idx = i // 8
            bit_idx = i % 8
            result[byte_idx] |= (bit << bit_idx)

        return bytes(result)

    def random_int(self, low: int, high: int) -> int:
        """Generate random integer in [low, high)."""
        range_size = high - low
        bits_needed = range_size.bit_length()

        while True:
            bits = self.random_bits(bits_needed)
            value = sum(b << i for i, b in enumerate(bits))
            if value < range_size:
                return low + value

    def random_float(self) -> float:
        """Generate random float in [0, 1)."""
        bits = self.random_bits(53)  # Double precision mantissa
        value = sum(b << i for i, b in enumerate(bits))
        return value / (2**53)


def test_randomness(qrng: QuantumRNG, num_samples: int = 10000):
    """Statistical tests for randomness quality."""
    print("=== Randomness Quality Tests ===\n")

    # Generate samples
    bits = qrng.random_bits(num_samples)

    # Test 1: Frequency (monobit) test
    ones = sum(bits)
    zeros = num_samples - ones
    freq_ratio = ones / num_samples

    print(f"Frequency Test:")
    print(f"  Ones: {ones} ({100*freq_ratio:.2f}%)")
    print(f"  Zeros: {zeros} ({100*(1-freq_ratio):.2f}%)")
    print(f"  Expected: 50.00%")
    print(f"  Deviation: {abs(freq_ratio - 0.5)*100:.2f}%")
    print(f"  Status: {'PASS' if abs(freq_ratio - 0.5) < 0.02 else 'FAIL'}")

    # Test 2: Runs test
    runs = 1
    for i in range(1, len(bits)):
        if bits[i] != bits[i-1]:
            runs += 1

    expected_runs = (2 * ones * zeros) / num_samples + 1
    print(f"\nRuns Test:")
    print(f"  Runs: {runs}")
    print(f"  Expected: {expected_runs:.0f}")
    print(f"  Status: {'PASS' if abs(runs - expected_runs) < expected_runs * 0.1 else 'FAIL'}")

    # Test 3: Autocorrelation (lag-1)
    correlation = sum(bits[i] == bits[i+1] for i in range(len(bits)-1))
    correlation /= (len(bits) - 1)

    print(f"\nAutocorrelation Test (lag=1):")
    print(f"  Correlation: {correlation:.4f}")
    print(f"  Expected: 0.5000")
    print(f"  Status: {'PASS' if abs(correlation - 0.5) < 0.02 else 'FAIL'}")

    # Test 4: Chi-squared on bytes
    bytes_data = qrng.random_bytes(num_samples // 8)
    byte_counts = Counter(bytes_data)

    expected = (num_samples // 8) / 256
    chi_sq = sum((count - expected)**2 / expected
                 for count in byte_counts.values())

    print(f"\nChi-Squared Test (bytes):")
    print(f"  Chi-squared: {chi_sq:.2f}")
    print(f"  Expected range: 210-302 (for 255 DOF)")
    print(f"  Status: {'PASS' if 210 < chi_sq < 302 else 'FAIL'}")


# Demo
if __name__ == "__main__":
    print("=== Quantum Random Number Generator Demo ===\n")

    qrng = QuantumRNG(qubits_per_sample=8)

    # Generate random data
    print("Generating 32 random bytes...")
    random_bytes = qrng.random_bytes(32)
    print("Hex:", random_bytes.hex())

    print("\nGenerating 10 random integers [0, 100)...")
    for _ in range(10):
        print(f"  {qrng.random_int(0, 100)}", end="")
    print()

    print("\nGenerating 5 random floats [0, 1)...")
    for _ in range(5):
        print(f"  {qrng.random_float():.6f}", end="")
    print()

    print()
    test_randomness(qrng)
```

## Entropy Mixing

### XorShift+ Algorithm

Moonlab uses XorShift128+ for fast entropy extraction:

```c
static uint64_t xorshift128plus(uint64_t* s) {
    uint64_t x = s[0];
    uint64_t y = s[1];
    s[0] = y;
    x ^= x << 23;
    s[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
    return s[1] + y;
}
```

### SplitMix64 for Seeding

```c
static uint64_t splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
```

## Use Cases

### Cryptographic Key Generation

```c
// Generate 256-bit encryption key
uint8_t key[32];
entropy_ctx_t* ctx = entropy_create();
entropy_bytes(ctx, key, 32);
entropy_destroy(ctx);

// Use key for encryption...
```

### Monte Carlo Simulation

```python
qrng = QuantumRNG()

# Estimate π using Monte Carlo
inside = 0
total = 100000

for _ in range(total):
    x = qrng.random_float()
    y = qrng.random_float()
    if x*x + y*y < 1:
        inside += 1

pi_estimate = 4 * inside / total
print(f"π ≈ {pi_estimate:.6f}")
```

### Lottery/Gaming

```python
# Fair dice roll
def roll_dice() -> int:
    return qrng.random_int(1, 7)

# Shuffle deck
def shuffle_deck(deck: list) -> list:
    shuffled = deck.copy()
    for i in range(len(shuffled) - 1, 0, -1):
        j = qrng.random_int(0, i + 1)
        shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
    return shuffled
```

## Performance

| Method | Speed (MB/s) | Quality | Use Case |
|--------|--------------|---------|----------|
| Hardware RNG (RDRAND) | 500+ | Excellent | General purpose |
| OS Entropy | 100+ | Excellent | Cryptography |
| Quantum (simulated) | 10-50 | Excellent | Research |
| Jitter | 0.1-1 | Good | Fallback |

## Running the Example

```bash
# Build
make examples

# Run entropy test
./examples/applications/quantum_rng

# Run with specific size
./examples/applications/quantum_rng --bytes 1024 --test
```

## See Also

- [API: entropy.h](../../api/c/entropy.md)
- [Concepts: Measurement Theory](../../concepts/measurement-theory.md)
- [Guide: Debugging](../../guides/debugging.md)
