"""
Pytest configuration and fixtures for Moonlab Python tests.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add the parent directory to path so moonlab can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Test Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "ml: marks tests requiring ML dependencies"
    )


# =============================================================================
# Fixtures - Quantum States
# =============================================================================

@pytest.fixture
def single_qubit_state():
    """Create a fresh single-qubit state |0>."""
    from moonlab import QuantumState
    return QuantumState(1)


@pytest.fixture
def two_qubit_state():
    """Create a fresh two-qubit state |00>."""
    from moonlab import QuantumState
    return QuantumState(2)


@pytest.fixture
def three_qubit_state():
    """Create a fresh three-qubit state |000>."""
    from moonlab import QuantumState
    return QuantumState(3)


@pytest.fixture
def four_qubit_state():
    """Create a fresh four-qubit state |0000>."""
    from moonlab import QuantumState
    return QuantumState(4)


@pytest.fixture
def bell_state():
    """Create a Bell state (|00> + |11>) / sqrt(2)."""
    from moonlab import QuantumState
    state = QuantumState(2)
    state.h(0).cnot(0, 1)
    return state


@pytest.fixture
def ghz_state():
    """Create a 3-qubit GHZ state (|000> + |111>) / sqrt(2)."""
    from moonlab import QuantumState
    state = QuantumState(3)
    state.h(0).cnot(0, 1).cnot(1, 2)
    return state


@pytest.fixture
def superposition_state():
    """Create an equal superposition over 2 qubits."""
    from moonlab import QuantumState
    state = QuantumState(2)
    state.h(0).h(1)
    return state


# =============================================================================
# Fixtures - Test Data
# =============================================================================

@pytest.fixture
def random_angles():
    """Generate random rotation angles for gate tests."""
    np.random.seed(42)
    return np.random.uniform(0, 2 * np.pi, size=10)


@pytest.fixture
def sample_features():
    """Sample feature vectors for encoding tests."""
    return np.array([
        [0.5, 1.0, 1.5, 2.0],
        [0.3, 0.6, 0.9, 1.2],
        [0.1, 0.2, 0.3, 0.4],
        [1.0, 1.5, 2.0, 2.5],
    ])


@pytest.fixture
def xor_dataset():
    """XOR-like dataset for classification tests."""
    X = np.array([
        [0.1, 0.1],
        [0.1, 0.9],
        [0.9, 0.1],
        [0.9, 0.9],
        [0.2, 0.2],
        [0.2, 0.8],
        [0.8, 0.2],
        [0.8, 0.8],
    ])
    y = np.array([1, -1, -1, 1, 1, -1, -1, 1])
    return X, y


# =============================================================================
# Utility Functions
# =============================================================================

@pytest.fixture
def assert_probability():
    """Fixture providing probability assertion helper."""
    def _assert_prob(state, basis_state, expected, tolerance=0.01):
        """Assert that probability of basis_state is close to expected."""
        actual = state.probability(basis_state)
        assert abs(actual - expected) < tolerance, (
            f"Probability mismatch for |{basis_state:b}>: "
            f"expected {expected:.4f}, got {actual:.4f}"
        )
    return _assert_prob


@pytest.fixture
def assert_normalized():
    """Fixture providing normalization assertion helper."""
    def _assert_normalized(state, tolerance=1e-10):
        """Assert that state probabilities sum to 1."""
        probs = state.probabilities()
        total = np.sum(probs)
        assert abs(total - 1.0) < tolerance, (
            f"State not normalized: probabilities sum to {total:.10f}"
        )
    return _assert_normalized


@pytest.fixture
def assert_statevector_close():
    """Fixture providing statevector comparison helper."""
    def _assert_close(state, expected, tolerance=1e-6):
        """Assert that statevector is close to expected (up to global phase)."""
        actual = state.get_statevector()

        # Find global phase by comparing first non-zero elements
        for i in range(len(expected)):
            if abs(expected[i]) > 1e-10:
                phase = actual[i] / expected[i]
                break
        else:
            # All zeros - just compare directly
            phase = 1.0

        adjusted = actual / phase if abs(phase) > 1e-10 else actual

        assert np.allclose(adjusted, expected, atol=tolerance), (
            f"Statevector mismatch:\nExpected: {expected}\nActual: {actual}"
        )
    return _assert_close


# =============================================================================
# Conditional Fixtures
# =============================================================================

@pytest.fixture
def torch_available():
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        pytest.skip("PyTorch not available")
        return False


@pytest.fixture
def tensorflow_available():
    """Check if TensorFlow is available."""
    try:
        import tensorflow
        return True
    except ImportError:
        pytest.skip("TensorFlow not available")
        return False
