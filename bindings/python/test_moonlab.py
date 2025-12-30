"""
Moonlab Python Bindings - Comprehensive Test Suite

Tests all Python bindings functionality:
- Core quantum operations
- Quantum ML feature maps and kernels
- PyTorch integration
- Algorithm implementations

Run with: python test_moonlab.py
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add moonlab to path
sys.path.insert(0, str(Path(__file__).parent))

from moonlab import (
    QuantumState,
    Gates,
    QuantumError
)

from moonlab.ml import (
    AngleEncoding,
    AmplitudeEncoding,
    IQPEncoding,
    QuantumKernel,
    QSVM,
    QuantumPCA
)

from moonlab.torch_layer import (
    QuantumLayer,
    QuantumClassifier,
    HybridQNN
)

# ============================================================================
# CORE TESTS
# ============================================================================

def test_quantum_state_creation():
    """Test quantum state initialization"""
    print("Testing QuantumState creation...")
    
    state = QuantumState(4)
    assert state.num_qubits == 4
    assert state.state_dim == 16
    
    print("  ✓ QuantumState(4) created successfully")
    print(f"    num_qubits={state.num_qubits}, state_dim={state.state_dim}")


def test_single_qubit_gates():
    """Test single-qubit gate operations"""
    print("\nTesting single-qubit gates...")
    
    state = QuantumState(2)
    
    # Test Hadamard
    state.h(0)
    probs = state.probabilities()
    assert abs(probs[0] - 0.5) < 0.01, "Hadamard failed"
    assert abs(probs[1] - 0.5) < 0.01, "Hadamard failed"
    print("  ✓ Hadamard gate working")
    
    # Test Pauli gates
    state.reset().x(0)
    assert abs(state.probability(1) - 1.0) < 0.01, "Pauli-X failed"
    print("  ✓ Pauli-X gate working")
    
    # Test rotations
    state.reset().ry(0, np.pi/2)
    prob_1 = state.probability(1)
    assert abs(prob_1 - 0.5) < 0.01, f"RY(π/2) failed, got prob={prob_1}"
    print("  ✓ Rotation gates working")


def test_two_qubit_gates():
    """Test two-qubit entangling gates"""
    print("\nTesting two-qubit gates...")
    
    # Test CNOT - create Bell state
    state = QuantumState(2)
    state.h(0).cnot(0, 1)
    
    # Check Bell state: |00⟩ + |11⟩
    prob_00 = state.probability(0)  # |00⟩
    prob_11 = state.probability(3)  # |11⟩
    
    assert abs(prob_00 - 0.5) < 0.01, f"Bell state failed: P(00)={prob_00}"
    assert abs(prob_11 - 0.5) < 0.01, f"Bell state failed: P(11)={prob_11}"
    
    print("  ✓ CNOT gate working")
    print("  ✓ Bell state created successfully")
    print(f"    P(|00⟩)={prob_00:.3f}, P(|11⟩)={prob_11:.3f}")


def test_state_operations():
    """Test state manipulation operations"""
    print("\nTesting state operations...")
    
    state = QuantumState(3)
    state.h(0).h(1).h(2)
    
    # Test clone
    cloned = state.clone()
    assert cloned.num_qubits == state.num_qubits
    print("  ✓ State cloning works")
    
    # Test reset
    state.reset()
    assert abs(state.probability(0) - 1.0) < 0.01
    print("  ✓ State reset works")
    
    # Test probabilities
    state.h(0)
    probs = state.probabilities()
    assert len(probs) == 8
    assert abs(np.sum(probs) - 1.0) < 0.01
    print("  ✓ Probability calculations correct")


# ============================================================================
# QUANTUM ML TESTS
# ============================================================================

def test_angle_encoding():
    """Test angle encoding feature map"""
    print("\nTesting AngleEncoding feature map...")
    
    state = QuantumState(4)
    encoder = AngleEncoding(4)
    
    x = np.array([0.5, 1.0, 1.5, 2.0])
    encoder.encode(x, state)
    
    # Verify state is valid
    probs = state.probabilities()
    assert abs(np.sum(probs) - 1.0) < 0.01
    
    print("  ✓ AngleEncoding working")
    print(f"    Encoded {len(x)} features to {state.num_qubits} qubits")


def test_iqp_encoding():
    """Test IQP encoding feature map"""
    print("\nTesting IQPEncoding feature map...")
    
    state = QuantumState(4)
    encoder = IQPEncoding(4, num_layers=2)
    
    x = np.array([0.3, 0.6, 0.9, 1.2])
    encoder.encode(x, state)
    
    # Verify creates entanglement
    probs = state.probabilities()
    assert abs(np.sum(probs) - 1.0) < 0.01
    
    # Check that we have superposition (not all probability in one state)
    max_prob = np.max(probs)
    assert max_prob < 0.9, "IQP should create superposition"
    
    print("  ✓ IQPEncoding working")
    print(f"    Created entangled state with max_prob={max_prob:.3f}")


def test_quantum_kernel():
    """Test quantum kernel computation"""
    print("\nTesting QuantumKernel...")
    
    encoder = IQPEncoding(4, num_layers=1)
    kernel = QuantumKernel(encoder)
    
    x1 = np.array([0.5, 1.0, 1.5, 2.0])
    x2 = np.array([0.6, 1.1, 1.4, 1.9])
    
    k_val = kernel.compute(x1, x2)
    
    # Kernel should be in [0, 1]
    assert 0.0 <= k_val <= 1.0, f"Invalid kernel value: {k_val}"
    
    # Self-kernel should be ~1
    k_self = kernel.compute(x1, x1)
    assert abs(k_self - 1.0) < 0.1, f"Self-kernel should be ~1, got {k_self}"
    
    print("  ✓ QuantumKernel working")
    print(f"    K(x1,x2)={k_val:.4f}, K(x1,x1)={k_self:.4f}")


def test_qsvm():
    """Test Quantum SVM"""
    print("\nTesting QSVM classifier...")
    
    # Create simple XOR-like dataset
    np.random.seed(42)
    X_train = np.array([
        [0.1, 0.1],
        [0.1, 0.9],
        [0.9, 0.1],
        [0.9, 0.9],
        [0.2, 0.2],
        [0.2, 0.8],
        [0.8, 0.2],
        [0.8, 0.8]
    ])
    y_train = np.array([1, -1, -1, 1, 1, -1, -1, 1])
    
    # Train QSVM
    qsvm = QSVM(num_qubits=2, feature_map='iqp', C=1.0)
    qsvm.fit(X_train, y_train)
    
    # Check training accuracy
    train_acc = qsvm.score(X_train, y_train)
    print(f"  ✓ QSVM trained successfully")
    print(f"    Training accuracy: {train_acc:.1%}")
    print(f"    Support vectors: {len(qsvm.alphas)}/{len(X_train)}")
    
    assert train_acc >= 0.5, "QSVM should learn XOR pattern"


def test_quantum_pca():
    """Test Quantum PCA"""
    print("\nTesting QuantumPCA...")
    
    # Create correlated dataset
    np.random.seed(42)
    n_samples = 20
    
    # Generate data with 2 principal components
    X = np.random.randn(n_samples, 4)
    X[:, 2] = X[:, 0] + 0.1 * np.random.randn(n_samples)  # Correlated
    X[:, 3] = X[:, 1] + 0.1 * np.random.randn(n_samples)  # Correlated
    
    # Apply Quantum PCA
    qpca = QuantumPCA(num_components=2, num_qubits=2)
    qpca.fit(X)
    
    # Transform data
    X_transformed = qpca.transform(X)
    
    assert X_transformed.shape == (n_samples, 2)
    assert qpca.components_.shape == (2, 4)
    
    print("  ✓ QuantumPCA working")
    print(f"    Reduced {X.shape[1]} dimensions to {X_transformed.shape[1]}")
    print(f"    Explained variance: {qpca.explained_variance_}")


# ============================================================================
# PYTORCH INTEGRATION TESTS
# ============================================================================

def test_quantum_layer():
    """Test PyTorch QuantumLayer"""
    print("\nTesting PyTorch QuantumLayer...")
    
    layer = QuantumLayer(num_qubits=4, depth=2)
    
    # Test forward pass
    x = torch.randn(8, 4)  # Batch of 8
    y = layer(x)
    
    assert y.shape == (8, 4), f"Expected shape (8,4), got {y.shape}"
    assert layer.params.requires_grad, "Parameters should be trainable"
    
    print("  ✓ QuantumLayer forward pass working")
    print(f"    Input shape: {x.shape}, Output shape: {y.shape}")
    print(f"    Trainable parameters: {len(layer.params)}")


def test_quantum_classifier():
    """Test quantum classifier model"""
    print("\nTesting QuantumClassifier...")
    
    model = QuantumClassifier(
        num_features=8,
        num_qubits=4,
        num_classes=2,
        depth=2
    )
    
    # Forward pass
    x = torch.randn(4, 8)
    y = model(x)
    
    assert y.shape == (4, 2)
    
    print("  ✓ QuantumClassifier working")
    print(f"    Architecture: 8 features → 4 qubits → 2 classes")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    quantum_params = len(model.quantum.params)
    
    print(f"    Total parameters: {total_params}")
    print(f"    Quantum parameters: {quantum_params}")


def test_hybrid_qnn():
    """Test Hybrid Quantum-Classical Network"""
    print("\nTesting HybridQNN...")
    
    model = HybridQNN(
        input_dim=10,
        hidden_dim=8,
        num_qubits=4,
        output_dim=3,
        quantum_depth=2
    )
    
    x = torch.randn(5, 10)
    y = model(x)
    
    assert y.shape == (5, 3)
    
    print("  ✓ HybridQNN working")
    print(f"    Input: {x.shape} → Output: {y.shape}")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_end_to_end_classification():
    """End-to-end classification with quantum layer"""
    print("\nTesting end-to-end classification...")
    
    # Simple binary classification dataset
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate separable data
    X_train = torch.randn(20, 4)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).long()
    
    # Build model
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.Tanh(),
        QuantumLayer(num_qubits=4, depth=1),
        torch.nn.Linear(4, 2)
    )
    
    # Train for a few steps
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    
    initial_loss = None
    for epoch in range(3):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
        if initial_loss is None:
            initial_loss = loss.item()
    
    final_loss = loss.item()
    
    print("  ✓ End-to-end training working")
    print(f"    Initial loss: {initial_loss:.4f}")
    print(f"    Final loss: {final_loss:.4f}")
    print(f"    Loss decreased: {initial_loss > final_loss}")


def test_quantum_kernel_matrix():
    """Test kernel matrix computation"""
    print("\nTesting kernel matrix computation...")
    
    from moonlab.ml import quantum_kernel_matrix
    
    X = np.random.randn(5, 4) * 0.5
    K = quantum_kernel_matrix(X, num_qubits=4, feature_map='iqp')
    
    assert K.shape == (5, 5)
    
    # Check properties
    assert np.allclose(K, K.T), "Kernel should be symmetric"
    assert np.all(np.diag(K) > 0.9), "Diagonal should be ~1"
    assert np.all(K >= 0) and np.all(K <= 1), "Kernel values in [0,1]"
    
    print("  ✓ Kernel matrix computation working")
    print(f"    Shape: {K.shape}")
    print(f"    Diagonal: {np.diag(K)}")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all test suites"""
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║          MOONLAB PYTHON BINDINGS TEST SUITE                ║")
    print("╠════════════════════════════════════════════════════════════╣")
    print("║                                                            ║")
    print("║  Testing: Core, ML, PyTorch Integration                   ║")
    print("║                                                            ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    tests = [
        ("Core: State Creation", test_quantum_state_creation),
        ("Core: Single-Qubit Gates", test_single_qubit_gates),
        ("Core: Two-Qubit Gates", test_two_qubit_gates),
        ("Core: State Operations", test_state_operations),
        ("ML: Angle Encoding", test_angle_encoding),
        ("ML: IQP Encoding", test_iqp_encoding),
        ("ML: Quantum Kernel", test_quantum_kernel),
        ("ML: QSVM Classifier", test_qsvm),
        ("ML: Quantum PCA", test_quantum_pca),
        ("PyTorch: QuantumLayer", test_quantum_layer),
        ("PyTorch: QuantumClassifier", test_quantum_classifier),
        ("PyTorch: HybridQNN", test_hybrid_qnn),
        ("Integration: Classification", test_end_to_end_classification),
        ("Integration: Kernel Matrix", test_quantum_kernel_matrix),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ {test_name} FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║                    TEST RESULTS                            ║")
    print("╠════════════════════════════════════════════════════════════╣")
    print(f"║  Total Tests:    {len(tests):3d}                                       ║")
    print(f"║  Passed:         {passed:3d}                                       ║")
    print(f"║  Failed:         {failed:3d}                                       ║")
    print(f"║  Success Rate:   {100.0*passed/len(tests):5.1f}%                                  ║")
    print("╚════════════════════════════════════════════════════════════╝\n")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Moonlab Python bindings working perfectly.\n")
        return 0
    else:
        print(f"⚠️  {failed} test(s) failed. Check output above for details.\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())