# Archived Moonlab Documentation: Python Machine Learning API

This local Moonlab document is retained as archived vendor text for the QGTL integration audit; current supported claims are measured by `scripts/moonlab_doc_claim_audit.py` and grounded against `external/moonlab/README.md`, `external/moonlab/CMakeLists.txt`, and `docs/MOONLAB_OPEN_CORE_INTEGRATION.md`.

The historical text below is preserved as an archival snapshot, not as current release documentation.

```text
# Python Machine Learning API

Complete reference for quantum machine learning in the Python API.

**Module**: `moonlab.ml`

## Overview

The ML module provides quantum machine learning tools including:

- **Quantum Feature Maps**: Encode classical data in quantum states
- **Quantum Kernels**: Kernel methods with exponential feature spaces
- **QSVM**: Quantum Support Vector Machine
- **Quantum PCA**: Principal component analysis with quantum speedup
- **Quantum Autoencoder**: Data compression via quantum circuits

## Quantum Feature Maps

Map classical data $x \in \mathbb{R}^n$ to quantum states $|\phi(x)\rangle$.

### AngleEncoding

Simple rotation-based encoding.

[archived fence delimiter: ```python]
AngleEncoding(num_qubits: int)
[archived fence delimiter: ```]

**Formula**: $|\phi(x)\rangle = \prod_i R_Y(x_i)|0\rangle$

**Properties**:
- Linear encoding
- Works well for normalized data
- Efficient circuit depth: O(n)

**Example**:
[archived fence delimiter: ```python]
from moonlab.ml import AngleEncoding
from moonlab import QuantumState
import numpy as np

encoding = AngleEncoding(num_qubits=4)
state = QuantumState(4)

x = np.array([0.5, -0.3, 0.8, 0.1])
encoding.encode(x, state)
[archived fence delimiter: ```]

### AmplitudeEncoding

Encode data directly in quantum amplitudes.

[archived fence delimiter: ```python]
AmplitudeEncoding(num_qubits: int)
[archived fence delimiter: ```]

**Formula**: $|\phi(x)\rangle = \sum_i \frac{x_i}{\|x\|} |i\rangle$

**Properties**:
- Exponential compression: n features → log₂(n) qubits
- Requires data normalization
- Most efficient use of quantum resources

**Example**:
[archived fence delimiter: ```python]
from moonlab.ml import AmplitudeEncoding
from moonlab import QuantumState
import numpy as np

# Encode 16 features in 4 qubits
encoding = AmplitudeEncoding(num_qubits=4)
state = QuantumState(4)

x = np.random.randn(16)
encoding.encode(x, state)

# Verify encoding
sv = state.get_statevector()
print(np.allclose(np.abs(sv)**2, (x/np.linalg.norm(x))**2))
[archived fence delimiter: ```]

### IQPEncoding

Instantaneous Quantum Polynomial encoding with entanglement.

[archived fence delimiter: ```python]
IQPEncoding(num_qubits: int, num_layers: int = 2)
[archived fence delimiter: ```]

**Formula**: $|\phi(x)\rangle = \exp\left(i \sum_{ij} x_i x_j Z_i Z_j\right) H^{\otimes n}|0\rangle^n$

**Properties**:
- Creates entanglement proportional to feature products
- Induces quantum kernel for SVM
- Provable quantum advantage for some distributions

**Example**:
[archived fence delimiter: ```python]
from moonlab.ml import IQPEncoding
from moonlab import QuantumState
import numpy as np

encoding = IQPEncoding(num_qubits=4, num_layers=2)
state = QuantumState(4)

x = np.array([0.5, -0.3, 0.8, 0.1])
encoding.encode(x, state)
[archived fence delimiter: ```]

## Quantum Kernel

Compute kernel using quantum feature map overlaps.

[archived fence delimiter: ```python]
QuantumKernel(feature_map: QuantumFeatureMap)
[archived fence delimiter: ```]

**Kernel Formula**: $K(x, x') = |\langle\phi(x)|\phi(x')\rangle|^2$

### Methods

#### compute

[archived fence delimiter: ```python]
compute(x1: np.ndarray, x2: np.ndarray) -> float
[archived fence delimiter: ```]

Compute kernel value between two data points.

#### compute_matrix

[archived fence delimiter: ```python]
compute_matrix(X: np.ndarray) -> np.ndarray
[archived fence delimiter: ```]

Compute full kernel matrix for dataset.

**Example**:
[archived fence delimiter: ```python]
from moonlab.ml import QuantumKernel, IQPEncoding
import numpy as np

# Create kernel
feature_map = IQPEncoding(num_qubits=4)
kernel = QuantumKernel(feature_map)

# Compute kernel value
x1 = np.array([0.5, 0.3, -0.2, 0.1])
x2 = np.array([0.4, 0.35, -0.15, 0.2])
k_val = kernel.compute(x1, x2)
print(f"K(x1, x2) = {k_val:.4f}")

# Compute kernel matrix
X = np.random.randn(20, 4)
K = kernel.compute_matrix(X)
print(f"Kernel matrix shape: {K.shape}")
[archived fence delimiter: ```]

## QSVM (Quantum Support Vector Machine)

Quantum-enhanced SVM using quantum kernel.

[archived fence delimiter: ```python]
QSVM(
    num_qubits: int = 4,
    feature_map: str = 'iqp',
    C: float = 1.0
)
[archived fence delimiter: ```]

**Parameters**:
- `num_qubits`: Qubits for quantum feature map
- `feature_map`: Type ('angle', 'amplitude', 'iqp')
- `C`: SVM regularization parameter

### Methods

#### fit

[archived fence delimiter: ```python]
fit(X: np.ndarray, y: np.ndarray) -> None
[archived fence delimiter: ```]

Train QSVM on dataset.

**Parameters**:
- `X`: Training features (n_samples, n_features)
- `y`: Training labels (-1 or +1)

#### predict

[archived fence delimiter: ```python]
predict(X: np.ndarray) -> np.ndarray
[archived fence delimiter: ```]

Predict labels for new data.

#### score

[archived fence delimiter: ```python]
score(X: np.ndarray, y: np.ndarray) -> float
[archived fence delimiter: ```]

Compute accuracy on dataset.

### Complete Example

[archived fence delimiter: ```python]
from moonlab.ml import QSVM
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

# Generate dataset
X, y = make_moons(n_samples=100, noise=0.1)
y = 2 * y - 1  # Convert to {-1, +1}

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train QSVM
qsvm = QSVM(num_qubits=4, feature_map='iqp', C=1.0)
qsvm.fit(X_train, y_train)

# Evaluate
train_acc = qsvm.score(X_train, y_train)
test_acc = qsvm.score(X_test, y_test)

print(f"Train accuracy: {train_acc:.2%}")
print(f"Test accuracy: {test_acc:.2%}")
print(f"Support vectors: {len(qsvm.alphas)}")
[archived fence delimiter: ```]

## Quantum PCA

Principal component analysis with quantum amplitude amplification.

[archived fence delimiter: ```python]
QuantumPCA(num_components: int, num_qubits: int = None)
[archived fence delimiter: ```]

**Parameters**:
- `num_components`: Number of principal components
- `num_qubits`: Qubits for quantum algorithm (default: log₂(features))

### Methods

#### fit

[archived fence delimiter: ```python]
fit(X: np.ndarray) -> None
[archived fence delimiter: ```]

Fit Quantum PCA to data.

#### transform

[archived fence delimiter: ```python]
transform(X: np.ndarray) -> np.ndarray
[archived fence delimiter: ```]

Project data onto principal components.

#### fit_transform

[archived fence delimiter: ```python]
fit_transform(X: np.ndarray) -> np.ndarray
[archived fence delimiter: ```]

Fit and transform in one step.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `components_` | np.ndarray | Principal component vectors |
| `explained_variance_` | np.ndarray | Variance explained by each component |

### Example

[archived fence delimiter: ```python]
from moonlab.ml import QuantumPCA
import numpy as np
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X = iris.data

# Fit Quantum PCA
qpca = QuantumPCA(num_components=2)
X_reduced = qpca.fit_transform(X)

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_reduced.shape}")
print(f"Explained variance: {qpca.explained_variance_}")
[archived fence delimiter: ```]

## Variational Circuit

General-purpose parameterized quantum circuit for ML.

[archived fence delimiter: ```python]
VariationalCircuit(num_qubits: int, num_layers: int = 2)
[archived fence delimiter: ```]

### Methods

#### forward

[archived fence delimiter: ```python]
forward(state: QuantumState) -> QuantumState
[archived fence delimiter: ```]

Apply variational circuit to quantum state.

### Example

[archived fence delimiter: ```python]
from moonlab.ml import VariationalCircuit
from moonlab import QuantumState
import torch

circuit = VariationalCircuit(num_qubits=4, num_layers=3)

state = QuantumState(4)
for q in range(4):
    state.h(q)  # Initialize superposition

circuit(state)  # Apply variational circuit

print(f"Parameters: {len(circuit.params)}")
[archived fence delimiter: ```]

## Quantum Autoencoder

Quantum autoencoder for data compression.

[archived fence delimiter: ```python]
QuantumAutoencoder(input_dim: int, latent_qubits: int)
[archived fence delimiter: ```]

**Parameters**:
- `input_dim`: Input feature dimension
- `latent_qubits`: Number of qubits in latent space

### Methods

#### encode

[archived fence delimiter: ```python]
encode(x: torch.Tensor) -> torch.Tensor
[archived fence delimiter: ```]

Encode to latent space.

#### decode

[archived fence delimiter: ```python]
decode(latent: torch.Tensor) -> torch.Tensor
[archived fence delimiter: ```]

Decode from latent space.

#### forward

[archived fence delimiter: ```python]
forward(x: torch.Tensor) -> torch.Tensor
[archived fence delimiter: ```]

Full autoencoder forward pass.

### Example

[archived fence delimiter: ```python]
from moonlab.ml import QuantumAutoencoder
import torch

# Create autoencoder: 16 → 4 qubits → 16
autoencoder = QuantumAutoencoder(input_dim=16, latent_qubits=4)

# Encode/decode
x = torch.randn(10, 16)  # Batch of 10 samples
reconstruction = autoencoder(x)

# Training
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

for epoch in range(100):
    optimizer.zero_grad()
    output = autoencoder(x)
    loss = criterion(output, x)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
[archived fence delimiter: ```]

## Utility Functions

### quantum_kernel_matrix

[archived fence delimiter: ```python]
quantum_kernel_matrix(
    X: np.ndarray,
    num_qubits: int = 4,
    feature_map: str = 'iqp'
) -> np.ndarray
[archived fence delimiter: ```]

Convenience function to compute quantum kernel matrix.

[archived fence delimiter: ```python]
from moonlab.ml import quantum_kernel_matrix
import numpy as np

X = np.random.randn(30, 4)
K = quantum_kernel_matrix(X, num_qubits=4, feature_map='iqp')
[archived fence delimiter: ```]

### train_qsvm

[archived fence delimiter: ```python]
train_qsvm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_qubits: int = 4,
    feature_map: str = 'iqp'
) -> dict
[archived fence delimiter: ```]

Complete QSVM training pipeline with metrics.

[archived fence delimiter: ```python]
from moonlab.ml import train_qsvm
import numpy as np

# Generate data
X_train = np.random.randn(50, 4)
y_train = np.sign(X_train[:, 0] + X_train[:, 1])
X_test = np.random.randn(20, 4)
y_test = np.sign(X_test[:, 0] + X_test[:, 1])

# Train and evaluate
results = train_qsvm(X_train, y_train, X_test, y_test, num_qubits=4)

print(f"Train accuracy: {results['train_accuracy']:.2%}")
print(f"Test accuracy: {results['test_accuracy']:.2%}")
print(f"Training time: {results['train_time']:.2f}s")
[archived fence delimiter: ```]

## Comparison: Classical vs Quantum

| Aspect | Classical SVM | Quantum SVM |
|--------|---------------|-------------|
| Feature space | Explicit kernel | $2^n$ dimensional |
| Kernel computation | O(n²) | O(n² × circuit) |
| Entanglement | None | Full connectivity |
| Advantage regime | Large datasets | Small, structured data |

## Performance Tips

1. **Feature scaling**: Normalize inputs to [-1, 1] or [0, 1]
2. **Qubit count**: Start with 4-6 qubits, increase as needed
3. **IQP vs Angle**: IQP better for complex correlations
4. **QSVM datasets**: Works best for small datasets (< 1000 samples)
5. **Kernel caching**: Precompute kernel matrix when possible

## References

1. Havlíček et al., "Supervised learning with quantum-enhanced feature spaces", Nature 567, 209-212 (2019)
2. Schuld & Killoran, "Quantum Machine Learning in Feature Hilbert Spaces", Phys. Rev. Lett. 122, 040504 (2019)
3. Lloyd et al., "Quantum principal component analysis", Nat. Phys. 10, 631-633 (2014)

## See Also

- [PyTorch Integration](torch-layer.md) - Quantum neural network layers
- [Core API](core.md) - QuantumState, Gates
- [Algorithms API](algorithms.md) - VQE, QAOA
```
