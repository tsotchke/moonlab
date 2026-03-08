# Tutorial 10: PyTorch Integration

Build hybrid quantum-classical neural networks using Moonlab's PyTorch integration.

## Prerequisites

- Completed [Tutorial 9: GPU Acceleration](09-gpu-acceleration.md)
- Familiarity with PyTorch fundamentals
- Understanding of gradient-based optimization

## Learning Objectives

By the end of this tutorial, you will:
- Integrate quantum circuits as differentiable PyTorch layers
- Implement the parameter-shift rule for quantum gradients
- Build and train a variational quantum classifier
- Combine classical and quantum layers in hybrid architectures

## Introduction

Quantum machine learning combines the expressive power of quantum circuits with the optimization machinery of classical deep learning. Moonlab's `QuantumLayer` integrates seamlessly with PyTorch's autograd system, enabling end-to-end gradient-based training.

### Why Hybrid Models?

1. **Quantum Feature Maps**: Encode classical data into quantum states
2. **Variational Circuits**: Parameterized quantum circuits as trainable layers
3. **Classical Post-Processing**: Neural networks process quantum measurements
4. **Potential Advantage**: Quantum circuits may express functions more efficiently

## Setup

### Installation

```bash
pip install moonlab torch torchvision
```

### Import Libraries

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from moonlab import QuantumState
from moonlab.torch_layer import QuantumLayer, QuantumCircuit
from moonlab.torch_layer import AngleEncoding, AmplitudeEncoding
```

## Part 1: Quantum Layers as PyTorch Modules

### Creating a Basic Quantum Layer

```python
class SimpleQuantumLayer(nn.Module):
    """A simple parameterized quantum circuit layer."""

    def __init__(self, num_qubits: int, num_layers: int = 2):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        # Trainable parameters: rotation angles
        # 3 rotations per qubit per layer (Rx, Ry, Rz)
        num_params = num_qubits * num_layers * 3
        self.params = nn.Parameter(torch.randn(num_params) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input data [batch_size, num_features]

        Returns:
            Expectation values [batch_size, num_qubits]
        """
        batch_size = x.shape[0]
        outputs = []

        for i in range(batch_size):
            # Create quantum state
            state = QuantumState(self.num_qubits)

            # Encode classical data
            self._encode_data(state, x[i])

            # Apply variational circuit
            self._apply_ansatz(state)

            # Measure expectation values
            expectations = self._measure(state)
            outputs.append(expectations)

        return torch.stack(outputs)

    def _encode_data(self, state: QuantumState, x: torch.Tensor):
        """Angle encoding: data → rotation angles."""
        for j in range(min(len(x), self.num_qubits)):
            state.ry(j, x[j].item() * np.pi)

    def _apply_ansatz(self, state: QuantumState):
        """Apply parameterized gates."""
        idx = 0
        for layer in range(self.num_layers):
            # Single-qubit rotations
            for q in range(self.num_qubits):
                state.rx(q, self.params[idx].item())
                state.ry(q, self.params[idx + 1].item())
                state.rz(q, self.params[idx + 2].item())
                idx += 3

            # Entangling layer
            for q in range(self.num_qubits - 1):
                state.cnot(q, q + 1)

    def _measure(self, state: QuantumState) -> torch.Tensor:
        """Measure Z expectation on each qubit."""
        expectations = []
        for q in range(self.num_qubits):
            exp_z = state.expectation_z(q)
            expectations.append(exp_z)
        return torch.tensor(expectations, dtype=torch.float32)
```

### Using the Built-in QuantumLayer

Moonlab provides an optimized `QuantumLayer` with automatic differentiation:

```python
from moonlab.torch_layer import QuantumLayer

# Create quantum layer with 4 qubits, 2 variational layers
quantum_layer = QuantumLayer(
    num_qubits=4,
    num_layers=2,
    encoding='angle',           # 'angle', 'amplitude', or 'iqp'
    ansatz='strongly_entangling',  # 'strongly_entangling', 'basic', 'hardware_efficient'
    measurement='expectation_z'    # 'expectation_z', 'probabilities', 'samples'
)

# Forward pass
x = torch.randn(32, 4)  # batch of 32 samples, 4 features each
output = quantum_layer(x)  # [32, 4] expectation values
```

## Part 2: The Parameter-Shift Rule

Quantum circuits require special gradient computation since we can't directly backpropagate through quantum operations.

### Mathematical Foundation

For a gate $R(\theta) = e^{-i\theta G/2}$ where $G$ has eigenvalues $\pm 1$:

$$\frac{\partial}{\partial \theta} \langle \psi(\theta) | O | \psi(\theta) \rangle = \frac{1}{2} \left[ f(\theta + \frac{\pi}{2}) - f(\theta - \frac{\pi}{2}) \right]$$

where $f(\theta) = \langle O \rangle_\theta$.

### Implementation

```python
class ParameterShiftQuantumLayer(nn.Module):
    """Quantum layer with parameter-shift gradient."""

    def __init__(self, num_qubits: int, num_params: int):
        super().__init__()
        self.num_qubits = num_qubits
        self.params = nn.Parameter(torch.randn(num_params) * 0.1)
        self.shift = np.pi / 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return QuantumFunction.apply(x, self.params, self.num_qubits)


class QuantumFunction(torch.autograd.Function):
    """Custom autograd function for quantum circuits."""

    @staticmethod
    def forward(ctx, x, params, num_qubits):
        ctx.save_for_backward(x, params)
        ctx.num_qubits = num_qubits

        # Execute quantum circuit
        output = execute_circuit(x, params, num_qubits)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, params = ctx.saved_tensors
        num_qubits = ctx.num_qubits
        shift = np.pi / 2

        # Parameter-shift gradients
        grad_params = torch.zeros_like(params)

        for i in range(len(params)):
            # Shifted parameters
            params_plus = params.clone()
            params_plus[i] += shift

            params_minus = params.clone()
            params_minus[i] -= shift

            # Evaluate at shifted points
            f_plus = execute_circuit(x, params_plus, num_qubits)
            f_minus = execute_circuit(x, params_minus, num_qubits)

            # Parameter-shift formula
            grad = (f_plus - f_minus) / 2
            grad_params[i] = (grad_output * grad).sum()

        return None, grad_params, None


def execute_circuit(x, params, num_qubits):
    """Execute parameterized quantum circuit."""
    batch_size = x.shape[0]
    outputs = []

    for i in range(batch_size):
        state = QuantumState(num_qubits)

        # Encoding
        for q in range(num_qubits):
            if q < x.shape[1]:
                state.ry(q, x[i, q].item() * np.pi)

        # Variational layer
        for q in range(num_qubits):
            state.rx(q, params[q * 3].item())
            state.ry(q, params[q * 3 + 1].item())
            state.rz(q, params[q * 3 + 2].item())

        # Measurement
        exp = state.expectation_z(0)
        outputs.append(exp)

    return torch.tensor(outputs, dtype=torch.float32)
```

## Part 3: Building a Hybrid Classifier

### Dataset: Moons Classification

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate dataset
X, y = make_moons(n_samples=500, noise=0.15, random_state=42)
X = StandardScaler().fit_transform(X)

# Split and convert to tensors
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)
```

### Hybrid Model Architecture

```python
class HybridQuantumClassifier(nn.Module):
    """
    Classical → Quantum → Classical hybrid architecture.

    Architecture:
        Input (2) → Linear (4) → QuantumLayer (4 qubits) → Linear (2) → Output
    """

    def __init__(self):
        super().__init__()

        # Classical pre-processing
        self.pre_net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.Tanh()  # Bound inputs to [-1, 1] for encoding
        )

        # Quantum layer
        self.quantum = QuantumLayer(
            num_qubits=4,
            num_layers=3,
            encoding='angle',
            ansatz='strongly_entangling'
        )

        # Classical post-processing
        self.post_net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        x = self.pre_net(x)
        x = self.quantum(x)
        x = self.post_net(x)
        return x
```

### Training Loop

```python
def train_hybrid_model():
    model = HybridQuantumClassifier()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Training
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

        if (epoch + 1) % 10 == 0:
            acc = 100 * correct / total
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Acc: {acc:.2f}%")

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = outputs.max(1)
        test_acc = (predicted == y_test).sum().item() / len(y_test)
        print(f"\nTest Accuracy: {100*test_acc:.2f}%")

    return model

# Train the model
model = train_hybrid_model()
```

**Expected Output:**
```
Epoch 10/50, Loss: 0.4523, Acc: 78.50%
Epoch 20/50, Loss: 0.3102, Acc: 86.25%
Epoch 30/50, Loss: 0.2341, Acc: 90.75%
Epoch 40/50, Loss: 0.1876, Acc: 93.00%
Epoch 50/50, Loss: 0.1542, Acc: 95.25%

Test Accuracy: 94.00%
```

## Part 4: Advanced Techniques

### Data Re-uploading

Interleave data encoding with variational layers for increased expressivity:

```python
class DataReuploadingLayer(nn.Module):
    """
    Data re-uploading: encode data multiple times between variational layers.

    Reference: Pérez-Salinas et al., "Data re-uploading for a universal
               quantum classifier" (2020)
    """

    def __init__(self, num_qubits: int, num_layers: int, input_dim: int):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.input_dim = input_dim

        # Trainable parameters for each layer
        # Each layer: input scaling + rotation angles
        params_per_layer = input_dim + num_qubits * 3
        total_params = num_layers * params_per_layer
        self.params = nn.Parameter(torch.randn(total_params) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        outputs = []

        for b in range(batch_size):
            state = QuantumState(self.num_qubits)
            param_idx = 0

            for layer in range(self.num_layers):
                # Data encoding with learnable scaling
                for q in range(min(self.input_dim, self.num_qubits)):
                    scale = self.params[param_idx].item()
                    param_idx += 1
                    angle = scale * x[b, q].item()
                    state.ry(q, angle)

                # Variational rotations
                for q in range(self.num_qubits):
                    state.rx(q, self.params[param_idx].item())
                    state.ry(q, self.params[param_idx + 1].item())
                    state.rz(q, self.params[param_idx + 2].item())
                    param_idx += 3

                # Entangling
                for q in range(self.num_qubits - 1):
                    state.cnot(q, q + 1)

            # Measure
            exp = state.expectation_z(0)
            outputs.append(exp)

        return torch.tensor(outputs, dtype=torch.float32).unsqueeze(1)
```

### Quantum Kernel Methods

Use quantum circuits to compute kernel functions:

```python
class QuantumKernel:
    """
    Quantum kernel: k(x, x') = |⟨φ(x)|φ(x')⟩|²

    Use with sklearn's SVC for kernel-based classification.
    """

    def __init__(self, num_qubits: int, num_layers: int = 2):
        self.num_qubits = num_qubits
        self.num_layers = num_layers

    def _encode(self, state: QuantumState, x: np.ndarray):
        """Apply feature map."""
        for layer in range(self.num_layers):
            # Single-qubit rotations
            for q in range(self.num_qubits):
                if q < len(x):
                    state.ry(q, x[q] * np.pi)
                    state.rz(q, x[q] * np.pi)

            # Entangling
            for q in range(self.num_qubits - 1):
                state.cnot(q, q + 1)

    def compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix K[i,j] = k(X1[i], X2[j])."""
        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._kernel_element(X1[i], X2[j])

        return K

    def _kernel_element(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute single kernel element via swap test or fidelity."""
        state1 = QuantumState(self.num_qubits)
        state2 = QuantumState(self.num_qubits)

        self._encode(state1, x1)
        self._encode(state2, x2)

        # Fidelity = |⟨ψ₁|ψ₂⟩|²
        fidelity = state1.fidelity(state2)
        return fidelity


# Usage with sklearn
from sklearn.svm import SVC

qkernel = QuantumKernel(num_qubits=4)
K_train = qkernel.compute_kernel(X_train.numpy(), X_train.numpy())
K_test = qkernel.compute_kernel(X_test.numpy(), X_train.numpy())

svc = SVC(kernel='precomputed')
svc.fit(K_train, y_train.numpy())
accuracy = svc.score(K_test, y_test.numpy())
print(f"Quantum SVM Accuracy: {100*accuracy:.2f}%")
```

### Batched Execution for Performance

```python
from moonlab.torch_layer import BatchedQuantumLayer

# Batched execution on GPU
quantum_layer = BatchedQuantumLayer(
    num_qubits=4,
    num_layers=2,
    device='metal',  # Use Metal GPU
    batch_parallel=True  # Execute batch elements in parallel
)

# Much faster for large batches
x = torch.randn(256, 4)
output = quantum_layer(x)  # Parallelized on GPU
```

## Part 5: Training Tips

### Hyperparameter Guidelines

| Parameter | Recommended Range | Notes |
|-----------|------------------|-------|
| Learning rate | 0.001 - 0.1 | Quantum params may need higher LR |
| Num qubits | 4-8 | Balance expressivity vs. simulation cost |
| Num layers | 2-6 | More layers = more parameters |
| Batch size | 16-64 | Smaller batches for noisy gradients |

### Avoiding Barren Plateaus

Barren plateaus occur when gradients vanish exponentially with circuit depth:

```python
# Strategy 1: Use shallow circuits with local initialization
def initialize_near_identity(params, scale=0.1):
    """Initialize parameters so circuit is close to identity."""
    return torch.randn_like(params) * scale

# Strategy 2: Layer-wise training
def train_layerwise(model, X, y, epochs_per_layer=20):
    """Train one layer at a time, freezing previous layers."""
    for layer_idx in range(model.num_layers):
        # Freeze all layers except current
        for i, param in enumerate(model.parameters()):
            param.requires_grad = (i == layer_idx)

        # Train current layer
        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad])
        for epoch in range(epochs_per_layer):
            # ... training loop
            pass

    # Finally, fine-tune all layers together
    for param in model.parameters():
        param.requires_grad = True
```

### Gradient Clipping

```python
# Quantum gradients can be noisy
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## Exercises

### Exercise 1: Iris Classification

Build a hybrid model for the Iris dataset (4 features, 3 classes).

```python
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
# ... build and train model
```

### Exercise 2: Custom Encoding

Implement IQP (Instantaneous Quantum Polynomial) encoding:

```python
def iqp_encoding(state, x):
    """
    IQP encoding with ZZ interactions.

    φ(x) = exp(i Σᵢⱼ xᵢxⱼ ZᵢZⱼ) H⊗ⁿ |0⟩ⁿ
    """
    # Apply Hadamard to all qubits
    for q in range(state.num_qubits):
        state.h(q)

    # ZZ rotations based on data products
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            if i < state.num_qubits and j < state.num_qubits:
                angle = x[i] * x[j]
                state.rzz(i, j, angle)
```

### Exercise 3: Quantum Transfer Learning

Use a pre-trained classical network with a quantum layer:

```python
import torchvision.models as models

# Pre-trained ResNet features
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Identity()  # Remove classifier

class QuantumTransferModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = resnet
        self.reduce = nn.Linear(512, 4)
        self.quantum = QuantumLayer(num_qubits=4, num_layers=2)
        self.classifier = nn.Linear(4, 10)

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
        x = self.reduce(x)
        x = torch.tanh(x)
        x = self.quantum(x)
        return self.classifier(x)
```

## Summary

In this tutorial, you learned:

1. **Quantum Layers**: Integrate quantum circuits as PyTorch modules
2. **Parameter-Shift Rule**: Compute gradients for quantum parameters
3. **Hybrid Architectures**: Combine classical and quantum layers
4. **Advanced Techniques**: Data re-uploading, quantum kernels
5. **Training Best Practices**: Avoiding barren plateaus, gradient clipping

## Next Steps

- Explore [VQE for Chemistry](06-vqe-molecular-simulation.md) for scientific applications
- See [QAOA Optimization](07-qaoa-optimization.md) for combinatorial problems
- Check the [PyTorch Layer API](../api/python/torch-layer.md) for full reference

## References

1. Schuld, M., & Petruccione, F. (2021). *Machine Learning with Quantum Computers*
2. Mitarai, K., et al. (2018). "Quantum Circuit Learning"
3. Pérez-Salinas, A., et al. (2020). "Data re-uploading for a universal quantum classifier"
4. Havlíček, V., et al. (2019). "Supervised learning with quantum-enhanced feature spaces"
