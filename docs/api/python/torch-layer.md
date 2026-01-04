# PyTorch Integration API

Complete reference for PyTorch quantum neural network layers.

**Module**: `moonlab.torch_layer`

## Overview

The torch_layer module provides differentiable quantum computing layers that integrate seamlessly with PyTorch. Features:

- **QuantumLayer**: General parameterized quantum circuit
- **QuantumConv1D**: Quantum convolution for sequence data
- **QuantumPooling**: Quantum dimensionality reduction
- **Hybrid Networks**: Combine classical and quantum layers
- **Automatic Differentiation**: Parameter shift rule for gradients

## Requirements

```python
import torch
from moonlab.torch_layer import QuantumLayer, QuantumConv1D, QuantumPooling
```

Requires PyTorch >= 1.9.0.

## QuantumLayer

General-purpose parameterized quantum circuit layer.

### Constructor

```python
QuantumLayer(
    num_qubits: int,
    num_layers: int = 2,
    encoding: str = 'angle',
    measurement: str = 'expectation'
)
```

**Parameters**:
- `num_qubits`: Number of qubits in the circuit
- `num_layers`: Depth of variational ansatz
- `encoding`: Input encoding method
  - `'angle'`: Rotation-based encoding (default)
  - `'amplitude'`: Amplitude encoding
  - `'iqp'`: IQP feature map
- `measurement`: Output measurement type
  - `'expectation'`: Pauli-Z expectations (default)
  - `'probabilities'`: Full probability distribution
  - `'samples'`: Sampled bitstrings

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `num_qubits` | int | Number of qubits |
| `num_layers` | int | Circuit depth |
| `num_params` | int | Total trainable parameters |
| `params` | nn.Parameter | Variational parameters |

### Methods

#### forward

```python
forward(x: torch.Tensor) -> torch.Tensor
```

Apply quantum layer to input.

**Parameters**:
- `x`: Input tensor of shape (batch_size, num_qubits) or (batch_size, features)

**Returns**: Output tensor, shape depends on measurement type

**Example**:
```python
import torch
from moonlab.torch_layer import QuantumLayer

# Create quantum layer
layer = QuantumLayer(num_qubits=4, num_layers=2)

# Forward pass
x = torch.randn(32, 4)  # Batch of 32, 4 features
output = layer(x)
print(output.shape)  # torch.Size([32, 4]) for expectation

# Backward pass works automatically
loss = output.sum()
loss.backward()
print(layer.params.grad.shape)  # Gradients computed
```

#### get_circuit_parameters

```python
get_circuit_parameters() -> torch.Tensor
```

Get current variational parameters.

#### set_circuit_parameters

```python
set_circuit_parameters(params: torch.Tensor) -> None
```

Set variational parameters.

### Parameter Count

For a layer with $n$ qubits and $L$ layers:

$$\text{params} = L \times (3n + n-1)$$

- $3n$ rotation gates per layer (RX, RY, RZ on each qubit)
- $n-1$ entangling gates per layer (CNOT chain)

## QuantumConv1D

Quantum convolutional layer for sequence data.

### Constructor

```python
QuantumConv1D(
    in_channels: int,
    out_channels: int,
    kernel_qubits: int = 4,
    stride: int = 1,
    num_layers: int = 2
)
```

**Parameters**:
- `in_channels`: Input feature channels
- `out_channels`: Output feature channels
- `kernel_qubits`: Qubits per convolutional kernel
- `stride`: Convolution stride
- `num_layers`: Quantum circuit depth

### Methods

#### forward

```python
forward(x: torch.Tensor) -> torch.Tensor
```

Apply quantum convolution.

**Parameters**:
- `x`: Input tensor of shape (batch, in_channels, length)

**Returns**: Output tensor of shape (batch, out_channels, new_length)

**Example**:
```python
from moonlab.torch_layer import QuantumConv1D

# Quantum convolution
qconv = QuantumConv1D(
    in_channels=3,
    out_channels=8,
    kernel_qubits=4,
    stride=2
)

# Apply to sequence data
x = torch.randn(16, 3, 64)  # Batch 16, 3 channels, length 64
output = qconv(x)
print(output.shape)  # torch.Size([16, 8, 32])
```

### Architecture

The quantum convolution:
1. Slides quantum kernel over input sequence
2. Encodes local patch into quantum state
3. Applies variational circuit
4. Measures to produce output features

## QuantumPooling

Quantum pooling layer for dimensionality reduction.

### Constructor

```python
QuantumPooling(
    pool_size: int = 2,
    num_qubits: int = 4,
    method: str = 'measure'
)
```

**Parameters**:
- `pool_size`: Pooling window size
- `num_qubits`: Qubits for pooling circuit
- `method`: Pooling method
  - `'measure'`: Measurement-based reduction (default)
  - `'trace'`: Partial trace
  - `'swap'`: SWAP test similarity

### Methods

#### forward

```python
forward(x: torch.Tensor) -> torch.Tensor
```

Apply quantum pooling.

**Example**:
```python
from moonlab.torch_layer import QuantumPooling

pool = QuantumPooling(pool_size=2, num_qubits=4)

x = torch.randn(16, 8, 32)  # Batch 16, 8 channels, length 32
output = pool(x)
print(output.shape)  # torch.Size([16, 8, 16])
```

## Gradient Computation

### Parameter Shift Rule

Gradients are computed using the parameter shift rule:

$$\frac{\partial f}{\partial \theta} = \frac{f(\theta + \pi/2) - f(\theta - \pi/2)}{2}$$

This provides exact gradients for rotation gates.

### Custom Gradient Function

```python
class QuantumGradientFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, params, circuit):
        # Execute quantum circuit
        output = circuit.execute(input, params)
        ctx.save_for_backward(input, params)
        ctx.circuit = circuit
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Parameter shift rule
        input, params = ctx.saved_tensors
        circuit = ctx.circuit

        grad_params = torch.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.clone()
            params_plus[i] += np.pi / 2
            params_minus = params.clone()
            params_minus[i] -= np.pi / 2

            out_plus = circuit.execute(input, params_plus)
            out_minus = circuit.execute(input, params_minus)

            grad_params[i] = (grad_output * (out_plus - out_minus) / 2).sum()

        return None, grad_params, None
```

## Hybrid Neural Networks

### QuantumClassifier

Complete hybrid quantum-classical classifier.

```python
from moonlab.torch_layer import QuantumLayer
import torch.nn as nn

class QuantumClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_qubits=4):
        super().__init__()

        # Classical preprocessing
        self.classical1 = nn.Linear(input_dim, 64)
        self.classical2 = nn.Linear(64, num_qubits)

        # Quantum layer
        self.quantum = QuantumLayer(
            num_qubits=num_qubits,
            num_layers=3,
            encoding='angle',
            measurement='expectation'
        )

        # Classical postprocessing
        self.classifier = nn.Linear(num_qubits, num_classes)

    def forward(self, x):
        # Classical encoding
        x = torch.relu(self.classical1(x))
        x = torch.tanh(self.classical2(x))  # Scale to [-1, 1]

        # Quantum processing
        x = self.quantum(x)

        # Classification
        return self.classifier(x)

# Usage
model = QuantumClassifier(input_dim=784, num_classes=10, num_qubits=6)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(100):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
```

### QuantumAutoencoder

Hybrid quantum autoencoder.

```python
class QuantumAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_qubits=4):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_qubits)
        )

        # Quantum bottleneck
        self.quantum = QuantumLayer(
            num_qubits=latent_qubits,
            num_layers=2,
            measurement='probabilities'
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2**latent_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        q = self.quantum(z)
        return self.decoder(q)
```

## Training Utilities

### QuantumScheduler

Learning rate scheduler for quantum layers.

```python
from moonlab.torch_layer import QuantumScheduler

scheduler = QuantumScheduler(
    optimizer,
    warmup_epochs=5,
    decay_epochs=50,
    min_lr=1e-5
)

for epoch in range(100):
    train_one_epoch()
    scheduler.step()
```

### GradientClipping

Quantum-aware gradient clipping.

```python
from moonlab.torch_layer import clip_quantum_gradients

# After backward pass
clip_quantum_gradients(model, max_norm=1.0)
optimizer.step()
```

## Advanced Features

### Data Re-uploading

Re-encode data at each layer for increased expressivity.

```python
layer = QuantumLayer(
    num_qubits=4,
    num_layers=3,
    encoding='angle',
    data_reuploading=True  # Re-encode at each layer
)
```

### Entanglement Patterns

Control entanglement structure.

```python
layer = QuantumLayer(
    num_qubits=6,
    num_layers=2,
    entanglement='linear'  # Options: 'linear', 'circular', 'full', 'none'
)
```

| Pattern | Description | CNOT Count |
|---------|-------------|------------|
| `linear` | Chain connectivity | n-1 |
| `circular` | Ring connectivity | n |
| `full` | All-to-all | n(n-1)/2 |
| `none` | No entanglement | 0 |

### Observable Selection

Choose measurement observables.

```python
layer = QuantumLayer(
    num_qubits=4,
    num_layers=2,
    observables=['Z0', 'Z1', 'Z2', 'Z3', 'Z0Z1', 'Z2Z3']
)
```

## Complete Training Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from moonlab.torch_layer import QuantumLayer
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(1000, 4).astype(np.float32)
y = (X[:, 0] * X[:, 1] > 0).astype(np.int64)

# Create dataset
dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define model
class HybridClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantum = QuantumLayer(num_qubits=4, num_layers=3)
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        x = self.quantum(x)
        return self.fc(x)

model = HybridClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Training
for epoch in range(50):
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(1)
        correct += predicted.eq(batch_y).sum().item()
        total += batch_y.size(0)

    if epoch % 10 == 0:
        acc = 100. * correct / total
        print(f"Epoch {epoch}: Loss = {total_loss:.4f}, Acc = {acc:.2f}%")

# Final accuracy
print(f"Final accuracy: {100. * correct / total:.2f}%")
```

## Performance Tips

1. **Batch Size**: Larger batches amortize quantum circuit setup
2. **Qubit Count**: Start with 4-8 qubits for reasonable training times
3. **Layer Depth**: 2-4 layers often sufficient; deeper may not help
4. **Learning Rate**: Quantum parameters often need smaller LR (0.001-0.01)
5. **Gradient Accumulation**: Use for effective larger batches

```python
# Gradient accumulation example
accumulation_steps = 4

for i, (batch_x, batch_y) in enumerate(train_loader):
    output = model(batch_x)
    loss = criterion(output, batch_y) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## GPU Acceleration

Quantum layers automatically use Metal GPU when available.

```python
# Check if quantum GPU is available
from moonlab.torch_layer import quantum_gpu_available

if quantum_gpu_available():
    print("Quantum operations will use Metal GPU")
```

## Serialization

Save and load quantum models.

```python
# Save
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
}, 'quantum_model.pt')

# Load
checkpoint = torch.load('quantum_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## See Also

- [Core API](core.md) - QuantumState, Gates
- [ML API](ml.md) - Quantum kernels, QSVM
- [Algorithms API](algorithms.md) - VQE, QAOA

