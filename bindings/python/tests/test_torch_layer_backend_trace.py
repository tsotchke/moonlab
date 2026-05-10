"""Runtime provenance tests for Moonlab's optional PyTorch integration."""

import pytest

torch = pytest.importorskip("torch")

from moonlab.torch_layer import (  # noqa: E402
    ParameterShiftGradient,
    QuantumConv1D,
    QuantumPooling,
    _measurement_tensor,
    moonlab_torch_backend_probe,
    moonlab_torch_last_backend_trace,
)


def _assert_trace(trace, owner):
    assert trace["owner"] == owner
    assert trace["quantum_backend"] == "moonlab-c-ctypes"
    assert trace["library_path"]
    assert trace["available"] is True
    assert trace["native_quantum_on_cpu"] is True
    assert trace["torch_device"] in {"cpu", "cuda", "mps"}
    assert isinstance(trace["fallback_intentional"], bool)
    assert trace["missing_symbols"] == []


@pytest.mark.ml
def test_torch_backend_probe_reports_native_runtime():
    probe = moonlab_torch_backend_probe()
    assert probe["quantum_backend"] == "moonlab-c-ctypes"
    assert probe["available"] is True
    assert probe["missing_symbols"] == []
    assert probe["library_path"]


@pytest.mark.ml
def test_measurement_tensor_records_backend_trace():
    reference = torch.ones(2, dtype=torch.float64)
    result = _measurement_tensor([0.25, -0.5], reference)
    assert result.dtype == torch.float64
    assert result.device == reference.device
    _assert_trace(moonlab_torch_last_backend_trace(), "_measurement_tensor")


@pytest.mark.ml
def test_parameter_shift_backward_records_backend_trace():
    inputs = torch.tensor([0.2, -0.1], dtype=torch.float32, requires_grad=True)
    params = torch.tensor([0.3, 0.7], dtype=torch.float32, requires_grad=True)

    def circuit_fn(x, p):
        return torch.stack((torch.sin(x[0] + p[0]), torch.cos(x[1] + p[1])))

    output = ParameterShiftGradient.apply(inputs, params, circuit_fn, lambda y: y, 2)
    output.sum().backward()

    assert inputs.grad is not None
    assert params.grad is not None
    _assert_trace(ParameterShiftGradient.last_backend_trace, "ParameterShiftGradient.backward")


@pytest.mark.ml
def test_quantum_conv1d_forward_records_backend_trace():
    layer = QuantumConv1D(in_features=3, num_qubits=2, kernel_size=2, stride=1, depth=1)
    output = layer(torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32))
    assert tuple(output.shape) == (1, 2, 2)
    _assert_trace(layer.last_backend_trace, "QuantumConv1D.forward")


@pytest.mark.ml
def test_quantum_pooling_forward_records_backend_trace():
    layer = QuantumPooling(pool_size=2, method="measure")
    output = layer(torch.tensor([[1.0, 3.0, 2.0, 6.0]], dtype=torch.float32))
    assert tuple(output.shape) == (1, 2)
    assert torch.allclose(output, torch.tensor([[2.0, 4.0]], dtype=torch.float32))
    _assert_trace(layer.last_backend_trace, "QuantumPooling.forward")
