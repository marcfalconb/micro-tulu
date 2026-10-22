"""Unit tests for Frontier Architecture Primitives: SiTU-GLU & Block AttnRes (Kimi K3)."""

import pytest
import torch

from src.models.activations import SiTUGLU, situglu_activation
from src.models.attn_res import BlockAttentionResiduals


def test_situglu_forward_and_bound():
    d_in = 64
    d_hidden = 128
    beta1 = 4.0
    beta2 = 25.0

    model = SiTUGLU(d_in=d_in, d_hidden=d_hidden, beta1=beta1, beta2=beta2)

    # Test standard input
    x = torch.randn(2, 8, d_in)
    out = model(x)
    assert out.shape == (2, 8, d_in)

    # Test extreme positive/negative inputs to verify bounded property
    extreme_x = torch.full((1, 4, d_in), 500.0)
    # Check intermediate gated activation before down_proj
    gate_linear = model.gate_proj(extreme_x)
    up_linear = model.up_proj(extreme_x)
    act = situglu_activation(gate_linear, up_linear, beta1=beta1, beta2=beta2)

    # Maximum theoretical bound is beta1 * beta2 = 100.0
    assert torch.all(act <= (beta1 * beta2 + 1e-4))
    assert torch.all(act >= -(beta1 * beta2 + 1e-4))


def test_situglu_backward_mps():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    d_in = 32
    d_hidden = 64
    model = SiTUGLU(d_in=d_in, d_hidden=d_hidden).to(device)

    x = torch.randn(4, 16, d_in, device=device, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(model.gate_proj.weight.grad).all()
    assert torch.isfinite(model.down_proj.weight.grad).all()


def test_block_attnres_forward_and_backward():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    num_layers = 8
    num_blocks = 4
    d_model = 32
    batch_size = 2
    seq_len = 6

    attn_res = BlockAttentionResiduals(
        num_layers=num_layers,
        num_blocks=num_blocks,
        d_model=d_model,
    ).to(device)

    # Simulate token embedding h1
    embedding = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

    # Simulate layer outputs fi(hi)
    layer_outputs = [
        torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        for _ in range(num_layers)
    ]

    # Compute depth attention representation for layer 5 (which is in block 2)
    h_layer5, weights = attn_res(layer_idx=5, embedding=embedding, prior_layer_outputs=layer_outputs[:5])

    assert h_layer5.shape == (batch_size, seq_len, d_model)
    assert weights.dim() >= 2  # attention distribution over preceding block representations
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-4)

    # Test backward pass
    loss = h_layer5.sum()
    loss.backward()

    assert embedding.grad is not None
    assert layer_outputs[0].grad is not None
    assert torch.isfinite(embedding.grad).all()
