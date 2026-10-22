"""Unit tests for Miniature Stable LatentMoE with Quantile Balancing (Kimi K3)."""

import pytest
import torch

from src.models.latent_moe import StableLatentMoE
from src.models.router import QuantileBalancingRouter


def test_latent_moe_shapes_and_routing():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    d_model = 64
    d_latent = 16  # 4x compression in latent space
    num_routed_experts = 8
    num_shared_experts = 1
    top_k = 2
    batch_size = 2
    seq_len = 8

    moe = StableLatentMoE(
        d_model=d_model,
        d_latent=d_latent,
        num_routed_experts=num_routed_experts,
        num_shared_experts=num_shared_experts,
        top_k=top_k,
    ).to(device)

    x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    out, aux_info = moe(x)

    # Output shape matches input shape
    assert out.shape == (batch_size, seq_len, d_model)
    assert "selected_indices" in aux_info
    assert "routing_weights" in aux_info
    assert aux_info["selected_indices"].shape == (batch_size, seq_len, top_k)
    assert aux_info["routing_weights"].shape == (batch_size, seq_len, top_k)

    # Verify routing weights sum to 1.0 for each token
    weights_sum = aux_info["routing_weights"].sum(dim=-1)
    assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-4)

    # Verify backward pass
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_pre_up_rmsnorm_stability():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    d_model = 32
    d_latent = 8

    moe = StableLatentMoE(
        d_model=d_model,
        d_latent=d_latent,
        num_routed_experts=4,
        num_shared_experts=1,
        top_k=2,
    ).to(device)

    # With pre-up RMSNorm, latent representation scale is strictly normalized
    extreme_x = torch.full((1, 4, d_model), 100.0, device=device)
    out, aux = moe(extreme_x)
    assert torch.isfinite(out).all()
    assert not torch.isnan(out).any()


def test_quantile_balancing_load():
    num_experts = 4
    top_k = 1
    d_model = 16
    router = QuantileBalancingRouter(d_model=d_model, num_experts=num_experts, top_k=top_k)

    # Simulate tokens routed to experts
    x = torch.randn(20, d_model)
    indices, weights = router(x)

    # Count dispatches per expert
    counts = torch.bincount(indices.view(-1), minlength=num_experts).float()
    initial_bias = router.bias.clone()

    # Apply quantile balancing update to correct imbalance
    target_count = 20.0 * top_k / num_experts
    router.update_bias(counts, target_count=target_count, step_size=0.1)

    # Expert with high count should have bias decreased, low count should have bias increased
    max_expert = torch.argmax(counts).item()
    min_expert = torch.argmin(counts).item()
    if counts[max_expert] > counts[min_expert]:
        assert router.bias[max_expert] < initial_bias[max_expert]
        assert router.bias[min_expert] > initial_bias[min_expert]
