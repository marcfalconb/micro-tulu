"""Unit tests for Per-Head Muon Optimizer and Multi-Teacher On-Policy Distillation (Kimi K3)."""

import pytest
import torch
import torch.nn as nn

from src.training.optimizers import PerHeadMuon, zeropower_via_newtonschulz5
from src.training.distill import compute_mopd_reward, MultiTeacherOnPolicyDistillation


def test_per_head_muon_orthogonalization():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    d_out = 32
    d_in = 64
    num_heads = 4

    linear = nn.Linear(d_in, d_out, bias=False).to(device)
    optimizer = PerHeadMuon(
        [linear.weight],
        lr=0.02,
        momentum=0.95,
        num_heads=num_heads,
        ns_steps=5,
    )

    # Perform optimization step
    x = torch.randn(2, 8, d_in, device=device)
    loss = linear(x).pow(2).sum()
    loss.backward()

    initial_weight = linear.weight.clone()
    optimizer.step()
    updated_weight = linear.weight

    # Ensure weight was updated
    assert not torch.allclose(initial_weight, updated_weight)

    # Test Newton-Schulz directly on a random matrix
    test_mat = torch.randn(d_out // num_heads, d_in, device=device)
    ortho_mat = zeropower_via_newtonschulz5(test_mat, steps=5)
    assert torch.isfinite(ortho_mat).all()


def test_mopd_clipped_log_ratio():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    batch_size = 2
    seq_len = 4
    r_max = 2.0

    # Student and teacher logprobs per token
    student_logprobs = torch.tensor([
        [-1.0, -2.0, -1.5, -0.5],
        [-5.0, -0.1, -10.0, -2.0],
    ], device=device, requires_grad=True)

    # Teacher with both much higher likelihood (ratio > r_max) and much lower (ratio < -r_max)
    teacher_logprobs = torch.tensor([
        [-0.1, -1.8, -1.6, -0.4],
        [0.0, -5.0, -1.0, -2.0],  # First token has log_ratio = 5.0 (exceeds +2.0), 3rd token log_ratio = 9.0
    ], device=device)

    # Mask: 1 for active tokens
    mask = torch.ones(batch_size, seq_len, device=device)

    # Compute MOPD rewards
    rewards = compute_mopd_reward(
        student_logprobs=student_logprobs,
        teacher_logprobs=teacher_logprobs,
        r_max=r_max,
    )

    # Verify bounds: all rewards must lie in [-r_max, r_max]
    assert torch.all(rewards <= (r_max + 1e-5))
    assert torch.all(rewards >= -(r_max + 1e-5))
    assert (rewards == r_max).any()  # Clamped upper bound triggered

    # Test MultiTeacher distillation module
    mopd = MultiTeacherOnPolicyDistillation(r_max=r_max)
    distill_loss, metrics = mopd.compute_loss(student_logprobs, teacher_logprobs, mask)

    assert distill_loss.item() != 0.0
    assert "mean_opd_reward" in metrics
    assert "clipped_fraction" in metrics

    distill_loss.backward()
    assert student_logprobs.grad is not None
