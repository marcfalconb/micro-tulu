"""Quantile Balancing Router (Kimi K3).

Reference:
Kimi K3 Technical Report (Moonshot AI, 2026 - Section 2.3.3):
"Unlike auxiliary-loss-based routing, Kimi K3 adopts auxiliary-loss-free routing.
Load balancing is implemented by adding an expert-specific bias b_j to the router
score used for Top-k selection:
    s_i = Sigmoid(W_r x_i)
    T_i = argtopk(s_i + b)
    p_{i,j} = s_{i,j} / sum_{r in T_i} s_{i,r}
Because b is omitted from p_{i,j}, it regulates dispatch without altering mixture
weights or the gradient-based optimization of the router."
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantileBalancingRouter(nn.Module):
    """Auxiliary-loss-free router with adaptive expert bias load balancing."""

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        # Router projection matrix W_r: R^d -> R^E
        self.router_linear = nn.Linear(d_model, num_experts, bias=False)

        # Non-trainable adaptive bias buffer for dispatch regulation
        self.register_buffer("bias", torch.zeros(num_experts))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Top-k expert selection and routing weights for input tokens.

        Args:
            x: Input tensor of shape [..., d_model]

        Returns:
            selected_indices: Tensor of shape [..., top_k] containing expert IDs.
            routing_weights: Tensor of shape [..., top_k] containing normalized mixture weights.
        """
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.d_model)

        # 1. Raw scores via Sigmoid
        raw_scores = torch.sigmoid(self.router_linear(x_flat))  # [N, num_experts]

        # 2. Dispatch selection uses biased score: s + b
        biased_scores = raw_scores + self.bias  # [N, num_experts]
        topk_scores, topk_indices = torch.topk(biased_scores, k=self.top_k, dim=-1)

        # 3. Mixture weights omit bias b: gather from raw_scores and normalize
        chosen_raw_scores = torch.gather(raw_scores, dim=-1, index=topk_indices)
        weights = chosen_raw_scores / chosen_raw_scores.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Reshape to original batch dimensions
        out_indices = topk_indices.view(*orig_shape, self.top_k)
        out_weights = weights.view(*orig_shape, self.top_k)

        return out_indices, out_weights

    def update_bias(
        self,
        expert_counts: torch.Tensor,
        target_count: float,
        step_size: float = 0.05,
    ) -> None:
        """Update router biases to steer expert utilization toward target count.

        Args:
            expert_counts: 1D Tensor of token counts dispatched to each expert.
            target_count: Target number of tokens per expert (e.g. total_tokens * top_k / num_experts).
            step_size: Learning rate for bias adjustments.
        """
        # Under-utilized experts get positive bias delta, over-utilized get negative
        error = (target_count - expert_counts.to(self.bias.device)).sign()
        self.bias.add_(step_size * error)
