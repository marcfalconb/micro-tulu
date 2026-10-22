"""Stable LatentMoE with Pre-Up RMSNorm and Shared Experts (Kimi K3).

Reference:
Kimi K3 Technical Report (Moonshot AI, 2026 - Section 2.3):
"LatentMoE separates the full model width from the routed-expert width:
shared experts retain a full-width path for common transformations, whereas
specialized routed experts operate in a compact latent space of width ell << d.
To stabilize training across extreme sparsity, Stable LatentMoE introduces:
1. Shared full-width experts (N_s >= 1).
2. Latent down-projection W_down: d -> ell.
3. Routed experts operating purely in R^ell with SiTU-GLU.
4. Pre-up RMSNorm: inserts RMSNorm before W_up:
       u = sum_{i in TopK} p_i E_i^routed(W_down x)
       y = sum_{j=1}^{N_s} E_j^shared(x) + W_up RMSNorm(u)
   This normalizes sensitivity of the routed branch to expert scale variations."
"""

from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.activations import SiTUGLU
from src.models.router import QuantileBalancingRouter


class StableLatentMoE(nn.Module):
    """Stable Latent Mixture-of-Experts layer."""

    def __init__(
        self,
        d_model: int,
        d_latent: int,
        num_routed_experts: int = 8,
        num_shared_experts: int = 1,
        top_k: int = 2,
        d_hidden_mult: int = 2,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.eps = eps

        # 1. Full-width shared experts: R^d -> R^d
        self.shared_experts = nn.ModuleList([
            SiTUGLU(d_in=d_model, d_hidden=d_model * d_hidden_mult, d_out=d_model)
            for _ in range(num_shared_experts)
        ])

        # 2. Latent projections: W_down : R^d -> R^ell, W_up : R^ell -> R^d
        self.w_down = nn.Linear(d_model, d_latent, bias=False)
        self.w_up = nn.Linear(d_latent, d_model, bias=False)

        # 3. Routed experts operating in compact latent space: R^ell -> R^ell
        self.routed_experts = nn.ModuleList([
            SiTUGLU(d_in=d_latent, d_hidden=d_latent * d_hidden_mult, d_out=d_latent)
            for _ in range(num_routed_experts)
        ])

        # 4. Quantile Balancing Router
        self.router = QuantileBalancingRouter(
            d_model=d_model,
            num_experts=num_routed_experts,
            top_k=top_k,
        )

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Root Mean Square Normalization before up-projection."""
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Forward pass through Stable LatentMoE layer.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]

        Returns:
            y: Output tensor of shape [batch_size, seq_len, d_model]
            aux_info: Dictionary containing routing indices, weights, and metrics.
        """
        orig_shape = x.shape
        B, T, D = orig_shape
        x_flat = x.view(-1, D)
        N = x_flat.shape[0]

        # 1. Shared expert branch: sum_j E_shared_j(x)
        shared_out = torch.zeros_like(x_flat)
        for shared_exp in self.shared_experts:
            shared_out = shared_out + shared_exp(x_flat)

        # 2. Router Top-k selection
        selected_indices, routing_weights = self.router(x_flat)  # [N, top_k], [N, top_k]

        # 3. Latent down-projection: z = W_down x in R^ell
        z_latent = self.w_down(x_flat)  # [N, d_latent]

        # 4. Dispatch tokens to selected routed experts
        routed_aggregate = torch.zeros_like(z_latent)  # [N, d_latent]

        for k in range(self.top_k):
            k_indices = selected_indices[:, k]  # [N]
            k_weights = routing_weights[:, k].unsqueeze(-1)  # [N, 1]

            # Accumulate each expert's contribution
            for exp_id in range(self.num_routed_experts):
                mask = (k_indices == exp_id)
                if mask.any():
                    tokens_in = z_latent[mask]
                    exp_out = self.routed_experts[exp_id](tokens_in)
                    routed_aggregate[mask] = routed_aggregate[mask] + k_weights[mask] * exp_out

        # 5. Pre-Up RMSNorm (Kimi K3 Section 2.3.1): RMSNorm(u)
        normalized_u = self._rms_norm(routed_aggregate)  # [N, d_latent]

        # 6. Latent up-projection: W_up RMSNorm(u)
        routed_out = self.w_up(normalized_u)  # [N, d_model]

        # 7. Final combined output
        y_flat = shared_out + routed_out
        y = y_flat.view(B, T, D)

        aux_info = {
            "selected_indices": selected_indices.view(B, T, self.top_k),
            "routing_weights": routing_weights.view(B, T, self.top_k),
            "shared_norm": shared_out.norm(dim=-1).mean().item(),
            "routed_norm": routed_out.norm(dim=-1).mean().item(),
        }

        return y, aux_info
