"""Sigmoid Tanh Unit Gated Linear Unit (SiTU-GLU).

Reference:
Kimi K3 Technical Report (Moonshot AI, 2026 - Section 2.3.2):
"SwiGLU computes [x * sigma(x)] * W_u x. However, both multiplicative factors in
SwiGLU are unbounded, so coincident large coordinates produce activation outliers
and overflow in low-precision (FP8 / MXFP4) arithmetic.
SiTU-GLU applies a smooth soft-cap softcap(x, beta) = beta * tanh(x / beta) to
the linear factor of the Swish gate and independently to the up-branch:
    SiTU-GLU(x) = [beta1 * tanh(W_g x / beta1) * sigmoid(W_g x)] * [beta2 * tanh(W_u x / beta2)]
For Kimi K3, beta1 = 4.0 and beta2 = 25.0, strictly bounding activations at |f(x)| <= 100."
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def situglu_activation(
    gate_linear: torch.Tensor,
    up_linear: torch.Tensor,
    beta1: float = 4.0,
    beta2: float = 25.0,
) -> torch.Tensor:
    """Compute SiTU-GLU non-linear elementwise activation.

    Args:
        gate_linear: W_g x tensor
        up_linear: W_u x tensor
        beta1: Soft-cap bound for gate linear component (default 4.0)
        beta2: Soft-cap bound for up branch (default 25.0)

    Returns:
        Bounded activation tensor, strictly |act| <= beta1 * beta2
    """
    # Gate branch: beta1 * tanh(gate / beta1) * sigmoid(gate)
    gate_capped = beta1 * torch.tanh(gate_linear / beta1)
    gate_branch = gate_capped * torch.sigmoid(gate_linear)

    # Up branch: beta2 * tanh(up / beta2)
    up_branch = beta2 * torch.tanh(up_linear / beta2)

    return gate_branch * up_branch


class SiTUGLU(nn.Module):
    """Feed-Forward Network (FFN) block with SiTU-GLU activation."""

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int | None = None,
        beta1: float = 4.0,
        beta2: float = 25.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out or d_in
        self.beta1 = beta1
        self.beta2 = beta2

        self.gate_proj = nn.Linear(self.d_in, self.d_hidden, bias=bias)
        self.up_proj = nn.Linear(self.d_in, self.d_hidden, bias=bias)
        self.down_proj = nn.Linear(self.d_hidden, self.d_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SiTU-GLU FFN."""
        g = self.gate_proj(x)
        u = self.up_proj(x)
        act = situglu_activation(g, u, beta1=self.beta1, beta2=self.beta2)
        return self.down_proj(act)
