"""Per-Head Muon Optimizer with Newton-Schulz Matrix Orthogonalization.

Reference:
Kimi K3 Technical Report (Moonshot AI, 2026 - Section 2.5):
"The Muon optimizer applies Newton-Schulz matrix orthogonalization to 2D weight matrices.
When applied to projection matrices (W_q, W_k, W_v), heads with larger gradient norms
dominate the singular value normalization.
Per-Head Muon partitions each 2D matrix into H distinct heads, performs Newton-Schulz
orthogonalization independently on each head slice, and recombines them.
This equalizes learning rates across attention heads."
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    r"""Newton-Schulz iteration (order 5) to compute approximate matrix orthogonalization.

    Orthogonalizes matrix G such that X X^T \approx I.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.dtype == torch.bfloat16 else G.float()
    X = X / (X.norm() + eps)

    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T

    return X.to(dtype=G.dtype)


class PerHeadMuon(Optimizer):
    """Per-Head Muon Optimizer for 2D Transformer weights."""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        num_heads: int = 8,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            num_heads=num_heads,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            num_heads = group["num_heads"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                if g.ndim != 2:
                    # Fallback standard SGD for 1D or non-matrix parameters
                    p.data.add_(g, alpha=-lr)
                    continue

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                d_out, d_in = g.shape
                # If d_out is divisible by num_heads, partition per-head
                if d_out % num_heads == 0 and num_heads > 1:
                    head_dim = d_out // num_heads
                    # Reshape to [H, head_dim, d_in]
                    buf_heads = buf.view(num_heads, head_dim, d_in)
                    ortho_heads = torch.empty_like(buf_heads)

                    for h in range(num_heads):
                        ortho_heads[h] = zeropower_via_newtonschulz5(buf_heads[h], steps=ns_steps)

                    update = ortho_heads.view(d_out, d_in)
                else:
                    update = zeropower_via_newtonschulz5(buf, steps=ns_steps)

                # Weight decay
                if weight_decay > 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # Update parameter: W <- W - lr * update * scale
                scale = max(1.0, (d_out / d_in) ** 0.5)
                p.data.add_(update, alpha=-lr * scale)

        return loss
