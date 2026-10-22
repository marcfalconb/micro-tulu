"""Attention Residuals (AttnRes & Block AttnRes).

Reference:
Kimi K3 Technical Report (Moonshot AI, 2026 - Section 2.2):
"Standard residual connections compress all prior information into a single state
h_l = h_{l-1} + f_l(h_{l-1}) over depth — an additive bottleneck.
Attention Residuals (AttnRes) applies attention to depth: each layer selectively
retrieves representations from all preceding layers rather than accumulating them uniformly.
Under Block AttnRes, we partition L layers into N blocks. Layer outputs within a block
are summed, and full softmax attention with RMSNorm keys is applied across block representations:
    alpha_{i -> l} = softmax(w_l^T RMSNorm(k_i))
    h_l = sum_i alpha_{i -> l} * v_i"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockAttentionResiduals(nn.Module):
    """Block Attention Residuals module for selective depth information access."""

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        d_model: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.d_model = d_model
        self.eps = eps

        # Learnable pseudo-query vectors for each layer: w_l in R^d
        self.query_vectors = nn.Parameter(torch.randn(num_layers, d_model) * 0.02)

        # Scale factor for query-key dot product
        self.scale = 1.0 / (d_model ** 0.5)

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Root Mean Square Normalization across feature dimension."""
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps)

    def forward(
        self,
        layer_idx: int,
        embedding: torch.Tensor,
        prior_layer_outputs: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute selective depth attention representation for layer l.

        Args:
            layer_idx: Current layer index (0 to num_layers - 1).
            embedding: Initial token embedding h1 of shape [B, T, D].
            prior_layer_outputs: List of output tensors from previous layers [f0(h0), ..., f_{l-1}(h_{l-1})].

        Returns:
            h_l: Attended depth representation of shape [B, T, D].
            weights: Attention weights of shape [B, T, num_sources].
        """
        B, T, D = embedding.shape

        if layer_idx == 0 or len(prior_layer_outputs) == 0:
            # First layer attends solely to embedding
            weights = torch.ones(B, T, 1, device=embedding.device, dtype=embedding.dtype)
            return embedding, weights

        layers_per_block = max(1, self.num_layers // self.num_blocks)
        curr_block_idx = layer_idx // layers_per_block
        layer_in_block = layer_idx % layers_per_block

        # Sources list begins with token embedding b0 = h1
        source_blocks = [embedding]

        # Aggregate prior completed blocks by summation: b_n = sum_{j in B_n} f_j(h_j)
        for b in range(curr_block_idx):
            start_l = b * layers_per_block
            end_l = start_l + layers_per_block
            block_sum = torch.stack(prior_layer_outputs[start_l:end_l], dim=0).sum(dim=0)
            source_blocks.append(block_sum)

        # If we are past the first layer of current block, include partial sum within current block
        if layer_in_block > 0:
            start_l = curr_block_idx * layers_per_block
            curr_intra_sum = torch.stack(prior_layer_outputs[start_l:layer_idx], dim=0).sum(dim=0)
            source_blocks.append(curr_intra_sum)

        # Stack sources into V: shape [K, B, T, D] -> [B, T, K, D]
        # K is the number of available block sources
        sources = torch.stack(source_blocks, dim=2)  # [B, T, K, D]
        K = sources.shape[2]

        # Normalized keys: RMSNorm(k_i)
        norm_keys = self._rms_norm(sources)  # [B, T, K, D]

        # Learnable layer query: w_l of shape [D]
        q_l = self.query_vectors[layer_idx]  # [D]

        # Compute dot product: (q_l . norm_k) * scale -> [B, T, K]
        scores = torch.einsum("d, btkd -> btk", q_l, norm_keys) * self.scale

        # Softmax over source blocks
        weights = F.softmax(scores, dim=-1)  # [B, T, K]

        # Weighted combination of values: sum_i alpha_i * v_i
        h_l = torch.einsum("btk, btkd -> btd", weights, sources)  # [B, T, D]

        return h_l, weights
