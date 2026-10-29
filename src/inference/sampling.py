"""High-Performance Autoregressive Token Sampling & Filtering Kernels.

Implements:
1. Min-P Sampling (Perez et al., 2024):
   Dynamically filters tokens below p_thresh = min_p * max(p).
   Eliminates arbitrary probability cutoffs and adapts to model confidence.
2. Top-K and Top-P (Nucleus) Sampling.
3. Temperature Scaling & Greedy Decoding.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def apply_min_p(logits: torch.Tensor, min_p: float = 0.05) -> torch.Tensor:
    """Filter logits using Min-P sampling.

    Tokens with probability p < min_p * max(p) are masked to -inf.

    Args:
        logits: Tensor of shape [..., vocab_size]
        min_p: Relative threshold multiplier in [0.0, 1.0]

    Returns:
        Filtered logits tensor with masked tokens set to -inf.
    """
    if min_p <= 0.0:
        return logits

    probs = F.softmax(logits, dim=-1)
    max_probs = probs.max(dim=-1, keepdim=True).values
    threshold = max_probs * min_p

    mask = probs < threshold
    filtered_logits = logits.masked_fill(mask, -float("inf"))
    return filtered_logits


def apply_top_k_top_p(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Filter logits using Top-K and Top-P (Nucleus) sampling."""
    filtered_logits = logits.clone()

    if top_k > 0:
        top_k = min(top_k, filtered_logits.size(-1))
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = filtered_logits < torch.topk(filtered_logits, top_k, dim=-1)[0][..., -1, None]
        filtered_logits.masked_fill_(indices_to_remove, -float("inf"))

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above top_p (shift right by 1 to keep first token)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted_indices_to_remove back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        filtered_logits.masked_fill_(indices_to_remove, -float("inf"))

    return filtered_logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
) -> torch.Tensor:
    """Sample next token from logits tensor.

    Args:
        logits: Logits of shape [batch_size, vocab_size] or [vocab_size]
        temperature: Temperature multiplier (0.0 for greedy argmax)
        top_k: Top-K filter threshold
        top_p: Top-P nucleus filter threshold
        min_p: Min-P relative filter threshold

    Returns:
        Sampled token tensor of shape [batch_size]
    """
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)

    # Greedy argmax if temperature is zero
    if temperature <= 1e-6:
        return torch.argmax(logits, dim=-1)

    # Scale logits by temperature
    scaled_logits = logits / temperature

    # Apply Min-P filter
    if min_p > 0.0:
        scaled_logits = apply_min_p(scaled_logits, min_p=min_p)

    # Apply Top-K / Top-P filters
    if top_k > 0 or top_p < 1.0:
        scaled_logits = apply_top_k_top_p(scaled_logits, top_k=top_k, top_p=top_p)

    probs = F.softmax(scaled_logits, dim=-1)

    # Sample from multinomial distribution
    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return next_token
