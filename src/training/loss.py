"""Loss formulations for Supervised Fine-Tuning.

TÜLU 3 Reference (Section 4.3.2 - Batch Aggregation):
Transformers default loss averaging weights each example equally rather than
weighting each token equally. Under gradient accumulation or variable sequence
lengths, short sequences disproportionately dominate gradients.

TÜLU 3 enforces Equal Token-Weighting (Sum-Loss):
Every non-masked target token contributes identically to the loss gradient:
    L_token_weighted = (sum_{b} sum_{i} l_{b, i}) / (total_active_tokens)
"""

from __future__ import annotations

from enum import Enum
import torch
import torch.nn as nn


class LossType(str, Enum):
    SUM_LOSS = "sum"    # TÜLU 3 Equal Token-Weighting
    MEAN_LOSS = "mean"  # Standard sample-averaged cross entropy


def compute_token_weighted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_type: LossType = LossType.SUM_LOSS,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute SFT loss with support for TÜLU 3 equal token-weighting (Sum-Loss)

    vs standard sample-averaged mean loss.

    Args:
        logits: Model output logits of shape [batch_size, seq_len, vocab_size]
        labels: Target token IDs of shape [batch_size, seq_len], with prompt/padding set to ignore_index
        loss_type: LossType.SUM_LOSS or LossType.MEAN_LOSS
        ignore_index: Index to ignore in loss calculation (-100 by default)

    Returns:
        loss: Scalar loss tensor for backpropagation
        metrics: Dictionary with loss metrics and token counts
    """
    # Shift for autoregressive causal language modeling:
    # token at position t predicts label at position t+1
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    batch_size, seq_len, vocab_size = shift_logits.shape

    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index)
    # Shape: [batch_size * seq_len]
    flat_losses = loss_fct(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
    )
    # Reshape back to [batch_size, seq_len]
    token_losses = flat_losses.view(batch_size, seq_len)
    active_mask = (shift_labels != ignore_index)

    total_active_tokens = active_mask.sum().item()
    total_loss_sum = token_losses.sum()

    if total_active_tokens == 0:
        # Fallback if no target tokens in batch (e.g. edge cases)
        dummy_loss = logits.sum() * 0.0
        return dummy_loss, {
            "sum_loss": 0.0,
            "mean_loss": 0.0,
            "active_tokens": 0,
        }

    # 1. TÜLU 3 Sum-Loss (Equal Token-Weighting)
    sum_loss = total_loss_sum / total_active_tokens

    # 2. Standard Sample-Averaged Mean-Loss
    sample_active_tokens = active_mask.sum(dim=-1).clamp(min=1)
    sample_losses = token_losses.sum(dim=-1) / sample_active_tokens
    mean_loss = sample_losses.mean()

    selected_loss = sum_loss if loss_type == LossType.SUM_LOSS else mean_loss

    metrics = {
        "loss": selected_loss.item(),
        "sum_loss": sum_loss.item(),
        "mean_loss": mean_loss.item(),
        "active_tokens": float(total_active_tokens),
        "loss_type": loss_type.value,
    }

    return selected_loss, metrics
