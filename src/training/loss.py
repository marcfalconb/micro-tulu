"""Loss formulations for Supervised Fine-Tuning and Reinforcement Learning.

TÜLU 3 Reference (Section 4.3.2 - Batch Aggregation):
Transformers default loss averaging weights each example equally rather than
weighting each token equally. Under gradient accumulation or variable sequence
lengths, short sequences disproportionately dominate gradients.

TÜLU 3 enforces Equal Token-Weighting (Sum-Loss):
Every non-masked target token contributes identically to the loss gradient:
    L_token_weighted = (sum_{b} sum_{i} l_{b, i}) / (total_active_tokens)

OLMo 3 / OlmoRL Reference (Section 2.3 & Kimi K3 Section 3.2):
OlmoRL upgrades GRPO with:
1. Unnormalized centered advantage: A_i = r_i - mu (dropping std division).
2. Zero-gradient signal filtering: prune groups where all rewards are equal.
3. Asymmetric PPO clip-higher: eps_low = 0.20, eps_high = 0.28.
4. Token-level loss normalization: sum(L) / sum(|y_i|) across active batch.
5. Reasoning effort budgeting: trajectories exceeding max thinking tokens receive -1.0.
"""

from __future__ import annotations

from enum import Enum
from typing import Any
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
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    batch_size, seq_len, vocab_size = shift_logits.shape

    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index)
    flat_losses = loss_fct(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
    )
    token_losses = flat_losses.view(batch_size, seq_len)
    active_mask = (shift_labels != ignore_index)

    total_active_tokens = active_mask.sum().item()
    total_loss_sum = token_losses.sum()

    if total_active_tokens == 0:
        dummy_loss = logits.sum() * 0.0
        return dummy_loss, {
            "sum_loss": 0.0,
            "mean_loss": 0.0,
            "active_tokens": 0,
        }

    sum_loss = total_loss_sum / total_active_tokens

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


def compute_olmorl_advantages(
    rewards: torch.Tensor,
    group_size: int = 4,
    unnormalized: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute Group-Relative advantages.

    OLMo 3 / OlmoRL Reference (Section 2.3 - No Standard Deviation Normalization):
    In vanilla GRPO:
        A_i = (r_i - mu) / (sigma + eps)
    When standard deviation sigma is near zero (e.g. 1 correct out of 16),
    advantage explodes, introducing severe difficulty bias on outlier questions.

    OlmoRL computes unnormalized centered advantage:
        A_i = r_i - mean({r_j}_{j=1}^G)
    dropping sigma entirely to ensure questions contribute gradient proportional
    to their actual reward delta.

    Args:
        rewards: Tensor of rewards, either 1D [N] or 2D [B, G].
        group_size: Number of rollouts per prompt group G.
        unnormalized: If True, computes A = r - mu (OlmoRL).
                      If False, computes A = (r - mu) / (sigma + eps) (Vanilla GRPO).
        eps: Small constant for numerical stability in vanilla mode.

    Returns:
        advantages: Tensor of advantages matching the shape of rewards.
    """
    orig_shape = rewards.shape
    if rewards.dim() == 1:
        if rewards.numel() % group_size != 0:
            raise ValueError(f"rewards length {rewards.numel()} must be divisible by group_size {group_size}")
        rewards_2d = rewards.view(-1, group_size)
    elif rewards.dim() == 2:
        rewards_2d = rewards
    else:
        raise ValueError(f"rewards must be 1D or 2D tensor, got {rewards.dim()}D")

    mean_r = rewards_2d.mean(dim=-1, keepdim=True)
    if unnormalized:
        adv = rewards_2d - mean_r
    else:
        if rewards_2d.shape[-1] > 1:
            std_r = rewards_2d.std(dim=-1, keepdim=True, unbiased=True)
        else:
            std_r = rewards_2d.std(dim=-1, keepdim=True, unbiased=False)
        adv = (rewards_2d - mean_r) / (std_r + eps)

    return adv.view(orig_shape)


def filter_zero_gradient_groups(
    rewards: torch.Tensor,
    group_size: int = 4,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, int]:
    """Filter out prompt groups where all rollouts have identical rewards.

    OLMo 3 / OlmoRL Reference (Section 2.3 - Zero-Gradient Signal Filtering):
    Groups where all completions have identical rewards (all passed or all failed)
    produce strictly zero advantage (A_i = 0) and zero gradient. OlmoRL filters
    these groups before backprop to avoid dead computation and preserve valid
    batch statistics.

    Args:
        rewards: Tensor of rewards, 1D [N] or 2D [B, G].
        group_size: Number of rollouts per prompt group.
        eps: Threshold for considering reward spread zero.

    Returns:
        valid_mask: Boolean tensor matching rewards shape (True for active/informative rollouts).
        num_filtered: Number of prompt groups filtered out.
    """
    orig_shape = rewards.shape
    if rewards.dim() == 1:
        if rewards.numel() % group_size != 0:
            raise ValueError(f"rewards length {rewards.numel()} must be divisible by group_size {group_size}")
        rewards_2d = rewards.view(-1, group_size)
    elif rewards.dim() == 2:
        rewards_2d = rewards
    else:
        raise ValueError(f"rewards must be 1D or 2D tensor, got {rewards.dim()}D")

    spread = rewards_2d.max(dim=-1).values - rewards_2d.min(dim=-1).values
    is_valid_group = spread > eps
    num_filtered = int((~is_valid_group).sum().item())

    valid_mask_2d = is_valid_group.unsqueeze(-1).expand(-1, rewards_2d.shape[-1])
    return valid_mask_2d.reshape(orig_shape), num_filtered


def apply_reasoning_budget_penalty(
    rewards: list[float] | torch.Tensor,
    lengths: list[int] | torch.Tensor,
    max_thinking_tokens: int | None,
    penalty_reward: float = -1.0,
) -> list[float] | torch.Tensor:
    """Apply test-time reasoning effort budget penalty (Kimi K3 Section 3.2).

    If generated reasoning trajectory length exceeds the budget threshold,
    the task reward is overridden to penalty_reward (-1.0).

    Args:
        rewards: List or Tensor of rollout rewards.
        lengths: List or Tensor of generated token counts.
        max_thinking_tokens: Max allowed generation tokens (or None for unconstrained).
        penalty_reward: Override reward for over-budget trajectories (default -1.0).

    Returns:
        Penalized rewards matching input type.
    """
    if max_thinking_tokens is None:
        return rewards

    if isinstance(rewards, torch.Tensor):
        new_rewards = rewards.clone()
        lengths_t = lengths if isinstance(lengths, torch.Tensor) else torch.tensor(lengths, device=rewards.device)
        mask_exceeded = lengths_t > max_thinking_tokens
        new_rewards[mask_exceeded] = penalty_reward
        return new_rewards
    else:
        new_rewards = list(rewards)
        for i, l in enumerate(lengths):
            if l > max_thinking_tokens:
                new_rewards[i] = penalty_reward
        return new_rewards


def compute_olmorl_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor | None,
    rewards: torch.Tensor,
    mask: torch.Tensor | None = None,
    group_size: int = 4,
    clip_eps_low: float = 0.20,
    clip_eps_high: float = 0.28,
    filter_zero_grad: bool = True,
    token_level_reduction: bool = True,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute OlmoRL Policy Gradient Loss (OLMo 3 / DAPO / Dr. GRPO).

    Mechanisms:
    1. Unnormalized advantage: A_i = r_i - mu (dropping sigma).
    2. Zero-gradient signal filtering: groups with identical rewards are pruned.
    3. Asymmetric PPO clipping: eps_low = 0.20, eps_high = 0.28.
    4. Token-level loss reduction: loss normalized by sum(|y_i|) across active batch.

    Args:
        logprobs: Current policy log-probabilities [N, T] or [N].
        old_logprobs: Rollout policy log-probabilities [N, T] or [N] (or None).
        rewards: Rollout rewards [N] or [B, G].
        mask: Token active mask [N, T] (1 for target tokens, 0 for prompt/pad).
        group_size: Number of rollouts per prompt group G.
        clip_eps_low: Lower bound clipping parameter (default 0.20).
        clip_eps_high: Upper bound clipping parameter (default 0.28).
        filter_zero_grad: Whether to filter out zero-gradient groups.
        token_level_reduction: If True, normalize by total active tokens across batch.
        eps: Numerical stability constant.

    Returns:
        loss: Scalar loss tensor.
        metrics: Dictionary of diagnostic metrics.
    """
    if logprobs.dim() == 1:
        logprobs_2d = logprobs.unsqueeze(-1)
    else:
        logprobs_2d = logprobs

    if old_logprobs is None:
        old_logprobs_2d = logprobs_2d.detach()
    elif old_logprobs.dim() == 1:
        old_logprobs_2d = old_logprobs.unsqueeze(-1)
    else:
        old_logprobs_2d = old_logprobs

    if mask is None:
        mask_2d = torch.ones_like(logprobs_2d)
    elif mask.dim() == 1:
        mask_2d = mask.unsqueeze(-1)
    else:
        mask_2d = mask.float()

    num_seqs = logprobs_2d.shape[0]
    if num_seqs % group_size != 0:
        raise ValueError(f"Total sequences {num_seqs} must be divisible by group_size {group_size}")
    num_groups = num_seqs // group_size

    # 1. Unnormalized advantage: A_i = r_i - mu
    advantages = compute_olmorl_advantages(
        rewards, group_size=group_size, unnormalized=True, eps=eps
    ).view(num_seqs, 1)

    # 2. Zero-gradient filtering
    valid_seq_mask, num_filtered = filter_zero_gradient_groups(
        rewards, group_size=group_size, eps=eps
    )
    valid_seq_mask = valid_seq_mask.view(num_seqs, 1)
    valid_groups = num_groups - num_filtered

    if filter_zero_grad:
        effective_mask = mask_2d * valid_seq_mask.float()
    else:
        effective_mask = mask_2d

    total_active_tokens = int(effective_mask.sum().item())

    # Handle fully-filtered batch edge case
    if total_active_tokens == 0:
        dummy_loss = (logprobs_2d * 0.0).sum()
        return dummy_loss, {
            "loss": 0.0,
            "clip_fraction": 0.0,
            "mean_ratio": 1.0,
            "active_tokens": 0,
            "filtered_groups": num_filtered,
            "valid_groups": valid_groups,
            "mean_advantage": 0.0,
            "std_advantage": 0.0,
        }

    # 3. Asymmetric PPO clipping
    log_ratio = logprobs_2d - old_logprobs_2d
    ratio = torch.exp(log_ratio)

    surr1 = ratio * advantages
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps_low, 1.0 + clip_eps_high)
    surr2 = clipped_ratio * advantages

    token_surr = torch.minimum(surr1, surr2)
    token_loss = -token_surr

    masked_loss = token_loss * effective_mask

    # 4. Token-level vs Sequence-level reduction
    if token_level_reduction:
        loss = masked_loss.sum() / total_active_tokens
    else:
        seq_tokens = effective_mask.sum(dim=-1).clamp(min=1.0)
        seq_loss = masked_loss.sum(dim=-1) / seq_tokens
        num_active_seqs = valid_seq_mask.sum().item() if filter_zero_grad else float(num_seqs)
        loss = seq_loss.sum() / max(1.0, float(num_active_seqs))

    # Diagnostic metrics
    active_bool = effective_mask.bool()
    active_ratios = ratio[active_bool]
    mean_ratio = active_ratios.mean().item() if active_ratios.numel() > 0 else 1.0

    is_clipped = (ratio < (1.0 - clip_eps_low)) | (ratio > (1.0 + clip_eps_high))
    clip_count = (is_clipped & active_bool).sum().item()
    clip_fraction = clip_count / total_active_tokens if total_active_tokens > 0 else 0.0

    active_adv = advantages.expand_as(effective_mask)[active_bool]
    mean_adv = active_adv.mean().item() if active_adv.numel() > 0 else 0.0
    std_adv = active_adv.std().item() if active_adv.numel() > 1 else 0.0

    metrics = {
        "loss": loss.item(),
        "clip_fraction": float(clip_fraction),
        "mean_ratio": float(mean_ratio),
        "active_tokens": total_active_tokens,
        "filtered_groups": num_filtered,
        "valid_groups": valid_groups,
        "mean_advantage": float(mean_adv),
        "std_advantage": float(std_adv),
    }

    return loss, metrics
