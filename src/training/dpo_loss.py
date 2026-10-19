"""Length-Normalized Direct Preference Optimization (DPO) Loss.

TÜLU 3 Reference (Section 5.1.2 & Section 5.4.1):
Standard DPO is prone to length/verbosity bias: long responses accumulate large
absolute log probabilities, causing models to favor verbose answers over concise,
high-quality responses.

TÜLU 3 adopts Length-Normalized DPO (Eq 6 in paper):
    L_DPO_norm = -log sigmoid( beta/|y_c| * log(pi_theta(y_c|x) / pi_ref(y_c|x))
                              - beta/|y_r| * log(pi_theta(y_r|x) / pi_ref(y_r|x)) )
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DPOMetrics:
    loss: float
    chosen_reward: float
    rejected_reward: float
    reward_margin: float
    accuracy: float
    chosen_length: float
    rejected_length: float


def compute_dpo_loss(
    policy_chosen_logprobs: torch.Tensor,
    policy_rejected_logprobs: torch.Tensor,
    ref_chosen_logprobs: torch.Tensor,
    ref_rejected_logprobs: torch.Tensor,
    chosen_lengths: torch.Tensor,
    rejected_lengths: torch.Tensor,
    beta: float = 5.0,
    length_normalized: bool = True,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, DPOMetrics]:
    """Compute Length-Normalized DPO loss and implicit reward metrics.

    Args:
        policy_chosen_logprobs: Log P_theta(y_c | x), shape [batch_size]
        policy_rejected_logprobs: Log P_theta(y_r | x), shape [batch_size]
        ref_chosen_logprobs: Log P_ref(y_c | x), shape [batch_size]
        ref_rejected_logprobs: Log P_ref(y_r | x), shape [batch_size]
        chosen_lengths: Token lengths |y_c|, shape [batch_size]
        rejected_lengths: Token lengths |y_r|, shape [batch_size]
        beta: KL penalty / temperature coefficient (default 5.0 for length-normalized DPO)
        length_normalized: Whether to apply TÜLU 3 length normalization
        eps: Small epsilon to prevent division by zero

    Returns:
        loss: Scalar loss tensor for backpropagation
        metrics: DPOMetrics containing rewards, margins, and accuracy
    """
    pi_chosen_diff = policy_chosen_logprobs - ref_chosen_logprobs
    pi_rejected_diff = policy_rejected_logprobs - ref_rejected_logprobs

    if length_normalized:
        # TÜLU 3 Equation 6: Length-Normalized DPO
        norm_c = chosen_lengths.clamp(min=1.0)
        norm_r = rejected_lengths.clamp(min=1.0)
        chosen_rewards = beta * (pi_chosen_diff / norm_c)
        rejected_rewards = beta * (pi_rejected_diff / norm_r)
    else:
        # Standard unnormalized DPO
        chosen_rewards = beta * pi_chosen_diff
        rejected_rewards = beta * pi_rejected_diff

    # Implicit reward difference (margin)
    reward_margin = chosen_rewards - rejected_rewards

    # DPO objective: minimize -log sigmoid(reward_margin) == F.softplus(-reward_margin)
    loss = F.softplus(-reward_margin).mean()

    # Accuracy: percentage of samples where chosen reward > rejected reward
    accuracy = (reward_margin > 0.0).float().mean()

    metrics = DPOMetrics(
        loss=loss.item(),
        chosen_reward=chosen_rewards.mean().item(),
        rejected_reward=rejected_rewards.mean().item(),
        reward_margin=reward_margin.mean().item(),
        accuracy=accuracy.item(),
        chosen_length=chosen_lengths.float().mean().item(),
        rejected_length=rejected_lengths.float().mean().item(),
    )

    return loss, metrics
