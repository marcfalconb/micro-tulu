"""Simple Preference Optimization (SimPO) Loss.

Reference:
Meng et al. (NeurIPS 2024) "SimPO: Simple Preference Optimization with a Reference-Free Reward":
Eliminates the reference model completely by directly optimizing the length-normalized
average log-likelihood with an explicit target margin gamma:
    L_SimPO(theta) = - E_{(x, y_c, y_r)} [ log sigma( beta/|y_c| log pi_theta(y_c|x) - beta/|y_r| log pi_theta(y_r|x) - gamma ) ]
This slashes GPU memory in half compared to standard DPO and eliminates offline reference caching.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SimPOMetrics:
    simpo_loss: float
    accuracy: float
    reward_chosen: float
    reward_rejected: float
    reward_margin: float


def compute_simpo_loss(
    policy_chosen_logprobs: torch.Tensor,
    policy_rejected_logprobs: torch.Tensor,
    chosen_lengths: torch.Tensor,
    rejected_lengths: torch.Tensor,
    beta: float = 2.0,
    gamma: float = 0.5,
) -> tuple[torch.Tensor, SimPOMetrics]:
    """Compute reference-free SimPO preference loss.

    Args:
        policy_chosen_logprobs: Log P_theta(y_c | x), shape [batch_size]
        policy_rejected_logprobs: Log P_theta(y_r | x), shape [batch_size]
        chosen_lengths: Token counts |y_c|, shape [batch_size]
        rejected_lengths: Token counts |y_r|, shape [batch_size]
        beta: Reward temperature scaling hyperparameter (default 2.0)
        gamma: Target reward margin offset (default 0.5)

    Returns:
        loss: Scalar loss tensor with attached gradient.
        metrics: SimPOMetrics dataclass with diagnostic statistics.
    """
    safe_chosen_len = chosen_lengths.clamp(min=1.0)
    safe_rejected_len = rejected_lengths.clamp(min=1.0)

    # Reference-free implicit rewards normalized by length
    r_chosen = (beta / safe_chosen_len) * policy_chosen_logprobs
    r_rejected = (beta / safe_rejected_len) * policy_rejected_logprobs

    # Margin with explicit target offset gamma
    margin = (r_chosen - r_rejected) - gamma

    # - log sigma(margin) = softplus(-margin)
    loss = F.softplus(-margin).mean()

    with torch.no_grad():
        accuracy = (r_chosen > r_rejected).float().mean().item()
        reward_margin = (r_chosen - r_rejected).mean().item()
        metrics = SimPOMetrics(
            simpo_loss=loss.item(),
            accuracy=accuracy,
            reward_chosen=r_chosen.mean().item(),
            reward_rejected=r_rejected.mean().item(),
            reward_margin=reward_margin,
        )

    return loss, metrics


class SimPOLoss(nn.Module):
    """PyTorch Module wrapper for SimPO loss."""

    def __init__(self, beta: float = 2.0, gamma: float = 0.5) -> None:
        super().__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(
        self,
        policy_chosen_logprobs: torch.Tensor,
        policy_rejected_logprobs: torch.Tensor,
        chosen_lengths: torch.Tensor,
        rejected_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, SimPOMetrics]:
        return compute_simpo_loss(
            policy_chosen_logprobs=policy_chosen_logprobs,
            policy_rejected_logprobs=policy_rejected_logprobs,
            chosen_lengths=chosen_lengths,
            rejected_lengths=rejected_lengths,
            beta=self.beta,
            gamma=self.gamma,
        )
