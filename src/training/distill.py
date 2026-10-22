r"""Multi-Teacher On-Policy Distillation (MOPD) (Kimi K3).

Reference:
Kimi K3 Technical Report (Moonshot AI, 2026 - Section 4.1.3):
"To consolidate specialized domain and reasoning-effort expert policies into a single
foundation model, we employ multi-teacher on-policy distillation with clipped advantage:
    r_opd(y_t | e, x, y_<t) = clip(log \frac{\pi_teacher(y_t | x, y_<t)}{\pi_\theta(y_t | e, x, y_<t)}, -R_max, R_max)
This token-level reverse-KL objective bounds gradient variance from teacher-student
distribution shift and seamlessly integrates with verifiable rewards."
"""

from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_mopd_reward(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    r_max: float = 2.0,
) -> torch.Tensor:
    """Compute token-level on-policy distillation reward with clipping.

    Args:
        student_logprobs: Current policy log P_theta(y_t | ...), shape [..., T]
        teacher_logprobs: Teacher policy log P_teacher(y_t | ...), shape [..., T]
        r_max: Maximum reward bound (default 2.0)

    Returns:
        Clipped distillation reward in [-r_max, r_max]
    """
    log_ratio = teacher_logprobs - student_logprobs
    return torch.clamp(log_ratio, -r_max, r_max)


class MultiTeacherOnPolicyDistillation(nn.Module):
    """Multi-Teacher On-Policy Distillation Loss module."""

    def __init__(self, r_max: float = 2.0) -> None:
        super().__init__()
        self.r_max = r_max

    def compute_loss(
        self,
        student_logprobs: torch.Tensor,
        teacher_logprobs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute MOPD policy gradient loss.

        Args:
            student_logprobs: Student policy logprobs, shape [B, T]
            teacher_logprobs: Teacher policy logprobs, shape [B, T]
            mask: Active token mask (1 for active, 0 for pad/prompt), shape [B, T]

        Returns:
            loss: Scalar distillation loss.
            metrics: Dictionary of diagnostic statistics.
        """
        # Distillation reward r_opd = clip(log(pi_teacher / pi_student), -R_max, R_max)
        # Detach reward so gradient flows through policy gradient: - r_opd * log pi_student
        r_opd = compute_mopd_reward(student_logprobs, teacher_logprobs, r_max=self.r_max).detach()

        # Policy gradient token loss
        token_loss = - (r_opd * student_logprobs)

        if mask is not None:
            active_tokens = mask.sum().clamp(min=1.0)
            loss = (token_loss * mask).sum() / active_tokens
            effective_rewards = r_opd[mask.bool()]
        else:
            loss = token_loss.mean()
            effective_rewards = r_opd

        clipped_count = ((r_opd.abs() >= (self.r_max - 1e-5)) & (mask.bool() if mask is not None else True)).sum().item()
        total_tokens = int(mask.sum().item()) if mask is not None else r_opd.numel()
        clipped_fraction = clipped_count / max(1, total_tokens)

        metrics = {
            "mopd_loss": loss.item(),
            "mean_opd_reward": effective_rewards.mean().item() if effective_rewards.numel() > 0 else 0.0,
            "clipped_fraction": float(clipped_fraction),
            "r_max": self.r_max,
        }

        return loss, metrics
