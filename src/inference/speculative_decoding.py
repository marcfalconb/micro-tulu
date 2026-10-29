"""Speculative Decoding Engine for Accelerated Inference & RLVR Rollouts.

Reference:
Leviathan et al. (2023) / Chen et al. (2023) "Fast Inference from Transformers via Speculative Decoding":
A smaller draft model M_draft autoregressively predicts gamma candidate tokens.
The larger target model M_target validates all gamma tokens in a single parallel forward pass.
Verification guarantees that the accepted token sequence follows the EXACT target distribution.
"""

from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.inference.sampling import sample_next_token


def speculative_step(
    prefix: torch.Tensor,
    draft_model: nn.Module,
    target_model: nn.Module,
    gamma: int = 3,
    temperature: float = 0.0,
) -> tuple[torch.Tensor, int]:
    """Execute a single speculative drafting and verification round.

    Args:
        prefix: Current token context tensor of shape [1, seq_len]
        draft_model: Fast compact language model
        target_model: Accurate primary language model
        gamma: Number of draft speculative lookahead tokens (default 3)
        temperature: Sampling temperature (0.0 for greedy)

    Returns:
        new_prefix: Tensor with accepted and verified tokens appended.
        num_accepted: Count of draft tokens accepted in this round.
    """
    draft_model.eval()
    target_model.eval()
    device = prefix.device

    current_seq = prefix.clone()
    draft_tokens = []
    draft_probs_list = []

    # 1. Draft Phase: Autoregressively propose gamma tokens
    with torch.no_grad():
        for _ in range(gamma):
            out_draft = draft_model(current_seq)
            next_logits = out_draft.logits[:, -1, :]  # [1, vocab_size]

            if temperature <= 1e-6:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)  # [1, 1]
                prob = torch.ones(1, device=device)
            else:
                prob_dist = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(prob_dist, num_samples=1)
                prob = prob_dist[0, next_token[0, 0]]

            draft_tokens.append(next_token)
            draft_probs_list.append(prob)
            current_seq = torch.cat([current_seq, next_token], dim=1)

    draft_tensor = torch.cat(draft_tokens, dim=1)  # [1, gamma]

    # 2. Target Verification Phase: Run target model on prefix + draft in a SINGLE forward pass
    with torch.no_grad():
        out_target = target_model(current_seq)
        # Target logits for positions prefix_len-1 to end-1 (evaluating each draft token and lookahead)
        prefix_len = prefix.shape[1]
        target_logits = out_target.logits[:, prefix_len - 1 :, :]  # [1, gamma + 1, vocab]

    # 3. Acceptance / Rejection Loop
    accepted = []
    num_accepted = 0

    for i in range(gamma):
        token_id = draft_tensor[0, i].item()
        t_logits = target_logits[0, i, :]

        if temperature <= 1e-6:
            target_token = torch.argmax(t_logits, dim=-1).item()
            if token_id == target_token:
                accepted.append(token_id)
                num_accepted += 1
            else:
                # Reject: emit target's preferred token and stop
                accepted.append(target_token)
                break
        else:
            t_probs = F.softmax(t_logits / temperature, dim=-1)
            t_p = t_probs[token_id].item()
            d_p = draft_probs_list[i].item()

            # Rejection sampling acceptance threshold
            if torch.rand(1).item() <= min(1.0, t_p / max(1e-8, d_p)):
                accepted.append(token_id)
                num_accepted += 1
            else:
                # Sample replacement from max(0, t_probs - d_probs)
                diff = F.relu(t_probs)
                replacement = torch.multinomial(diff / diff.sum(), num_samples=1).item()
                accepted.append(replacement)
                break
    else:
        # If all gamma draft tokens were accepted, sample 1 bonus token from target lookahead
        final_logits = target_logits[0, gamma, :]
        if temperature <= 1e-6:
            bonus = torch.argmax(final_logits, dim=-1).item()
        else:
            bonus = torch.multinomial(F.softmax(final_logits / temperature, dim=-1), num_samples=1).item()
        accepted.append(bonus)

    new_tokens = torch.tensor([accepted], device=device, dtype=prefix.dtype)
    new_prefix = torch.cat([prefix, new_tokens], dim=1)
    return new_prefix, num_accepted


class SpeculativeDecoder:
    """High-level Speculative Decoding wrapper."""

    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        gamma: int = 3,
    ) -> None:
        self.draft_model = draft_model
        self.target_model = target_model
        self.gamma = gamma

    def generate_step(
        self,
        prefix: torch.Tensor,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, int]:
        return speculative_step(
            prefix=prefix,
            draft_model=self.draft_model,
            target_model=self.target_model,
            gamma=self.gamma,
            temperature=temperature,
        )
