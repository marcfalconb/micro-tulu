"""Unit tests for OlmoRL Engine & Reasoning Effort Budgeting (Task 1).

Covers:
1. Unnormalized advantage: A_i = r_i - mean(r) without standard deviation division.
2. Zero-gradient signal filtering: groups where all rollouts have identical rewards are pruned.
3. Asymmetric PPO clipping (clip-higher): eps_low = 0.20, eps_high = 0.28.
4. Token-level loss reduction: global sum(loss) / sum(|y_i|) eliminating length bias.
5. Reasoning effort budget control: trajectories exceeding token budget penalized with -1.0.
"""

import pytest
import torch

from src.training.loss import (
    apply_reasoning_budget_penalty,
    compute_olmorl_advantages,
    compute_olmorl_loss,
    filter_zero_gradient_groups,
)


def test_unnormalized_advantage():
    """Verifies unnormalized advantage A_i = r_i - mu (dropping std division)."""
    # Group of G=4 rollouts: 1 success (reward 1.0), 3 failures (reward 0.0)
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    # Mean is 0.25
    # Unnormalized: [1.0 - 0.25, 0.0 - 0.25, 0.0 - 0.25, 0.0 - 0.25] = [0.75, -0.25, -0.25, -0.25]
    olmorl_adv = compute_olmorl_advantages(rewards, unnormalized=True)
    expected_olmorl = torch.tensor([0.75, -0.25, -0.25, -0.25], dtype=torch.float32)
    assert torch.allclose(olmorl_adv, expected_olmorl, atol=1e-6)

    # Vanilla GRPO divides by std (sample std = 0.5, ddof=1)
    vanilla_adv = compute_olmorl_advantages(rewards, unnormalized=False)
    # (1.0 - 0.25) / (0.5 + eps) = 1.5, not 0.75
    assert not torch.allclose(olmorl_adv, vanilla_adv)
    assert olmorl_adv[0].item() == pytest.approx(0.75)
    assert vanilla_adv[0].item() > 1.4

    # Test multi-group 2D tensor [B, G]
    batch_rewards = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],  # group 0: mean=0.25
            [1.0, 1.0, 0.0, 0.0],  # group 1: mean=0.50
        ],
        dtype=torch.float32,
    )
    batch_adv = compute_olmorl_advantages(batch_rewards, group_size=4, unnormalized=True)
    expected_batch = torch.tensor(
        [
            [0.75, -0.25, -0.25, -0.25],
            [0.50, 0.50, -0.50, -0.50],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(batch_adv, expected_batch, atol=1e-6)


def test_zero_gradient_filter():
    """Verifies that prompt groups where all rewards are identical are pruned."""
    # Batch with 2 prompt groups (G=4):
    # Group 0: [1.0, 0.0, 0.0, 0.0] -> non-zero variance (informative)
    # Group 1: [0.0, 0.0, 0.0, 0.0] -> zero variance (dead gradient)
    rewards = torch.tensor(
        [
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ],
        dtype=torch.float32,
    )

    valid_mask, num_filtered = filter_zero_gradient_groups(rewards, group_size=4)
    # Group 0 is valid (True for first 4), Group 1 is filtered (False for next 4)
    expected_mask = torch.tensor([True, True, True, True, False, False, False, False])
    assert torch.equal(valid_mask, expected_mask)
    assert num_filtered == 1

    # In compute_olmorl_loss, Group 1 must be pruned and contribute 0 tokens to active_tokens
    logprobs = torch.zeros(8, 5, requires_grad=True)
    old_logprobs = torch.zeros(8, 5)
    token_mask = torch.ones(8, 5)

    loss, metrics = compute_olmorl_loss(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        rewards=rewards,
        mask=token_mask,
        group_size=4,
        filter_zero_grad=True,
    )

    # Active tokens should only count Group 0 (4 seqs * 5 tokens = 20 tokens)
    assert metrics["active_tokens"] == 20
    assert metrics["filtered_groups"] == 1
    assert metrics["valid_groups"] == 1

    # Edge case: Entire batch has zero variance -> gracefully return loss=0.0 with requires_grad
    all_zero_rewards = torch.zeros(8, dtype=torch.float32)
    loss_all_zero, metrics_all_zero = compute_olmorl_loss(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        rewards=all_zero_rewards,
        mask=token_mask,
        group_size=4,
        filter_zero_grad=True,
    )
    assert loss_all_zero.item() == 0.0
    assert metrics_all_zero["active_tokens"] == 0
    assert metrics_all_zero["filtered_groups"] == 2
    assert metrics_all_zero["valid_groups"] == 0
    # Backward pass must execute without NaN or crashing
    loss_all_zero.backward()
    assert logprobs.grad is not None


def test_asymmetric_clip_higher():
    """Verifies asymmetric PPO clipping: eps_low = 0.20, eps_high = 0.28."""
    # Group size G=2, 1 token per sequence for exact arithmetic verification
    # Sequence 0 has advantage +1.0, Sequence 1 has advantage -1.0
    # Group rewards: [2.0, 0.0] -> mean = 1.0 -> advantages = [+1.0, -1.0]
    rewards = torch.tensor([2.0, 0.0], dtype=torch.float32)
    mask = torch.ones(2, 1)

    # Case A: Ratio is 1.25 (log(1.25) approx 0.22314)
    # Under symmetric clipping (eps=0.20): clipped to 1.20
    # Under asymmetric clip-higher (eps_high=0.28): upper bound is 1.28 -> NOT clipped!
    old_logprobs = torch.zeros(2, 1)
    # Seq 0: logprob = ln(1.25), Seq 1: logprob = 0.0
    logprobs_unclipped = torch.tensor([[torch.log(torch.tensor(1.25))], [0.0]])

    _, metrics_asym = compute_olmorl_loss(
        logprobs=logprobs_unclipped,
        old_logprobs=old_logprobs,
        rewards=rewards,
        mask=mask,
        group_size=2,
        clip_eps_low=0.20,
        clip_eps_high=0.28,
    )
    # Since ratio 1.25 <= 1.28, clip fraction should be 0.0
    assert metrics_asym["clip_fraction"] == 0.0

    # With symmetric eps=0.20 (low=0.20, high=0.20), ratio 1.25 > 1.20 -> clipped!
    _, metrics_sym = compute_olmorl_loss(
        logprobs=logprobs_unclipped,
        old_logprobs=old_logprobs,
        rewards=rewards,
        mask=mask,
        group_size=2,
        clip_eps_low=0.20,
        clip_eps_high=0.20,
    )
    assert metrics_sym["clip_fraction"] == pytest.approx(0.5)

    # Case B: Ratio exceeds eps_high (e.g. ratio = 1.35)
    # Must clip at 1.28
    logprobs_clipped = torch.tensor([[torch.log(torch.tensor(1.35))], [0.0]])
    _, metrics_clipped = compute_olmorl_loss(
        logprobs=logprobs_clipped,
        old_logprobs=old_logprobs,
        rewards=rewards,
        mask=mask,
        group_size=2,
        clip_eps_low=0.20,
        clip_eps_high=0.28,
    )
    assert metrics_clipped["clip_fraction"] == pytest.approx(0.5)

    # Case C: Downward drift on positive advantage (ratio = 0.75 < 0.80)
    # Must clip at 1.0 - eps_low = 0.80
    logprobs_down = torch.tensor([[torch.log(torch.tensor(0.75))], [0.0]])
    _, metrics_down = compute_olmorl_loss(
        logprobs=logprobs_down,
        old_logprobs=old_logprobs,
        rewards=rewards,
        mask=mask,
        group_size=2,
        clip_eps_low=0.20,
        clip_eps_high=0.28,
    )
    assert metrics_down["clip_fraction"] == pytest.approx(0.5)


def test_token_level_loss_reduction():
    """Verifies loss is normalized by sum(|y_i|) across active tokens rather than sequence count."""
    # 2 sequences in a group (G=2).
    # Seq 0: length = 2 tokens, loss per token = 1.0
    # Seq 1: length = 8 tokens, loss per token = 0.0
    # Total active tokens = 10
    # Token-level normalized loss: (2 * 1.0 + 8 * 0.0) / 10 = 0.20
    # Sequence-averaged loss: (1.0 + 0.0) / 2 = 0.50
    group_size = 2
    # Rewards giving advantage -1.0 to both sequences for testing raw reduction
    # group 0: rewards [0.0, 2.0] -> adv = [-1.0, 1.0]
    rewards = torch.tensor([0.0, 2.0], dtype=torch.float32)

    # Seq 0 has 2 tokens, Seq 1 has 8 tokens (max_len = 8)
    mask = torch.zeros(2, 8)
    mask[0, :2] = 1.0  # 2 tokens
    mask[1, :8] = 1.0  # 8 tokens

    # ratio = 1.0 everywhere (old_logprobs == logprobs)
    logprobs = torch.zeros(2, 8)
    old_logprobs = torch.zeros(2, 8)

    # With ratio = 1.0 and adv = [-1.0, 1.0]:
    # Seq 0: surrogate = 1.0 * (-1.0) = -1.0 -> loss per token = -(-1.0) = +1.0
    # Seq 1: surrogate = 1.0 * (+1.0) = +1.0 -> loss per token = -(+1.0) = -1.0
    # Let's adjust rewards so Seq 1 has adv 0.0:
    # Say 2 groups of G=1: or directly test token_level_reduction on compute_olmorl_loss
    # For Seq 0 (2 tokens): adv = -1.0, loss sum = 2 * (+1.0) = 2.0
    # For Seq 1 (8 tokens): adv = 0.0, loss sum = 8 * 0.0 = 0.0
    # Let's use rewards where Seq 0 gets adv -1.0 and Seq 1 gets adv 0.0:
    # We can pass rewards=[0.0, 1.0] with mean=0.5 -> adv=[-0.5, 0.5]
    # Better: let's verify exact token-level vs sequence-level reduction:
    rewards = torch.tensor([0.0, 2.0], dtype=torch.float32)
    # Seq 0 adv = -1.0, token loss = +1.0 (2 tokens -> total 2.0)
    # Seq 1 adv = +1.0, token loss = -1.0 (8 tokens -> total -8.0)
    # Total sum of token losses = 2.0 - 8.0 = -6.0
    # Token-level loss = -6.0 / 10 active tokens = -0.60
    # Sequence-level loss: Seq 0 avg = 2.0/2 = 1.0, Seq 1 avg = -8.0/8 = -1.0 -> (1.0 + -1.0)/2 = 0.0

    loss_token, metrics_token = compute_olmorl_loss(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        rewards=rewards,
        mask=mask,
        group_size=group_size,
        token_level_reduction=True,
    )
    assert loss_token.item() == pytest.approx(-0.60)
    assert metrics_token["active_tokens"] == 10

    loss_seq, _ = compute_olmorl_loss(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        rewards=rewards,
        mask=mask,
        group_size=group_size,
        token_level_reduction=False,
    )
    assert loss_seq.item() == pytest.approx(0.00)


def test_reasoning_budget_penalty():
    """Verifies that completions exceeding max-thinking-tokens receive -1.0 penalty."""
    rewards = [1.0, 1.0, 0.0, 1.0]
    token_lengths = [45, 150, 30, 220]
    max_thinking_tokens = 100

    penalized = apply_reasoning_budget_penalty(
        rewards=rewards,
        lengths=token_lengths,
        max_thinking_tokens=max_thinking_tokens,
        penalty_reward=-1.0,
    )

    # Rollout 0: len 45 <= 100 -> stays 1.0
    # Rollout 1: len 150 > 100 -> penalized to -1.0
    # Rollout 2: len 30 <= 100 -> stays 0.0
    # Rollout 3: len 220 > 100 -> penalized to -1.0
    assert penalized == [1.0, -1.0, 0.0, -1.0]

    # Test tensor variant
    rewards_tensor = torch.tensor([1.0, 1.0, 0.0, 1.0])
    lengths_tensor = torch.tensor([45, 150, 30, 220])
    penalized_tensor = apply_reasoning_budget_penalty(
        rewards=rewards_tensor,
        lengths=lengths_tensor,
        max_thinking_tokens=max_thinking_tokens,
        penalty_reward=-1.0,
    )
    expected_tensor = torch.tensor([1.0, -1.0, 0.0, -1.0])
    assert torch.equal(penalized_tensor, expected_tensor)

    # When max_thinking_tokens is None, rewards are unmodified
    untouched = apply_reasoning_budget_penalty(
        rewards=rewards,
        lengths=token_lengths,
        max_thinking_tokens=None,
    )
    assert untouched == rewards
