"""Unit tests for Inference Accelerators (Min-p, Speculative Decoding) and SimPO."""

import pytest
import torch
import torch.nn as nn

from src.inference.sampling import sample_next_token, apply_min_p, apply_top_k_top_p
from src.inference.speculative_decoding import speculative_step, SpeculativeDecoder
from src.training.simpo_loss import compute_simpo_loss, SimPOLoss


def test_min_p_sampling():
    # Logits where top token has 0.8 probability, second has 0.15, third has 0.05
    logits = torch.tensor([[10.0, 8.0, 5.0, 1.0, -10.0]])
    # min_p = 0.2 means threshold is 0.2 * max_prob
    filtered_logits = apply_min_p(logits, min_p=0.2)

    probs = torch.softmax(filtered_logits, dim=-1)
    # The lowest logits (-10.0, 1.0) must be masked to -inf (0 prob)
    assert probs[0, 3].item() == 0.0
    assert probs[0, 4].item() == 0.0
    assert probs[0, 0].item() > 0.0


def test_sample_next_token_greedy_and_temperature():
    logits = torch.tensor([[1.0, 5.0, 2.0]])
    token_greedy = sample_next_token(logits, temperature=0.0)
    assert token_greedy.item() == 1  # argmax is index 1

    token_temp = sample_next_token(logits, temperature=0.8, min_p=0.1)
    assert token_temp.ndim == 1


def test_speculative_verification_step():
    # Mock draft and target models
    vocab_size = 50
    d_model = 32

    # Create dummy tiny models
    class DummyLM(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, d_model)
            self.head = nn.Linear(d_model, vocab_size)

        def forward(self, input_ids):
            h = self.emb(input_ids)
            logits = self.head(h)
            return type("Output", (), {"logits": logits})()

    target_model = DummyLM(vocab_size)
    # When draft model shares weights with target, all gamma draft tokens are accepted
    decoder_aligned = SpeculativeDecoder(draft_model=target_model, target_model=target_model, gamma=3)

    prefix_tokens = torch.tensor([[1, 5, 10, 20]], dtype=torch.long)
    accepted_tokens, num_accepted = decoder_aligned.generate_step(prefix_tokens, temperature=0.0)

    # All 3 draft tokens accepted + 1 lookahead token = 4 new tokens
    assert num_accepted == 3
    assert accepted_tokens.shape[1] == prefix_tokens.shape[1] + 4

    # When draft model has different weights, it emits at least 1 corrected token from target
    different_draft = DummyLM(vocab_size)
    decoder_diff = SpeculativeDecoder(draft_model=different_draft, target_model=target_model, gamma=3)
    diff_tokens, diff_accepted = decoder_diff.generate_step(prefix_tokens, temperature=0.0)
    assert diff_tokens.shape[1] >= prefix_tokens.shape[1] + 1


def test_simpo_loss_reference_free():
    # Chosen has higher logprob than rejected
    policy_c = torch.tensor([-5.0, -8.0], requires_grad=True)
    policy_r = torch.tensor([-30.0, -45.0], requires_grad=True)
    len_c = torch.tensor([50.0, 80.0])
    len_r = torch.tensor([40.0, 70.0])
    gamma = 0.5
    beta = 2.0

    loss, metrics = compute_simpo_loss(
        policy_chosen_logprobs=policy_c,
        policy_rejected_logprobs=policy_r,
        chosen_lengths=len_c,
        rejected_lengths=len_r,
        beta=beta,
        gamma=gamma,
    )

    assert loss.item() > 0.0
    assert metrics.accuracy == 1.0
    assert metrics.reward_margin > 0.0

    loss.backward()
    assert policy_c.grad is not None
    assert policy_r.grad is not None


def test_simpo_module_wrapper():
    simpo = SimPOLoss(beta=2.5, gamma=0.8)
    policy_c = torch.tensor([-10.0], requires_grad=True)
    policy_r = torch.tensor([-25.0], requires_grad=True)
    len_c = torch.tensor([20.0])
    len_r = torch.tensor([20.0])

    loss, metrics = simpo(policy_c, policy_r, len_c, len_r)
    assert metrics.reward_margin > 0.0
    assert metrics.simpo_loss > 0.0
