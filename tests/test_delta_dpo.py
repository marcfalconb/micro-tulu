"""Unit tests for Delta Learning & Reasoning Traces in DPO (OLMo 3 / Dolci Think DPO)."""

import json
from pathlib import Path
import pytest
import torch

from src.data_engine.prepare_delta_dpo import (
    format_delta_dpo_pair,
    validate_think_tags,
    build_synthetic_delta_dataset,
    extract_think_and_answer,
)
from src.training.dpo_loss import compute_dpo_loss


def test_delta_pair_formatting():
    record = format_delta_dpo_pair(
        prompt="Solve: What is 2 + 2 * 3?",
        chosen_thought="Follow order of operations: 2 * 3 = 6, then 2 + 6 = 8.",
        chosen_answer="8",
        rejected_thought="Add left to right: 2 + 2 = 4, then 4 * 3 = 12.",
        rejected_answer="12",
        source="math_delta",
    )
    assert "prompt" in record
    assert "chosen" in record
    assert "rejected" in record
    assert "<think>" in record["chosen"] and "</think>" in record["chosen"]
    assert "<think>" in record["rejected"] and "</think>" in record["rejected"]
    assert "8" in record["chosen"]
    assert "12" in record["rejected"]


def test_think_tag_validation():
    valid_text = "<think>\nStep-by-step thinking.\n</think>\nFinal answer: 42."
    assert validate_think_tags(valid_text) is True

    unclosed_text = "<think>\nThinking without end tag.\nFinal answer: 42."
    assert validate_think_tags(unclosed_text) is False

    no_tag_text = "Just direct answer without think tag."
    assert validate_think_tags(no_tag_text) is False

    thought, answer = extract_think_and_answer(valid_text)
    assert "Step-by-step thinking." in thought
    assert "Final answer: 42." in answer


def test_length_normalized_delta_dpo_loss():
    # Simulate a strong capability delta:
    # Chosen (strong model) has higher logprob density per token than rejected (weak model)
    policy_c = torch.tensor([-5.0, -8.0], requires_grad=True)
    policy_r = torch.tensor([-40.0, -60.0], requires_grad=True)
    ref_c = torch.tensor([-10.0, -15.0])
    ref_r = torch.tensor([-35.0, -50.0])
    len_c = torch.tensor([50.0, 80.0])
    len_r = torch.tensor([40.0, 70.0])

    loss, metrics = compute_dpo_loss(
        policy_c, policy_r, ref_c, ref_r, len_c, len_r, beta=5.0, length_normalized=True
    )
    assert loss.item() > 0.0
    assert metrics.accuracy == 1.0  # Clear capability delta separation
    assert metrics.reward_margin > 0.0

    loss.backward()
    assert policy_c.grad is not None
    assert policy_r.grad is not None


def test_build_synthetic_delta_dataset(tmp_path: Path):
    out_file = tmp_path / "delta_dpo.jsonl"
    count = build_synthetic_delta_dataset(out_file, num_samples=5)
    assert count == 5
    assert out_file.exists()

    with open(out_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 5
    for row in lines:
        assert validate_think_tags(row["chosen"])
        assert validate_think_tags(row["rejected"])
