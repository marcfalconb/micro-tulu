"""Unit tests for Phase 3: Length-Normalized DPO and Reference Log-Prob Caching."""

import json
from pathlib import Path
import torch
from transformers import AutoTokenizer

from src.training.dpo_dataset import DPODataCollator, DPODataset
from src.training.dpo_loss import compute_dpo_loss
from src.training.dpo_visual_logger import DPOVisualLogger
from src.training.train_dpo import compute_batch_logprobs


def test_dpo_loss_length_normalized_vs_unnormalized():
    # Batch size 2
    policy_c = torch.tensor([-10.0, -20.0], requires_grad=True)
    policy_r = torch.tensor([-30.0, -40.0], requires_grad=True)
    ref_c = torch.tensor([-15.0, -25.0])
    ref_r = torch.tensor([-20.0, -35.0])
    len_c = torch.tensor([10.0, 20.0])
    len_r = torch.tensor([30.0, 40.0])

    # 1. Length-Normalized DPO (TÜLU 3)
    loss_norm, metrics_norm = compute_dpo_loss(
        policy_c, policy_r, ref_c, ref_r, len_c, len_r, beta=5.0, length_normalized=True
    )
    assert loss_norm.item() > 0.0
    assert metrics_norm.accuracy >= 0.0

    loss_norm.backward()
    assert policy_c.grad is not None
    assert policy_r.grad is not None
    assert torch.isfinite(policy_c.grad).all()

    # 2. Standard Unnormalized DPO
    policy_c2 = torch.tensor([-10.0, -20.0], requires_grad=True)
    policy_r2 = torch.tensor([-30.0, -40.0], requires_grad=True)
    loss_std, metrics_std = compute_dpo_loss(
        policy_c2, policy_r2, ref_c, ref_r, len_c, len_r, beta=0.1, length_normalized=False
    )
    assert loss_std.item() > 0.0


def test_compute_batch_logprobs():
    # Batch size 2, seq_len 4, vocab 8
    logits = torch.randn(2, 4, 8)
    labels = torch.tensor([
        [-100, 2, 4, -100],
        [-100, -100, 3, 5],
    ], dtype=torch.long)

    logprobs = compute_batch_logprobs(logits, labels)
    assert logprobs.shape == (2,)
    # Log probabilities must be strictly negative
    assert (logprobs < 0.0).all()


def test_dpo_dataset_and_collator(tmp_path: Path):
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")

    # Mock enriched preference records
    records = [
        {
            "id": "pair_1",
            "prompt": "What is 2+2?",
            "chosen": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}],
            "rejected": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "5"}],
            "ref_chosen_logprob": -5.2,
            "ref_rejected_logprob": -9.8,
            "ref_chosen_length": 3,
            "ref_rejected_length": 3,
        },
        {
            "id": "pair_2",
            "prompt": "Say hello.",
            "chosen": [{"role": "user", "content": "Say hello."}, {"role": "assistant", "content": "Hello!"}],
            "rejected": [{"role": "user", "content": "Say hello."}, {"role": "assistant", "content": "Bye."}],
            "ref_chosen_logprob": -4.1,
            "ref_rejected_logprob": -8.0,
            "ref_chosen_length": 4,
            "ref_rejected_length": 4,
        },
    ]

    jsonl_path = tmp_path / "pref.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    dataset = DPODataset(jsonl_path, tokenizer, max_length=128)
    assert len(dataset) == 2

    collator = DPODataCollator(tokenizer, pad_to_multiple_of=8)
    batch = collator([dataset[0], dataset[1]])

    assert "chosen_input_ids" in batch
    assert "rejected_input_ids" in batch
    assert "ref_chosen_logprobs" in batch
    assert batch["chosen_input_ids"].shape[0] == 2
    assert batch["ref_chosen_logprobs"].shape == (2,)


def test_dpo_visual_logger_and_report(tmp_path: Path):
    logger = DPOVisualLogger(output_dir=tmp_path, run_name="test_dpo")
    logger.log_step(
        step=1, epoch=0.2, loss=0.693, chosen_reward=0.1, rejected_reward=-0.1,
        reward_margin=0.2, accuracy=0.75, lr=5e-6
    )
    logger.log_step(
        step=2, epoch=0.4, loss=0.550, chosen_reward=0.3, rejected_reward=-0.2,
        reward_margin=0.5, accuracy=0.85, lr=4e-6
    )

    report_path = logger.generate_html_report({"beta": 5.0, "lr": 5e-6})
    assert report_path.exists()
    content = report_path.read_text()
    assert "Micro-TÜLU: Length-Normalized DPO Report" in content
    assert "Reward Margin" in content
    assert "0.5500" in content
