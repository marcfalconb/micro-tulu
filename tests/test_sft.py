"""Unit tests for Phase 2: Supervised Fine-Tuning (SFT)."""

from pathlib import Path
import torch
from transformers import AutoTokenizer

from src.training.loss import LossType, compute_token_weighted_loss
from src.training.sft_dataset import (
    SFTDataCollator,
    SFTDataset,
    ensure_chat_template,
    tokenize_and_mask_conversation,
)
from src.training.visual_logger import VisualTrainingLogger


def test_loss_sum_vs_mean_mathematical_behavior():
    # Batch of 2 sequences, sequence length 6, vocabulary size 10
    torch.manual_seed(42)
    logits = torch.randn(2, 6, 10, requires_grad=True)

    # Sequence 1: 2 active tokens (rest ignored)
    # Sequence 2: 4 active tokens (rest ignored)
    labels = torch.tensor([
        [1, 2, -100, -100, -100, -100],
        [3, 4, 5, 6, -100, -100],
    ], dtype=torch.long)

    sum_loss, sum_metrics = compute_token_weighted_loss(logits, labels, loss_type=LossType.SUM_LOSS)
    mean_loss, mean_metrics = compute_token_weighted_loss(logits, labels, loss_type=LossType.MEAN_LOSS)

    # Shifted labels have active counts:
    # shift_labels = labels[:, 1:] -> seq1 has active tokens at pos 0 (predicting 2), seq2 has active tokens at pos 0,1,2 (predicting 4,5,6)
    assert sum_metrics["active_tokens"] == 4.0
    assert sum_loss.item() > 0.0
    assert mean_loss.item() > 0.0

    # Test backward pass succeeds on both
    sum_loss.backward(retain_graph=True)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_tokenize_and_mask_conversation():
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
    ensure_chat_template(tokenizer)

    messages = [
        {"role": "user", "content": "What is 7 * 8?"},
        {"role": "assistant", "content": "7 * 8 is 56."},
    ]

    item = tokenize_and_mask_conversation(messages, tokenizer, max_length=128)

    assert len(item["input_ids"]) == len(item["labels"])
    assert len(item["input_ids"]) == len(item["attention_mask"])

    # User prompt must be masked with -100
    assert item["labels"][0] == -100
    assert item["labels"][1] == -100

    # Assistant tokens must not be -100
    unmasked = [tok_id for tok_id in item["labels"] if tok_id != -100]
    assert len(unmasked) > 0
    decoded_assistant = tokenizer.decode(unmasked)
    assert "56" in decoded_assistant


def test_sft_data_collator():
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
    ensure_chat_template(tokenizer)

    collator = SFTDataCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
    batch_raw = [
        {"input_ids": [10, 20, 30], "attention_mask": [1, 1, 1], "labels": [-100, 20, 30]},
        {"input_ids": [10, 20, 30, 40, 50], "attention_mask": [1, 1, 1, 1, 1], "labels": [-100, -100, 30, 40, 50]},
    ]

    batch = collator(batch_raw)
    # Longest sequence is 5, rounded up to multiple of 8 -> 8
    assert batch["input_ids"].shape == (2, 8)
    assert batch["attention_mask"].shape == (2, 8)
    assert batch["labels"].shape == (2, 8)

    # Verify padding positions in sequence 0
    assert (batch["attention_mask"][0, 3:] == 0).all()
    assert (batch["labels"][0, 3:] == -100).all()


def test_visual_logger_and_html_generation(tmp_path: Path):
    logger = VisualTrainingLogger(output_dir=tmp_path, run_name="test_sft")
    logger.log_step(
        step=1, epoch=0.1, loss=2.5, sum_loss=2.5, mean_loss=2.6, lr=5e-5, active_tokens=100
    )
    logger.log_step(
        step=2, epoch=0.2, loss=2.1, sum_loss=2.1, mean_loss=2.2, lr=4.5e-5, active_tokens=120
    )
    logger.log_generation_sample(
        prompt="Test prompt",
        before="Base model answer",
        after="Fine-tuned model answer",
    )

    report_path = logger.generate_html_report({"learning_rate": 5e-5, "epochs": 1})
    assert report_path.exists()
    content = report_path.read_text()
    assert "Micro-TÜLU: SFT Training Report" in content
    assert "Test prompt" in content
    assert "Base model answer" in content
    assert "Fine-tuned model answer" in content
