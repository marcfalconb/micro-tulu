"""Dataset and Collator for Supervised Fine-Tuning with TÜLU 3 prompt masking.

TÜLU 3 Reference:
- Jinja2 Chat Template from Appendix B.3 (Figure 27).
- Loss computed exclusively on assistant tokens (prompt tokens masked with -100).
- Dynamic batch padding to reduce waste on Apple Silicon Unified Memory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

# Exact TÜLU 3 Chat Template from paper Figure 27
TULU3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|system|>\n' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|user|>\n' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{% if not loop.last %}"
    "{{ '<|assistant|>\n' + message['content'] + eos_token + '\n' }}"
    "{% else %}"
    "{{ '<|assistant|>\n' + message['content'] + eos_token }}"
    "{% endif %}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|assistant|>\n' }}"
    "{% endif %}"
)


def ensure_chat_template(tokenizer: PreTrainedTokenizerBase) -> None:
    """Ensure the tokenizer has the TÜLU 3 chat template and valid pad token."""
    if tokenizer.chat_template is None:
        tokenizer.chat_template = TULU3_CHAT_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id


def tokenize_and_mask_conversation(
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 2048,
) -> dict[str, Any]:
    """Tokenize a multi-turn conversation and mask user/system prompt tokens with -100."""
    ensure_chat_template(tokenizer)

    # Render entire conversation
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]

    labels = [-100] * len(full_ids)

    # For each assistant turn, uncover its tokens in labels
    # We walk the conversation turns incrementally
    for i, msg in enumerate(messages):
        if msg.get("role") in ("assistant", "gpt"):
            # Text up to the start of this assistant turn (with generation prompt)
            history_messages = messages[:i]
            prefix_text = tokenizer.apply_chat_template(
                history_messages, tokenize=False, add_generation_prompt=True
            ) if history_messages else ""
            prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False) if prefix_text else []
            start_idx = len(prefix_ids)

            # Text including this assistant turn
            current_messages = messages[: i + 1]
            turn_text = tokenizer.apply_chat_template(current_messages, tokenize=False)
            turn_ids = tokenizer.encode(turn_text, add_special_tokens=False)
            end_idx = min(len(turn_ids), len(full_ids))

            for pos in range(start_idx, end_idx):
                labels[pos] = full_ids[pos]

    active_tokens = sum(1 for l in labels if l != -100)

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "active_tokens": active_tokens,
    }


class SFTDataset(Dataset):
    """PyTorch Dataset loading decontaminated JSONL conversations for SFT."""

    def __init__(
        self,
        jsonl_path: Path | str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2048,
        limit: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records: list[dict[str, Any]] = []

        file_path = Path(jsonl_path)
        if not file_path.exists():
            raise FileNotFoundError(f"SFT dataset file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if limit and idx >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "messages" in record and isinstance(record["messages"], list):
                    # Filter empty or invalid messages
                    valid_msgs = [m for m in record["messages"] if m.get("content", "").strip()]
                    if valid_msgs:
                        self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        return tokenize_and_mask_conversation(
            record["messages"],
            self.tokenizer,
            max_length=self.max_length,
        )


class SFTDataCollator:
    """Dynamic padding collator for SFT.

    Pads input_ids with pad_token_id, attention_mask with 0, and labels with -100.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, pad_to_multiple_of: int = 8) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, batch: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)
        if self.pad_to_multiple_of > 0:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for item in batch:
            input_ids = item["input_ids"]
            labels = item["labels"]
            pad_len = max_len - len(input_ids)

            batch_input_ids.append(input_ids + [self.pad_token_id] * pad_len)
            batch_attention_mask.append([1] * len(input_ids) + [0] * pad_len)
            batch_labels.append(labels + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }
