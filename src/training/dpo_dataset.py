"""Dataset and Dynamic Collator for Direct Preference Optimization (DPO).

Loads preference pairs with precomputed offline reference log-probabilities
and organizes them for split forward passes on Apple Silicon MPS.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.training.cache_logprobs import format_dpo_turns
from src.training.sft_dataset import ensure_chat_template, tokenize_and_mask_conversation


class DPODataset(Dataset):
    """PyTorch Dataset loading enriched preference records with cached reference log-probs."""

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
            raise FileNotFoundError(f"DPO dataset file not found: {file_path}")

        ensure_chat_template(tokenizer)

        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if limit and idx >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                # Verify reference log-probabilities exist
                if "ref_chosen_logprob" in record and "ref_rejected_logprob" in record:
                    self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        chosen_msgs, rejected_msgs = format_dpo_turns(rec)

        tok_chosen = tokenize_and_mask_conversation(
            chosen_msgs, self.tokenizer, max_length=self.max_length
        )
        tok_rejected = tokenize_and_mask_conversation(
            rejected_msgs, self.tokenizer, max_length=self.max_length
        )

        return {
            "chosen_input_ids": tok_chosen["input_ids"],
            "chosen_labels": tok_chosen["labels"],
            "rejected_input_ids": tok_rejected["input_ids"],
            "rejected_labels": tok_rejected["labels"],
            "ref_chosen_logprob": float(rec["ref_chosen_logprob"]),
            "ref_rejected_logprob": float(rec["ref_rejected_logprob"]),
            "chosen_length": int(rec.get("ref_chosen_length", tok_chosen["active_tokens"])),
            "rejected_length": int(rec.get("ref_rejected_length", tok_rejected["active_tokens"])),
        }


class DPODataCollator:
    """Dynamic collator for DPO that prepares separate batches for split forward passes."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, pad_to_multiple_of: int = 8) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def _pad_batch(self, input_ids_list: list[list[int]], labels_list: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_len = max(len(ids) for ids in input_ids_list)
        if self.pad_to_multiple_of > 0:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for ids, lbls in zip(input_ids_list, labels_list):
            pad_len = max_len - len(ids)
            batch_input_ids.append(ids + [self.pad_token_id] * pad_len)
            batch_attention_mask.append([1] * len(ids) + [0] * pad_len)
            batch_labels.append(lbls + [-100] * pad_len)

        return (
            torch.tensor(batch_input_ids, dtype=torch.long),
            torch.tensor(batch_attention_mask, dtype=torch.long),
            torch.tensor(batch_labels, dtype=torch.long),
        )

    def __call__(self, batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
        chosen_inputs, chosen_mask, chosen_labels = self._pad_batch(
            [item["chosen_input_ids"] for item in batch],
            [item["chosen_labels"] for item in batch],
        )
        rejected_inputs, rejected_mask, rejected_labels = self._pad_batch(
            [item["rejected_input_ids"] for item in batch],
            [item["rejected_labels"] for item in batch],
        )

        return {
            "chosen_input_ids": chosen_inputs,
            "chosen_attention_mask": chosen_mask,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_inputs,
            "rejected_attention_mask": rejected_mask,
            "rejected_labels": rejected_labels,
            "ref_chosen_logprobs": torch.tensor([item["ref_chosen_logprob"] for item in batch], dtype=torch.float32),
            "ref_rejected_logprobs": torch.tensor([item["ref_rejected_logprob"] for item in batch], dtype=torch.float32),
            "chosen_lengths": torch.tensor([item["chosen_length"] for item in batch], dtype=torch.float32),
            "rejected_lengths": torch.tensor([item["rejected_length"] for item in batch], dtype=torch.float32),
        }
