"""Offline Reference Log-Probability Caching for DPO.

TÜLU 3 Reference (Section 5):
In DPO, the reference policy pi_ref is completely frozen throughout training.
Instead of holding both the active policy model and reference model in memory
simultaneously (doubling VRAM and risking memory exhaustion on Apple Silicon),
we precompute pi_ref(y_c | x) and pi_ref(y_r | x) offline once.

This cuts active training memory by 50% and accelerates DPO steps.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from peft import PeftModel
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from src.training.sft_dataset import ensure_chat_template, tokenize_and_mask_conversation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
console = Console()


def format_dpo_turns(
    record: dict[str, Any],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Extract and format chosen and rejected multi-turn conversations."""
    prompt = record.get("prompt", "")

    # Format 1: chosen/rejected already list of message dicts
    if isinstance(record.get("chosen"), list) and record["chosen"]:
        chosen_msgs = record["chosen"]
    else:
        chosen_content = str(record.get("chosen", ""))
        chosen_msgs = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen_content},
        ]

    if isinstance(record.get("rejected"), list) and record["rejected"]:
        rejected_msgs = record["rejected"]
    else:
        rejected_content = str(record.get("rejected", ""))
        rejected_msgs = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected_content},
        ]

    return chosen_msgs, rejected_msgs


def compute_sequence_logprob(
    model: PreTrainedModel,
    input_ids: list[int],
    labels: list[int],
    device: str = "mps",
) -> tuple[float, int]:
    """Compute total log-probability across assistant tokens (where labels != -100)."""
    t_input = torch.tensor([input_ids], dtype=torch.long, device=device)
    t_labels = torch.tensor([labels], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids=t_input)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = t_labels[..., 1:].contiguous()

        vocab_size = shift_logits.size(-1)
        loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        token_losses = loss_fct(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
        )
        active_mask = shift_labels.view(-1) != -100
        active_count = active_mask.sum().item()

        if active_count == 0:
            return 0.0, 0

        total_logprob = -token_losses[active_mask].sum().item()
        return total_logprob, active_count


def cache_reference_logprobs(
    model_name_or_path: str = "checkpoints/sft",
    base_model_name: str = "unsloth/Llama-3.2-1B",
    input_path: str = "data/decontaminated/preference_clean.jsonl",
    output_path: str = "data/decontaminated/preference_with_ref_logprobs.jsonl",
    max_seq_len: int = 2048,
    limit: int | None = None,
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
) -> Path:
    """Precompute reference log-probabilities and save enriched dataset."""
    input_file = Path(input_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold cyan]Initializing Reference Model from:[/bold cyan] [yellow]{model_name_or_path}[/yellow]")

    # Check if model_name_or_path is a LoRA checkpoint directory
    adapter_cfg = Path(model_name_or_path) / "adapter_config.json"
    if adapter_cfg.exists():
        console.print(f"[cyan]Detected LoRA adapter checkpoint at {model_name_or_path}[/cyan]")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map=device,
        )
        model = PeftModel.from_pretrained(base_model, model_name_or_path)
        # Merge adapter into base weights for fast zero-overhead reference inference
        model = model.merge_and_unload()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map=device,
        )

    ensure_chat_template(tokenizer)
    model.eval()

    with open(input_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if limit is not None:
        records = records[:limit]

    console.print(f"[bold green]Caching log-probabilities for {len(records):,} preference pairs on {device}...[/bold green]")

    enriched_records: list[dict[str, Any]] = []
    chosen_logprobs: list[float] = []
    rejected_logprobs: list[float] = []
    chosen_lengths: list[int] = []
    rejected_lengths: list[int] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Caching ref logprobs", total=len(records))

        for rec in records:
            chosen_msgs, rejected_msgs = format_dpo_turns(rec)

            tok_chosen = tokenize_and_mask_conversation(chosen_msgs, tokenizer, max_length=max_seq_len)
            tok_rejected = tokenize_and_mask_conversation(rejected_msgs, tokenizer, max_length=max_seq_len)

            lp_chosen, len_chosen = compute_sequence_logprob(
                model, tok_chosen["input_ids"], tok_chosen["labels"], device=device
            )
            lp_rejected, len_rejected = compute_sequence_logprob(
                model, tok_rejected["input_ids"], tok_rejected["labels"], device=device
            )

            rec["ref_chosen_logprob"] = lp_chosen
            rec["ref_chosen_length"] = len_chosen
            rec["ref_rejected_logprob"] = lp_rejected
            rec["ref_rejected_length"] = len_rejected

            chosen_logprobs.append(lp_chosen)
            rejected_logprobs.append(lp_rejected)
            chosen_lengths.append(len_chosen)
            rejected_lengths.append(len_rejected)

            enriched_records.append(rec)
            progress.advance(task)

    # Save to output file
    with open(output_file, "w", encoding="utf-8") as f:
        for r in enriched_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary Statistics Table
    avg_c_lp = sum(chosen_logprobs) / len(chosen_logprobs) if chosen_logprobs else 0
    avg_r_lp = sum(rejected_logprobs) / len(rejected_logprobs) if rejected_logprobs else 0
    avg_c_len = sum(chosen_lengths) / len(chosen_lengths) if chosen_lengths else 0
    avg_r_len = sum(rejected_lengths) / len(rejected_lengths) if rejected_lengths else 0

    table = Table(title="[bold green]📊 Reference Model Log-Probability Cache Summary[/bold green]")
    table.add_column("Metric", style="cyan")
    table.add_column("Chosen", style="bold green")
    table.add_column("Rejected", style="bold red")

    table.add_row("Avg Log Probability", f"{avg_c_lp:.2f}", f"{avg_r_lp:.2f}")
    table.add_row("Avg Token Length", f"{avg_c_len:.1f} tokens", f"{avg_r_len:.1f} tokens")
    table.add_row("Normalized Log-Prob", f"{avg_c_lp / max(1, avg_c_len):.3f}", f"{avg_r_lp / max(1, avg_r_len):.3f}")
    console.print(table)
    console.print(f"[bold green]✅ Saved cached log-probabilities to[/bold green] [cyan]{output_file}[/cyan]")

    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Micro-TÜLU Reference Log-Prob Cacher")
    parser.add_argument("--model", type=str, default="checkpoints/sft", help="Reference model path or SFT adapter dir")
    parser.add_argument("--base-model", type=str, default="unsloth/Llama-3.2-1B", help="Base model name for LoRA adapter merge")
    parser.add_argument("--input", type=str, default="data/decontaminated/preference_clean.jsonl", help="Input preference JSONL")
    parser.add_argument("--output", type=str, default="data/decontaminated/preference_with_ref_logprobs.jsonl", help="Output enriched JSONL")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on records to process")
    args = parser.parse_args()

    cache_reference_logprobs(
        model_name_or_path=args.model,
        base_model_name=args.base_model,
        input_path=args.input,
        output_path=args.output,
        max_seq_len=args.max_seq_len,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
