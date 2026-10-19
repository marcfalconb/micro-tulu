"""Direct Preference Optimization (DPO) Training with Length Normalization.

TÜLU 3 Reference (Section 5.1.2 & Section 5.4.1):
1. Length-Normalized DPO objective (Eq 6 in paper) to eliminate verbosity bias.
2. Offline reference log-probability caching (halves active GPU/MPS memory).
3. Split forward passes (decouples chosen and rejected batches to minimize peak activation VRAM).
4. Rich terminal metrics and standalone interactive HTML report.
"""

from __future__ import annotations

import argparse
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
from peft import LoraConfig, PeftModel, get_peft_model
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_cosine_schedule_with_warmup,
)

from src.training.dpo_dataset import DPODataCollator, DPODataset
from src.training.dpo_loss import compute_dpo_loss
from src.training.dpo_visual_logger import DPOVisualLogger
from src.training.sft_dataset import ensure_chat_template

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
console = Console()


def compute_batch_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute per-sequence target token log-probabilities vectorized across batch."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    vocab_size = shift_logits.size(-1)

    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    flat_losses = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
    token_losses = flat_losses.view(labels.size(0), -1)
    mask = shift_labels != -100

    # Sum log probabilities for non-masked assistant tokens
    return -(token_losses * mask).sum(dim=-1)


def load_dpo_model_and_tokenizer(
    model_name_or_path: str = "checkpoints/sft",
    base_model_name: str = "unsloth/Llama-3.2-1B",
    device: str = "mps",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load model starting from SFT checkpoint or base model with LoRA."""
    console.print(f"[bold cyan]Loading model for DPO from:[/bold cyan] [yellow]{model_name_or_path}[/yellow]")

    adapter_cfg = Path(model_name_or_path) / "adapter_config.json"
    if adapter_cfg.exists():
        console.print(f"[cyan]Initializing DPO policy from SFT adapter at {model_name_or_path}[/cyan]")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map=device,
        )
        # Load SFT adapter as initial policy
        model = PeftModel.from_pretrained(base_model, model_name_or_path, is_trainable=True)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        except Exception:
            model_name_or_path = base_model_name
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map=device,
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)

    ensure_chat_template(tokenizer)
    model.gradient_checkpointing_enable()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())

    table = Table(title="[bold green]DPO Policy Model Configuration[/bold green]")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="bold magenta")
    table.add_row("Base Architecture", base_model_name)
    table.add_row("Init Checkpoint", model_name_or_path)
    table.add_row("Precision", "torch.bfloat16")
    table.add_row("Device", device)
    table.add_row("Trainable Parameters", f"{trainable_params:,} ({100*trainable_params/all_params:.2f}%)")
    console.print(table)

    return model, tokenizer


def train_dpo(
    model_name_or_path: str = "checkpoints/sft",
    base_model_name: str = "unsloth/Llama-3.2-1B",
    dataset_path: str = "data/decontaminated/preference_with_ref_logprobs.jsonl",
    output_dir: str = "checkpoints/dpo",
    beta: float = 5.0,
    length_normalized: bool = True,
    learning_rate: float = 5e-6,
    epochs: int = 1,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    max_seq_len: int = 1024,
    max_steps: int | None = None,
    limit_samples: int | None = None,
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
) -> None:
    """Execute Length-Normalized DPO training loop with split forward passes."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    visual_logger = DPOVisualLogger(output_dir=output_path, run_name="micro_tulu_dpo")

    # Load Model and Tokenizer
    model, tokenizer = load_dpo_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        base_model_name=base_model_name,
        device=device,
    )

    # Load DPO Dataset
    dataset = DPODataset(
        jsonl_path=dataset_path,
        tokenizer=tokenizer,
        max_length=max_seq_len,
        limit=limit_samples,
    )
    collator = DPODataCollator(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    total_training_steps = (len(dataloader) // gradient_accumulation_steps) * epochs
    if max_steps is not None and max_steps < total_training_steps:
        total_training_steps = max_steps

    console.print(
        f"[bold green]DPO Dataset loaded:[/bold green] {len(dataset):,} pairs | "
        f"[bold cyan]Total optimization steps:[/bold cyan] {total_training_steps}"
    )

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=max(2, int(total_training_steps * 0.1)),
        num_training_steps=total_training_steps,
    )

    # DPO Training Loop with Split Forward Passes
    global_step = 0
    model.train()
    optimizer.zero_grad()

    accum_loss = 0.0
    accum_chosen_reward = 0.0
    accum_rejected_reward = 0.0
    accum_margin = 0.0
    accum_accuracy = 0.0
    steps_in_accum = 0

    console.print(
        f"\n[bold green]🚀 Commencing DPO Training ("
        f"{'Length-Normalized (TÜLU 3)' if length_normalized else 'Standard Unnormalized'}, "
        f"β={beta})...[/bold green]\n"
    )

    for epoch in range(epochs):
        for batch_idx, batch in enumerate(dataloader):
            # Move scalar tensors to device
            ref_c_lp = batch["ref_chosen_logprobs"].to(device)
            ref_r_lp = batch["ref_rejected_logprobs"].to(device)
            len_c = batch["chosen_lengths"].to(device)
            len_r = batch["rejected_lengths"].to(device)

            # Split Forward Pass 1: Chosen Completion
            c_input_ids = batch["chosen_input_ids"].to(device)
            c_mask = batch["chosen_attention_mask"].to(device)
            c_labels = batch["chosen_labels"].to(device)

            c_outputs = model(input_ids=c_input_ids, attention_mask=c_mask)
            policy_chosen_logprobs = compute_batch_logprobs(c_outputs.logits, c_labels)

            # Split Forward Pass 2: Rejected Completion
            r_input_ids = batch["rejected_input_ids"].to(device)
            r_mask = batch["rejected_attention_mask"].to(device)
            r_labels = batch["rejected_labels"].to(device)

            r_outputs = model(input_ids=r_input_ids, attention_mask=r_mask)
            policy_rejected_logprobs = compute_batch_logprobs(r_outputs.logits, r_labels)

            # Compute Length-Normalized DPO Loss
            loss, dpo_metrics = compute_dpo_loss(
                policy_chosen_logprobs=policy_chosen_logprobs,
                policy_rejected_logprobs=policy_rejected_logprobs,
                ref_chosen_logprobs=ref_c_lp,
                ref_rejected_logprobs=ref_r_lp,
                chosen_lengths=len_c,
                rejected_lengths=len_r,
                beta=beta,
                length_normalized=length_normalized,
            )

            steps_in_accum += 1
            # Scale and backward
            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()

            accum_loss += dpo_metrics.loss
            accum_chosen_reward += dpo_metrics.chosen_reward
            accum_rejected_reward += dpo_metrics.rejected_reward
            accum_margin += dpo_metrics.reward_margin
            accum_accuracy += dpo_metrics.accuracy

            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                curr_epoch = epoch + (batch_idx + 1) / len(dataloader)
                curr_lr = scheduler.get_last_lr()[0]

                actual_divisor = max(1, steps_in_accum)
                visual_logger.log_step(
                    step=global_step,
                    epoch=curr_epoch,
                    loss=accum_loss / actual_divisor,
                    chosen_reward=accum_chosen_reward / actual_divisor,
                    rejected_reward=accum_rejected_reward / actual_divisor,
                    reward_margin=accum_margin / actual_divisor,
                    accuracy=accum_accuracy / actual_divisor,
                    lr=curr_lr,
                )

                visual_logger.print_step_summary(global_step, total_training_steps)

                accum_loss = 0.0
                accum_chosen_reward = 0.0
                accum_rejected_reward = 0.0
                accum_margin = 0.0
                accum_accuracy = 0.0
                steps_in_accum = 0

                if max_steps is not None and global_step >= max_steps:
                    break

        if max_steps is not None and global_step >= max_steps:
            break

    # Save fine-tuned DPO policy model
    console.print(f"[bold green]Saving fine-tuned DPO adapter to[/bold green] [cyan]{output_path}[/cyan]...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    hyperparams = {
        "Base Model": base_model_name,
        "Algorithm": "Length-Normalized DPO (TÜLU 3)" if length_normalized else "Standard DPO",
        "Beta (β)": beta,
        "Learning Rate": learning_rate,
        "Epochs": epochs,
        "Micro Batch Size": batch_size,
        "Gradient Accumulation": gradient_accumulation_steps,
        "Effective Batch Size": batch_size * gradient_accumulation_steps,
        "Device": device,
    }
    visual_logger.generate_html_report(hyperparams)
    console.print("[bold green]✅ Phase 3: Length-Normalized DPO Training Complete![/bold green]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Micro-TÜLU Length-Normalized DPO Training")
    parser.add_argument("--model", type=str, default="checkpoints/sft", help="Reference/Initial policy checkpoint")
    parser.add_argument("--base-model", type=str, default="unsloth/Llama-3.2-1B", help="Base model architecture")
    parser.add_argument("--dataset", type=str, default="data/decontaminated/preference_with_ref_logprobs.jsonl", help="Enriched preference JSONL")
    parser.add_argument("--output-dir", type=str, default="checkpoints/dpo", help="Output directory")
    parser.add_argument("--beta", type=float, default=5.0, help="KL penalty coefficient beta")
    parser.add_argument("--no-length-norm", action="store_true", help="Disable TÜLU 3 length normalization")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Micro batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max optimization steps")
    parser.add_argument("--limit-samples", type=int, default=None, help="Limit on dataset pairs")
    args = parser.parse_args()

    train_dpo(
        model_name_or_path=args.model,
        base_model_name=args.base_model,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        beta=args.beta,
        length_normalized=not args.no_length_norm,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_seq_len=args.max_seq_len,
        max_steps=args.max_steps,
        limit_samples=args.limit_samples,
    )


if __name__ == "__main__":
    main()
