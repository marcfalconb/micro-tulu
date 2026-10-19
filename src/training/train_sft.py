"""Supervised Fine-Tuning (SFT) for Micro-TÜLU.

Implements:
1. PEFT LoRA instruction-tuning on Apple Silicon MPS with native BF16.
2. TÜLU 3 Equal Token-Weighting (Sum-Loss vs. Mean-Loss).
3. Exact TÜLU 3 Chat Template formatting and assistant-only token masking.
4. Rich terminal metrics, live sparklines, and qualitative generation probes.
5. Interactive HTML visual training report.
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
from peft import LoraConfig, get_peft_model
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

from src.training.loss import LossType, compute_token_weighted_loss
from src.training.sft_dataset import SFTDataCollator, SFTDataset, ensure_chat_template
from src.training.visual_logger import VisualTrainingLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
console = Console()

PROBE_PROMPTS = [
    "Janet has 16 eggs. She bakes 3 cakes using 4 eggs each. How many eggs remain?",
    "Write a short, engaging description of quantum entanglement for a 10-year-old in exactly three sentences.",
]


def load_model_and_tokenizer(
    model_name_or_path: str,
    device: str = "mps",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load base model and tokenizer with fallback and attach LoRA adapters."""
    console.print(f"[bold cyan]Loading tokenizer and model:[/bold cyan] [yellow]{model_name_or_path}[/yellow]...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    except Exception as e:
        if "403" in str(e) or "gated" in str(e).lower():
            fallback_repo = "unsloth/Llama-3.2-1B"
            console.print(
                f"[yellow]Warning: Access to {model_name_or_path} restricted (gated).[/yellow] "
                f"[bold green]Falling back to identical ungated weights: {fallback_repo}[/bold green]"
            )
            model_name_or_path = fallback_repo
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        else:
            raise e

    ensure_chat_template(tokenizer)

    console.print(f"[cyan]Loading base model weights in BF16 onto {device}...[/cyan]")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=device,
    )

    # Enable gradient checkpointing for unified memory conservation
    model.gradient_checkpointing_enable()

    # Configure PEFT LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, lora_config)

    # Print LoRA parameters summary table
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in peft_model.parameters())

    param_table = Table(title="[bold green]LoRA Architecture Overview[/bold green]")
    param_table.add_column("Property", style="cyan")
    param_table.add_column("Value", style="bold magenta")
    param_table.add_row("Base Model", model_name_or_path)
    param_table.add_row("Precision", "torch.bfloat16 (Native Apple Silicon)")
    param_table.add_row("Target Device", device)
    param_table.add_row("LoRA Rank (r)", str(lora_r))
    param_table.add_row("LoRA Alpha", str(lora_alpha))
    param_table.add_row("Trainable Parameters", f"{trainable_params:,}")
    param_table.add_row("Total Parameters", f"{all_params:,}")
    param_table.add_row("Trainable %", f"{100 * trainable_params / all_params:.2f}%")
    console.print(param_table)

    return peft_model, tokenizer


def generate_probe_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int = 128,
    device: str = "mps",
) -> str:
    """Generate a greedy response to a probe prompt."""
    model.eval()
    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated response tokens
    gen_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


def train_sft(
    model_name_or_path: str = "meta-llama/Llama-3.2-1B",
    dataset_path: str = "data/decontaminated/sft_clean.jsonl",
    output_dir: str = "checkpoints/sft",
    loss_type_str: str = "sum",
    learning_rate: float = 5e-5,
    epochs: int = 1,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    max_seq_len: int = 1024,
    lora_r: int = 16,
    lora_alpha: int = 32,
    max_steps: int | None = None,
    eval_interval: int = 25,
    sample_limit: int | None = None,
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
) -> None:
    """Execute SFT training loop with visual logging and checkpointing."""
    loss_type = LossType(loss_type_str)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    visual_logger = VisualTrainingLogger(output_dir=output_path, run_name="micro_tulu_sft")

    # Load Model & Tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        device=device,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    )

    # Load SFT Dataset
    console.print(f"[cyan]Loading dataset:[/cyan] [yellow]{dataset_path}[/yellow]...")
    dataset = SFTDataset(
        jsonl_path=dataset_path,
        tokenizer=tokenizer,
        max_length=max_seq_len,
        limit=sample_limit,
    )
    collator = SFTDataCollator(tokenizer=tokenizer)
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
        f"[bold green]Dataset loaded:[/bold green] {len(dataset):,} samples | "
        f"[bold cyan]Total optimization steps:[/bold cyan] {total_training_steps}"
    )

    # Record Baseline Pre-training Responses
    console.print("[cyan]Evaluating baseline pre-training responses on probe prompts...[/cyan]")
    baseline_responses = {}
    for prompt in PROBE_PROMPTS:
        resp = generate_probe_response(model, tokenizer, prompt, device=device)
        baseline_responses[prompt] = resp

    # Optimizer & LR Scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )
    warmup_steps = max(2, int(total_training_steps * 0.05))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    # Training Loop
    global_step = 0
    model.train()
    optimizer.zero_grad()

    accumulated_active_tokens = 0
    accumulated_loss = 0.0
    accumulated_sum_loss = 0.0
    accumulated_mean_loss = 0.0
    steps_in_accum = 0

    console.print(f"\n[bold green]🚀 Commencing SFT Training ({loss_type.value.upper()} Loss Mode)...[/bold green]\n")

    for epoch in range(epochs):
        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            loss, metrics = compute_token_weighted_loss(
                logits=outputs.logits,
                labels=batch["labels"],
                loss_type=loss_type,
            )

            steps_in_accum += 1
            # Scale loss for gradient accumulation
            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()

            accumulated_loss += metrics["loss"]
            accumulated_sum_loss += metrics["sum_loss"]
            accumulated_mean_loss += metrics["mean_loss"]
            accumulated_active_tokens += int(metrics["active_tokens"])

            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                current_epoch = epoch + (batch_idx + 1) / len(dataloader)
                current_lr = scheduler.get_last_lr()[0]

                # Step metric averages over actual accumulation window
                actual_divisor = max(1, steps_in_accum)
                avg_loss = accumulated_loss / actual_divisor
                avg_sum_loss = accumulated_sum_loss / actual_divisor
                avg_mean_loss = accumulated_mean_loss / actual_divisor

                visual_logger.log_step(
                    step=global_step,
                    epoch=current_epoch,
                    loss=avg_loss,
                    sum_loss=avg_sum_loss,
                    mean_loss=avg_mean_loss,
                    lr=current_lr,
                    active_tokens=accumulated_active_tokens,
                )

                visual_logger.print_step_summary(global_step, total_training_steps)

                accumulated_loss = 0.0
                accumulated_sum_loss = 0.0
                accumulated_mean_loss = 0.0
                accumulated_active_tokens = 0
                steps_in_accum = 0

                # Periodic Probe Evaluation
                if global_step % eval_interval == 0 or global_step == total_training_steps:
                    for prompt in PROBE_PROMPTS:
                        curr_resp = generate_probe_response(model, tokenizer, prompt, device=device)
                        visual_logger.log_generation_sample(
                            prompt=prompt,
                            before=baseline_responses[prompt],
                            after=curr_resp,
                        )
                        visual_logger.print_generation_comparison(
                            prompt=prompt,
                            before=baseline_responses[prompt],
                            after=curr_resp,
                        )
                    model.train()

                if max_steps is not None and global_step >= max_steps:
                    break

        if max_steps is not None and global_step >= max_steps:
            break

    # Save Fine-tuned PEFT Model and Tokenizer
    console.print(f"[bold green]Saving fine-tuned LoRA adapter to[/bold green] [cyan]{output_path}[/cyan]...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Save visual HTML report and metrics
    hyperparams = {
        "Base Model": model_name_or_path,
        "Loss Formulation": f"{loss_type.value.upper()} Loss (TÜLU 3 Token-Weighted)" if loss_type == LossType.SUM_LOSS else "Mean Loss",
        "Learning Rate": learning_rate,
        "Epochs": epochs,
        "Batch Size (Micro)": batch_size,
        "Gradient Accumulation": gradient_accumulation_steps,
        "Effective Batch Size": batch_size * gradient_accumulation_steps,
        "LoRA Rank (r)": lora_r,
        "LoRA Alpha": lora_alpha,
        "Max Sequence Length": max_seq_len,
        "Device": device,
    }
    visual_logger.generate_html_report(hyperparams)
    console.print("[bold green]✅ Phase 2: SFT Training Complete![/bold green]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Micro-TÜLU Supervised Fine-Tuning")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B", help="Model name or path")
    parser.add_argument("--dataset", type=str, default="data/decontaminated/sft_clean.jsonl", help="Decontaminated SFT JSONL")
    parser.add_argument("--output-dir", type=str, default="checkpoints/sft", help="Output directory")
    parser.add_argument("--loss-type", type=str, choices=["sum", "mean"], default="sum", help="Loss weighting formulation")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Micro batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max optimization steps")
    parser.add_argument("--eval-interval", type=int, default=20, help="Interval for qualitative generation checks")
    parser.add_argument("--limit-samples", type=int, default=None, help="Optional limit on loaded samples")
    args = parser.parse_args()

    train_sft(
        model_name_or_path=args.model,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        loss_type_str=args.loss_type,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_seq_len=args.max_seq_len,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        sample_limit=args.limit_samples,
    )


if __name__ == "__main__":
    main()
