"""Reinforcement Learning with Verifiable Rewards (RLVR) Training Pipeline.

TÜLU 3 Reference (Section 6 & Section 6.2):
1. Replaces neural reward models with deterministic verification functions:
   - Symbolic math equivalence via SymPy
   - Deterministic IFEval constraint verification
2. Policy Optimization: Group-Relative Policy Gradient (GRPO / REINFORCE)
   sampling G rollouts per prompt with group-normalized advantages:
       A_i = (R_i - mean(R)) / (std(R) + eps)
       L = - 1/G sum_{i} (sum_{t} log pi_theta(y_{i,t} | x, y_{<t})) * A_i
3. Non-EOS Penalty: applies -10.0 penalty if generated response fails to emit EOS.
4. Rich terminal metrics, reward sparklines, and interactive HTML report.
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
from peft import LoraConfig, PeftModel, get_peft_model
from rich.console import Console
from rich.table import Table
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_cosine_schedule_with_warmup,
)

from src.envs.verifiers import DeterministicVerifier
from src.training.loss import (
    apply_reasoning_budget_penalty,
    compute_olmorl_advantages,
    filter_zero_gradient_groups,
)
from src.training.rlvr_visual_logger import RLVRVisualLogger
from src.training.sft_dataset import ensure_chat_template

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
console = Console()


def load_rlvr_model_and_tokenizer(
    model_name_or_path: str = "checkpoints/dpo",
    base_model_name: str = "unsloth/Llama-3.2-1B",
    device: str = "mps",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load model starting from DPO/SFT checkpoint or base model with LoRA.

    Note: Following TÜLU 3 Section 6.2, dropout is set to 0.0 for RL stability.
    """
    console.print(f"[bold cyan]Loading model for RLVR from:[/bold cyan] [yellow]{model_name_or_path}[/yellow]")

    adapter_cfg = Path(model_name_or_path) / "adapter_config.json"
    if adapter_cfg.exists():
        console.print(f"[cyan]Initializing RLVR policy from adapter at {model_name_or_path}[/cyan]")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map=device,
        )
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

    table = Table(title="[bold green]RLVR Policy Configuration[/bold green]")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="bold magenta")
    table.add_row("Base Architecture", base_model_name)
    table.add_row("Init Checkpoint", model_name_or_path)
    table.add_row("Precision", "torch.bfloat16")
    table.add_row("Device", device)
    table.add_row("Dropout", "0.0 (TÜLU 3 Section 6.2 rule)")
    table.add_row("Trainable Parameters", f"{trainable_params:,} ({100*trainable_params/all_params:.2f}%)")
    console.print(table)

    return model, tokenizer


def extract_prompt_from_rlvr_record(
    record: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> str:
    """Extract and format prompt with TÜLU 3 chat template and generation prompt."""
    if "messages" in record and isinstance(record["messages"], list):
        return tokenizer.apply_chat_template(record["messages"], tokenize=False, add_generation_prompt=True)

    prompt_text = str(record.get("prompt", record.get("question", record.get("problem", ""))))
    messages = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def train_rlvr(
    model_name_or_path: str = "checkpoints/dpo",
    base_model_name: str = "unsloth/Llama-3.2-1B",
    dataset_path: str = "data/decontaminated/rlvr_clean.jsonl",
    output_dir: str = "checkpoints/rlvr",
    group_size: int = 4,
    learning_rate: float = 3e-7,
    max_steps: int = 20,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.9,
    non_eos_penalty: float = -10.0,
    eval_interval: int = 5,
    limit_samples: int | None = None,
    olmorl: bool = False,
    max_thinking_tokens: int | None = None,
    clip_eps_low: float = 0.20,
    clip_eps_high: float = 0.28,
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
) -> None:
    """Execute RLVR policy gradient training on Apple Silicon MPS."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    visual_logger = RLVRVisualLogger(output_dir=output_path, run_name="micro_tulu_rlvr")
    verifier = DeterministicVerifier(correct_reward=1.0, non_eos_penalty=non_eos_penalty)

    # Load Model and Tokenizer
    model, tokenizer = load_rlvr_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        base_model_name=base_model_name,
        device=device,
    )

    # Load RLVR Records
    with open(dataset_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if limit_samples is not None:
        records = records[:limit_samples]

    console.print(f"[bold green]RLVR prompts loaded:[/bold green] {len(records):,} prompts")

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
        num_warmup_steps=max(2, int(max_steps * 0.1)),
        num_training_steps=max_steps,
    )

    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

    console.print(
        f"\n[bold green]🚀 Commencing RLVR Policy Optimization "
        f"(Group Size G={group_size}, lr={learning_rate:.1e})...[/bold green]\n"
    )

    global_step = 0
    rec_idx = 0

    while global_step < max_steps:
        record = records[rec_idx % len(records)]
        rec_idx += 1

        prompt_rendered = extract_prompt_from_rlvr_record(record, tokenizer)
        input_enc = tokenizer(prompt_rendered, return_tensors="pt").to(device)
        prompt_len = input_enc["input_ids"].shape[1]

        # 1. Rollout Phase: Sample G completions from current policy
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **input_enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=group_size,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # 2. Verification Phase: Deterministic Rule Verifier
        rewards = []
        is_correct_list = []
        completions = []
        gen_lengths = []

        for g in range(group_size):
            full_seq = outputs[g]
            gen_ids = full_seq[prompt_len:]
            gen_lengths.append(len(gen_ids))

            completion_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            completions.append(completion_text)

            has_eos = (tokenizer.eos_token_id in gen_ids.tolist()) if tokenizer.eos_token_id else True
            v_res = verifier.verify(completion_text, record, has_eos=has_eos)

            rewards.append(v_res.reward)
            is_correct_list.append(v_res.is_correct)

            # Log sample on evaluation step
            if global_step % eval_interval == 0 and g == 0:
                visual_logger.log_rollout_sample(
                    prompt=prompt_rendered[:200],
                    completion=completion_text[:300],
                    extracted_answer=v_res.predicted_answer,
                    ground_truth=v_res.ground_truth,
                    is_correct=v_res.is_correct,
                    reward=v_res.reward,
                    details=v_res.details,
                )
                visual_logger.print_rollout_sample({
                    "prompt": prompt_rendered[:200],
                    "completion": completion_text[:300],
                    "extracted_answer": v_res.predicted_answer,
                    "ground_truth": v_res.ground_truth,
                    "is_correct": v_res.is_correct,
                    "reward": v_res.reward,
                    "details": v_res.details,
                })

        # Apply test-time reasoning budget penalty (Kimi K3 Section 3.2)
        if max_thinking_tokens is not None:
            rewards = apply_reasoning_budget_penalty(
                rewards=rewards,
                lengths=gen_lengths,
                max_thinking_tokens=max_thinking_tokens,
                penalty_reward=-1.0,
            )

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        mean_r = rewards_tensor.mean()
        std_r = rewards_tensor.std()

        if olmorl:
            # OlmoRL unnormalized centered advantage: A_i = r_i - mu
            advantages = compute_olmorl_advantages(
                rewards_tensor, group_size=group_size, unnormalized=True
            )
            # Zero-gradient filtering: skip group if all rewards identical
            valid_mask, num_filtered = filter_zero_gradient_groups(
                rewards_tensor, group_size=group_size
            )
            is_group_informative = (num_filtered == 0)
        else:
            # Group-Relative Advantages: A_i = (R_i - mean(R)) / (std(R) + eps)
            if std_r > 1e-6:
                advantages = (rewards_tensor - mean_r) / (std_r + 1e-8)
            else:
                advantages = rewards_tensor - mean_r
            is_group_informative = True

        # 3. Policy Update Phase: Compute Policy Gradient
        model.train()
        optimizer.zero_grad()

        total_loss = torch.tensor(0.0, device=device)

        if is_group_informative:
            total_active_tokens = sum(gen_lengths) if olmorl else group_size
            for g in range(group_size):
                adv = advantages[g]
                if adv.abs() < 1e-8:
                    continue

                full_seq = outputs[g].unsqueeze(0)
                seq_len = full_seq.shape[1]

                # Labels: -100 for prompt tokens, token IDs for generated tokens
                labels = torch.full((1, seq_len), -100, dtype=torch.long, device=device)
                labels[0, prompt_len:] = full_seq[0, prompt_len:]

                out = model(input_ids=full_seq)
                shift_logits = out.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                vocab_size = shift_logits.size(-1)
                token_losses = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
                active_mask = (shift_labels.view(-1) != -100)

                # Sum of log probabilities: sum_t log P(y_t | x, y_<t)
                sum_logprob = -token_losses[active_mask].sum()

                # Policy gradient loss for this rollout:
                if olmorl:
                    # OlmoRL Token-Level Loss Reduction
                    policy_loss = - (sum_logprob * adv) / max(1, total_active_tokens)
                else:
                    policy_loss = - (sum_logprob * adv) / group_size

                policy_loss.backward()
                total_loss += policy_loss.detach()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        global_step += 1
        curr_lr = scheduler.get_last_lr()[0]
        step_acc = sum(is_correct_list) / len(is_correct_list)
        avg_len = sum(gen_lengths) / len(gen_lengths)

        visual_logger.log_step(
            step=global_step,
            mean_reward=mean_r.item(),
            accuracy=step_acc,
            loss=total_loss.item(),
            kl_div=0.0,
            avg_response_len=avg_len,
            lr=curr_lr,
        )

        visual_logger.print_step_summary(global_step, max_steps)

    # Save fine-tuned RLVR policy model
    console.print(f"[bold green]Saving fine-tuned RLVR adapter to[/bold green] [cyan]{output_path}[/cyan]...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    hyperparams = {
        "Base Model": base_model_name,
        "Init Checkpoint": model_name_or_path,
        "Algorithm": "OlmoRL (Unnormalized Advantage + Token Loss)" if olmorl else "RLVR with Group-Relative Advantages (GRPO)",
        "OlmoRL Enabled": str(olmorl),
        "Max Thinking Tokens": str(max_thinking_tokens) if max_thinking_tokens else "Unconstrained",
        "Clip Epsilon Low": str(clip_eps_low),
        "Clip Epsilon High": str(clip_eps_high),
        "Reward Signal": "Deterministic Rules (SymPy Math + IFEval)",
        "Group Size (G)": group_size,
        "Learning Rate": learning_rate,
        "Total Steps": max_steps,
        "Non-EOS Penalty": non_eos_penalty,
        "Device": device,
    }
    visual_logger.generate_html_report(hyperparams)
    console.print("[bold green]✅ Phase 4: RLVR Training Complete![/bold green]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Micro-TÜLU RLVR Training")
    parser.add_argument("--model", type=str, default="checkpoints/dpo", help="Model checkpoint to initialize from")
    parser.add_argument("--base-model", type=str, default="unsloth/Llama-3.2-1B", help="Base model architecture")
    parser.add_argument("--dataset", type=str, default="data/decontaminated/rlvr_clean.jsonl", help="Decontaminated RLVR JSONL")
    parser.add_argument("--output-dir", type=str, default="checkpoints/rlvr", help="Output directory")
    parser.add_argument("--group-size", type=int, default=4, help="Number of rollouts per prompt (G)")
    parser.add_argument("--lr", type=float, default=3e-7, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=20, help="Total optimization steps")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max generated tokens per rollout")
    parser.add_argument("--eval-interval", type=int, default=5, help="Interval for rollout logging")
    parser.add_argument("--limit-samples", type=int, default=None, help="Optional limit on loaded prompts")
    parser.add_argument("--olmorl", action="store_true", help="Enable OlmoRL unnormalized advantage, zero-grad filtering, and token loss")
    parser.add_argument("--max-thinking-tokens", type=int, default=None, help="Max reasoning token budget before -1.0 penalty (Kimi K3)")
    parser.add_argument("--clip-eps-low", type=float, default=0.20, help="Lower clipping epsilon")
    parser.add_argument("--clip-eps-high", type=float, default=0.28, help="Upper clipping epsilon (clip-higher)")
    args = parser.parse_args()

    train_rlvr(
        model_name_or_path=args.model,
        base_model_name=args.base_model,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        group_size=args.group_size,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        eval_interval=args.eval_interval,
        limit_samples=args.limit_samples,
        olmorl=args.olmorl,
        max_thinking_tokens=args.max_thinking_tokens,
        clip_eps_low=args.clip_eps_low,
        clip_eps_high=args.clip_eps_high,
    )


if __name__ == "__main__":
    main()
