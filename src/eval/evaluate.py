"""Standardized Evaluation Harness for Micro-TÜLU.

TÜLU 3 Reference (Section 2.2, Section 7 & Section 8):
Evaluates models across two distinct benchmark suites:
1. Development Suite:
   - GSM8K (Grade-school math reasoning, CoT extraction, SymPy equivalence)
   - MATH-500 (Competition mathematics, LaTeX \\boxed{} extraction)
   - IFEval (Deterministic instruction following and constraint verification)
2. Unseen (Held-out) Suite:
   - MMLU (Multi-task language understanding across STEM, humanities, and social sciences)

Generates:
- Rich terminal progression scoreboard comparing Base -> SFT -> DPO -> RLVR.
- Interactive visual HTML report (reports/evaluation_scoreboard.html).
- Structured JSON benchmark log (checkpoints/evaluation_results.json).
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from datasets import load_dataset
from peft import PeftModel
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from src.envs.verifiers import (
    DeterministicVerifier,
    check_math_equivalence,
    extract_math_answer,
    verify_ifeval_constraints,
)
from src.training.sft_dataset import ensure_chat_template

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
console = Console()


@dataclass
class BenchmarkItem:
    id: str
    suite: str  # "development" or "unseen"
    benchmark: str  # "gsm8k", "math500", "ifeval", "mmlu"
    prompt: str
    ground_truth: str
    extra_meta: dict[str, Any]


@dataclass
class EvaluationSample:
    benchmark: str
    prompt: str
    generated_text: str
    extracted_answer: str
    ground_truth: str
    is_correct: bool
    details: str


@dataclass
class StageResult:
    stage_name: str
    checkpoint_path: str
    gsm8k_accuracy: float
    math_accuracy: float
    ifeval_accuracy: float
    mmlu_accuracy: float
    average_score: float
    development_average: float
    unseen_average: float
    samples: list[EvaluationSample]


def extract_mmlu_choice(text: str) -> str:
    """Extract multiple-choice letter (A, B, C, or D) from completion."""
    m = re.search(r"(?:the\s+correct\s+answer\s+is|the\s+answer\s+is|answer\s*[:=])?\s*\(?([A-D])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Check first line or last line
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if lines:
        for ch in ("A", "B", "C", "D"):
            if lines[0].startswith(f"({ch})") or lines[0].startswith(f"{ch})") or lines[0] == ch:
                return ch
    return ""


def load_eval_suite(
    benchmarks_path: Path | str = "data/eval/benchmarks.jsonl",
    limit_per_bench: int = 20,
) -> list[BenchmarkItem]:
    """Load cached development benchmarks and fetch a slice of MMLU for the unseen suite."""
    items: list[BenchmarkItem] = []
    bench_file = Path(benchmarks_path)

    # 1. Load Development Suite (GSM8K, MATH-500, IFEval)
    bench_counts: dict[str, int] = {}
    if bench_file.exists():
        with open(bench_file, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                bench = rec.get("benchmark", "eval")
                if bench_counts.get(bench, 0) < limit_per_bench:
                    items.append(
                        BenchmarkItem(
                            id=rec["id"],
                            suite="development",
                            benchmark=bench,
                            prompt=rec["prompt"],
                            ground_truth=rec.get("reference", ""),
                            extra_meta=rec,
                        )
                    )
                    bench_counts[bench] = bench_counts.get(bench, 0) + 1

    # 2. Load Unseen Suite (MMLU)
    try:
        mmlu_ds = load_dataset("cais/mmlu", "all", split="test", streaming=True)
        choice_letters = ["A", "B", "C", "D"]
        count = 0
        for idx, row in enumerate(mmlu_ds):
            if count >= limit_per_bench:
                break
            # Format multiple choice prompt
            q_text = row["question"]
            choices = row["choices"]
            formatted_choices = "\n".join(f"{choice_letters[i]}) {c}" for i, c in enumerate(choices))
            prompt = (
                f"{q_text}\n\n{formatted_choices}\n\n"
                f"Answer with the single letter corresponding to the correct option: A, B, C, or D."
            )
            gt_letter = choice_letters[row["answer"]]
            items.append(
                BenchmarkItem(
                    id=f"mmlu_{idx}",
                    suite="unseen",
                    benchmark="mmlu",
                    prompt=prompt,
                    ground_truth=gt_letter,
                    extra_meta={"subject": row.get("subject", "")},
                )
            )
            count += 1
    except Exception as e:
        logger.warning("Could not fetch MMLU for unseen evaluation suite: %s", e)

    return items


def load_model_for_eval(
    checkpoint_or_repo: str,
    base_model_name: str = "unsloth/Llama-3.2-1B",
    device: str = "mps",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load model (base or LoRA adapter) in evaluation mode on Apple Silicon MPS."""
    adapter_cfg = Path(checkpoint_or_repo) / "adapter_config.json"
    if adapter_cfg.exists():
        console.print(f"[cyan]Loading LoRA adapter from {checkpoint_or_repo}...[/cyan]")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_or_repo)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map=device,
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_or_repo)
        model = model.merge_and_unload()
    else:
        console.print(f"[cyan]Loading base model from {checkpoint_or_repo}...[/cyan]")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_or_repo)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_or_repo,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map=device,
        )

    ensure_chat_template(tokenizer)
    model.eval()
    return model, tokenizer


def evaluate_stage(
    stage_name: str,
    checkpoint_path: str,
    benchmarks: list[BenchmarkItem],
    base_model_name: str = "unsloth/Llama-3.2-1B",
    max_new_tokens: int = 128,
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
) -> StageResult:
    """Run standardized greedy generation and assessment across all benchmarks."""
    console.print(f"\n[bold green]🔬 Evaluating Stage:[/bold green] [bold cyan]{stage_name}[/bold cyan] ({checkpoint_path})")

    model, tokenizer = load_model_for_eval(checkpoint_path, base_model_name=base_model_name, device=device)
    verifier = DeterministicVerifier(correct_reward=1.0)

    results_by_bench: dict[str, list[bool]] = {
        "gsm8k": [],
        "math500": [],
        "ifeval": [],
        "mmlu": [],
    }
    all_samples: list[EvaluationSample] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Evaluating {stage_name}", total=len(benchmarks))

        for item in benchmarks:
            # Format prompt with TÜLU 3 chat template and CoT instruction if math
            if item.benchmark == "gsm8k":
                formatted_prompt = f"Solve the following math problem step by step. Conclude your answer with '#### <number>'.\n\nQuestion: {item.prompt}"
            elif item.benchmark == "math500":
                formatted_prompt = f"Solve the following mathematical problem step by step. Put your final answer inside \\boxed{{}}.\n\nProblem: {item.prompt}"
            else:
                formatted_prompt = item.prompt

            messages = [{"role": "user", "content": formatted_prompt}]
            prompt_rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = tokenizer(prompt_rendered, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            gen_ids = outputs[0][inputs["input_ids"].shape[1] :]
            completion = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            # Benchmark-specific assessment
            is_correct = False
            extracted_ans = ""
            details = ""

            if item.benchmark in ("gsm8k", "math500"):
                extracted_ans = extract_math_answer(completion)
                is_correct = check_math_equivalence(extracted_ans, item.ground_truth)
                details = "SymPy math equivalence" if is_correct else "Math mismatch"

            elif item.benchmark == "ifeval":
                is_correct, details = verify_ifeval_constraints(completion, item.extra_meta)
                extracted_ans = "[CONSTRAINT_CHECK]"

            elif item.benchmark == "mmlu":
                extracted_ans = extract_mmlu_choice(completion)
                is_correct = (extracted_ans == item.ground_truth.strip().upper())
                details = f"Choice: {extracted_ans} vs {item.ground_truth}"

            if item.benchmark in results_by_bench:
                results_by_bench[item.benchmark].append(is_correct)

            sample = EvaluationSample(
                benchmark=item.benchmark,
                prompt=item.prompt[:150],
                generated_text=completion[:250],
                extracted_answer=extracted_ans,
                ground_truth=item.ground_truth[:50],
                is_correct=is_correct,
                details=details,
            )
            all_samples.append(sample)
            progress.advance(task)

    # Compute Accuracies
    def get_acc(key: str) -> float:
        vals = results_by_bench.get(key, [])
        return (sum(vals) / len(vals) * 100.0) if vals else 0.0

    gsm_acc = get_acc("gsm8k")
    math_acc = get_acc("math500")
    ifeval_acc = get_acc("ifeval")
    mmlu_acc = get_acc("mmlu")

    dev_avg = (gsm_acc + math_acc + ifeval_acc) / 3.0
    unseen_avg = mmlu_acc
    overall_avg = (gsm_acc + math_acc + ifeval_acc + mmlu_acc) / 4.0

    # Explicit memory cleanup on MPS
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return StageResult(
        stage_name=stage_name,
        checkpoint_path=checkpoint_path,
        gsm8k_accuracy=gsm_acc,
        math_accuracy=math_acc,
        ifeval_accuracy=ifeval_acc,
        mmlu_accuracy=mmlu_acc,
        average_score=overall_avg,
        development_average=dev_avg,
        unseen_average=unseen_avg,
        samples=all_samples,
    )


def generate_scoreboard_html(
    stages: list[StageResult],
    output_html_path: Path | str = "reports/evaluation_scoreboard.html",
) -> Path:
    """Generate interactive visual scoreboard dashboard with comparative SVG bar charts."""
    out_file = Path(output_html_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Table rows
    table_rows = []
    for s in stages:
        table_rows.append(f"""
        <tr>
            <td><strong>{html.escape(s.stage_name)}</strong><br><small style="color:#94a3b8"><code>{html.escape(s.checkpoint_path)}</code></small></td>
            <td style="color:#38bdf8; font-weight:bold;">{s.average_score:.1f}%</td>
            <td>{s.development_average:.1f}%</td>
            <td>{s.unseen_average:.1f}%</td>
            <td>{s.gsm8k_accuracy:.1f}%</td>
            <td>{s.math_accuracy:.1f}%</td>
            <td>{s.ifeval_accuracy:.1f}%</td>
            <td>{s.mmlu_accuracy:.1f}%</td>
        </tr>
        """)

    # SVG Multi-Bar Chart
    benchmarks = ["GSM8K", "MATH-500", "IFEval", "MMLU (Unseen)", "Average"]
    colors = ["#94a3b8", "#38bdf8", "#818cf8", "#4ade80"]  # Base, SFT, DPO, RLVR

    width = 750
    height = 280
    margin_left = 60
    margin_bottom = 40
    chart_w = width - margin_left - 40
    chart_h = height - margin_bottom - 40

    group_w = chart_w / len(benchmarks)
    bar_w = group_w / (len(stages) + 1)

    svg_bars = []
    # Grid lines
    for pct in [0, 25, 50, 75, 100]:
        y = height - margin_bottom - (pct / 100.0) * chart_h
        svg_bars.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width-40}" y2="{y:.1f}" stroke="#334155" stroke-dasharray="3"/>')
        svg_bars.append(f'<text x="{margin_left-10}" y="{y+4:.1f}" text-anchor="end" fill="#94a3b8" font-size="10">{pct}%</text>')

    for b_idx, b_name in enumerate(benchmarks):
        x_group = margin_left + b_idx * group_w
        svg_bars.append(f'<text x="{x_group + group_w/2:.1f}" y="{height-margin_bottom+20}" text-anchor="middle" fill="#cbd5e1" font-size="11" font-weight="bold">{b_name}</text>')

        for s_idx, s in enumerate(stages):
            if b_idx == 0:
                val = s.gsm8k_accuracy
            elif b_idx == 1:
                val = s.math_accuracy
            elif b_idx == 2:
                val = s.ifeval_accuracy
            elif b_idx == 3:
                val = s.mmlu_accuracy
            else:
                val = s.average_score

            h_bar = (val / 100.0) * chart_h
            x_bar = x_group + s_idx * bar_w + 10
            y_bar = height - margin_bottom - h_bar
            col = colors[s_idx % len(colors)]

            svg_bars.append(f'<rect x="{x_bar:.1f}" y="{y_bar:.1f}" width="{bar_w-4:.1f}" height="{h_bar:.1f}" fill="{col}" rx="3"/>')
            if val > 0:
                svg_bars.append(f'<text x="{x_bar + (bar_w-4)/2:.1f}" y="{y_bar-5:.1f}" text-anchor="middle" fill="{col}" font-size="10">{val:.0f}%</text>')

    # Legend
    legend_items = []
    for s_idx, s in enumerate(stages):
        col = colors[s_idx % len(colors)]
        legend_items.append(f'<span style="display:inline-flex; align-items:center; margin-right:16px; font-size:12px;"><span style="width:12px; height:12px; background:{col}; border-radius:3px; margin-right:6px;"></span>{html.escape(s.stage_name)}</span>')

    # Qualitative comparison cards (first 3 samples)
    sample_cards = []
    if stages:
        ref_samples = stages[0].samples[:4]
        for idx, r_samp in enumerate(ref_samples):
            stage_outputs = []
            for s in stages:
                if idx < len(s.samples):
                    samp = s.samples[idx]
                    status = "✅" if samp.is_correct else "❌"
                    stage_outputs.append(f"""
                    <div style="flex:1; background:#0f172a; border:1px solid #334155; border-radius:6px; padding:10px;">
                        <div style="font-size:11px; font-weight:bold; color:#94a3b8; margin-bottom:4px;">{html.escape(s.stage_name)} {status}</div>
                        <pre style="margin:0; font-size:11px; white-space:pre-wrap; font-family:monospace; color:#e2e8f0;">{html.escape(samp.generated_text)}</pre>
                        <div style="margin-top:6px; font-size:11px; color:#cbd5e1;">Extracted: <code>{html.escape(samp.extracted_answer)}</code></div>
                    </div>
                    """)

            sample_cards.append(f"""
            <div style="background:#1e293b; border:1px solid #334155; border-radius:8px; padding:16px; margin-bottom:16px;">
                <div style="font-size:13px; font-weight:bold; color:#fbbf24; margin-bottom:8px;">[{html.escape(r_samp.benchmark.upper())}] Prompt: {html.escape(r_samp.prompt)}</div>
                <div style="font-size:12px; color:#94a3b8; margin-bottom:12px;">Ground Truth: <code>{html.escape(r_samp.ground_truth)}</code></div>
                <div style="display:flex; gap:12px;">{"".join(stage_outputs)}</div>
            </div>
            """)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Micro-TÜLU: Standardized Evaluation Scoreboard</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0b0f19; color: #f1f5f9; margin: 0; padding: 24px; }}
        h1 {{ color: #38bdf8; margin-top: 0; }}
        .scoreboard-card {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 20px; margin-bottom: 24px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th, td {{ padding: 10px 14px; border-bottom: 1px solid #334155; text-align: left; }}
        th {{ color: #94a3b8; font-weight: 600; text-transform: uppercase; font-size: 11px; }}
        code {{ background: #0f172a; padding: 2px 6px; border-radius: 4px; color: #38bdf8; }}
    </style>
</head>
<body>
    <h1>🏆 Micro-TÜLU: Post-Training Progression Scoreboard</h1>
    <p>Tracking the full alignment trajectory: <code>Base Model &rarr; SFT &rarr; DPO &rarr; RLVR</code> across Development and Unseen Benchmark Suites.</p>

    <div class="scoreboard-card">
        <h3>📊 Comparative Benchmark Performance</h3>
        <div style="margin-bottom:16px;">{"".join(legend_items)}</div>
        <svg width="100%" height="{height}" viewBox="0 0 {width} {height}">
            {"".join(svg_bars)}
        </svg>
    </div>

    <div class="scoreboard-card" style="padding:0;">
        <table>
            <thead>
                <tr>
                    <th>Alignment Stage</th>
                    <th>Average Score</th>
                    <th>Development Avg</th>
                    <th>Unseen Avg</th>
                    <th>GSM8K (Math)</th>
                    <th>MATH-500</th>
                    <th>IFEval</th>
                    <th>MMLU (Held-out)</th>
                </tr>
            </thead>
            <tbody>{"".join(table_rows)}</tbody>
        </table>
    </div>

    <h3>🔍 Qualitative Cross-Stage Response Comparison</h3>
    {"".join(sample_cards)}
</body>
</html>
"""
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    console.print(f"[bold green]Saved interactive evaluation scoreboard to[/bold green] [cyan]{out_file}[/cyan]")
    return out_file


def run_full_evaluation_progression(
    limit_per_bench: int = 15,
    base_model_name: str = "unsloth/Llama-3.2-1B",
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
    output_html: str = "reports/evaluation_scoreboard.html",
    output_json: str = "checkpoints/evaluation_results.json",
) -> list[StageResult]:
    """Execute evaluation across Base, SFT, DPO, and RLVR checkpoints."""
    benchmarks = load_eval_suite(limit_per_bench=limit_per_bench)
    console.print(f"[bold green]Loaded {len(benchmarks)} benchmark evaluation items.[/bold green]")

    stages_to_eval = [
        ("Base Llama-3.2-1B", base_model_name),
    ]

    # Add available checkpoint adapters
    if Path("checkpoints/sft/adapter_config.json").exists():
        stages_to_eval.append(("Stage 1: SFT (Sum-Loss)", "checkpoints/sft"))
    if Path("checkpoints/dpo/adapter_config.json").exists():
        stages_to_eval.append(("Stage 2: DPO (Length-Norm)", "checkpoints/dpo"))
    if Path("checkpoints/rlvr/adapter_config.json").exists():
        stages_to_eval.append(("Stage 3: RLVR (Verifiable)", "checkpoints/rlvr"))
    if Path("checkpoints/olmorl/adapter_config.json").exists():
        stages_to_eval.append(("Stage 4: OlmoRL (Frontier)", "checkpoints/olmorl"))

    stage_results: list[StageResult] = []
    for stage_name, ckpt in stages_to_eval:
        res = evaluate_stage(
            stage_name=stage_name,
            checkpoint_path=ckpt,
            benchmarks=benchmarks,
            base_model_name=base_model_name,
            device=device,
        )
        stage_results.append(res)

    # Print Progression Scoreboard Table
    score_table = Table(title="[bold green]🏆 Micro-TÜLU Standardized Evaluation Scoreboard[/bold green]")
    score_table.add_column("Stage", style="cyan", no_wrap=True)
    score_table.add_column("Overall Avg", justify="right", style="bold magenta")
    score_table.add_column("Dev Avg", justify="right", style="blue")
    score_table.add_column("Unseen Avg", justify="right", style="green")
    score_table.add_column("GSM8K", justify="right", style="white")
    score_table.add_column("MATH", justify="right", style="white")
    score_table.add_column("IFEval", justify="right", style="white")
    score_table.add_column("MMLU", justify="right", style="white")

    for s in stage_results:
        score_table.add_row(
            s.stage_name,
            f"{s.average_score:.1f}%",
            f"{s.development_average:.1f}%",
            f"{s.unseen_average:.1f}%",
            f"{s.gsm8k_accuracy:.1f}%",
            f"{s.math_accuracy:.1f}%",
            f"{s.ifeval_accuracy:.1f}%",
            f"{s.mmlu_accuracy:.1f}%",
        )
    console.print(score_table)

    # Save JSON results
    out_json = Path(output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in stage_results], f, indent=2)
    console.print(f"[bold green]Saved JSON evaluation metrics to[/bold green] [cyan]{out_json}[/cyan]")

    # Generate HTML Scoreboard
    generate_scoreboard_html(stage_results, output_html_path=output_html)

    return stage_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Micro-TÜLU Standardized Evaluation Harness")
    parser.add_argument("--limit-per-bench", type=int, default=10, help="Number of benchmark items to evaluate per benchmark")
    parser.add_argument("--base-model", type=str, default="unsloth/Llama-3.2-1B", help="Base model architecture")
    parser.add_argument("--output-html", type=str, default="reports/evaluation_scoreboard.html", help="Path to output HTML scoreboard")
    parser.add_argument("--output-json", type=str, default="checkpoints/evaluation_results.json", help="Path to output JSON metrics")
    args = parser.parse_args()

    run_full_evaluation_progression(
        limit_per_bench=args.limit_per_bench,
        base_model_name=args.base_model,
        output_html=args.output_html,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
