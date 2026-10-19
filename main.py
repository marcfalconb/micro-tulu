"""Micro-TÜLU: Unified CLI Entry Point.

An end-to-end local post-training pipeline inspired by the Allen Institute for AI's TÜLU 3.
Supports: Decontamination, SFT (Sum-Loss), DPO (Length-Norm), RLVR (SymPy Verifiers), and Evaluation.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Micro-TÜLU: Local Post-Training Pipeline (AI2 TÜLU 3 Implementation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  decontaminate  Run 8-gram prompt decontamination scanner against eval benchmarks
  sft            Execute Supervised Fine-Tuning with TÜLU 3 Sum-Loss on Apple Silicon MPS
  cache-logprobs Precompute and cache offline reference log-probabilities for DPO
  dpo            Execute Length-Normalized Direct Preference Optimization (DPO)
  rlvr           Execute Reinforcement Learning with Verifiable Rewards (SymPy + IFEval)
  evaluate       Run standardized evaluation across Development and Unseen suites
  inspect        Visually inspect local dataset samples in terminal
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Post-training stage to execute")

    # 1. Decontaminate
    sub_decontam = subparsers.add_parser("decontaminate", help="8-Gram Decontamination Scanner")
    sub_decontam.add_argument("--input", type=str, default="data/raw/sft_raw.jsonl")
    sub_decontam.add_argument("--eval-benchmarks", type=str, default="data/eval/benchmarks.jsonl")
    sub_decontam.add_argument("--output", type=str, default="data/decontaminated/sft_clean.jsonl")
    sub_decontam.add_argument("--threshold", type=float, default=0.50)

    # 2. SFT
    sub_sft = subparsers.add_parser("sft", help="Supervised Fine-Tuning (SFT)")
    sub_sft.add_argument("--model", type=str, default="unsloth/Llama-3.2-1B")
    sub_sft.add_argument("--dataset", type=str, default="data/decontaminated/sft_clean.jsonl")
    sub_sft.add_argument("--output-dir", type=str, default="checkpoints/sft")
    sub_sft.add_argument("--loss-type", type=str, choices=["sum", "mean"], default="sum")
    sub_sft.add_argument("--lr", type=float, default=5e-5)
    sub_sft.add_argument("--epochs", type=int, default=1)
    sub_sft.add_argument("--batch-size", type=int, default=2)
    sub_sft.add_argument("--grad-accum", type=int, default=8)
    sub_sft.add_argument("--max-steps", type=int, default=None)

    # 3. Cache Log-probs
    sub_cache = subparsers.add_parser("cache-logprobs", help="Precompute Reference Log-Probs for DPO")
    sub_cache.add_argument("--model", type=str, default="checkpoints/sft")
    sub_cache.add_argument("--input", type=str, default="data/decontaminated/preference_clean.jsonl")
    sub_cache.add_argument("--output", type=str, default="data/decontaminated/preference_with_ref_logprobs.jsonl")
    sub_cache.add_argument("--limit", type=int, default=None)

    # 4. DPO
    sub_dpo = subparsers.add_parser("dpo", help="Length-Normalized DPO")
    sub_dpo.add_argument("--model", type=str, default="checkpoints/sft")
    sub_dpo.add_argument("--dataset", type=str, default="data/decontaminated/preference_with_ref_logprobs.jsonl")
    sub_dpo.add_argument("--output-dir", type=str, default="checkpoints/dpo")
    sub_dpo.add_argument("--beta", type=float, default=5.0)
    sub_dpo.add_argument("--lr", type=float, default=5e-6)
    sub_dpo.add_argument("--max-steps", type=int, default=None)

    # 5. RLVR
    sub_rlvr = subparsers.add_parser("rlvr", help="Reinforcement Learning with Verifiable Rewards")
    sub_rlvr.add_argument("--model", type=str, default="checkpoints/dpo")
    sub_rlvr.add_argument("--dataset", type=str, default="data/decontaminated/rlvr_clean.jsonl")
    sub_rlvr.add_argument("--output-dir", type=str, default="checkpoints/rlvr")
    sub_rlvr.add_argument("--group-size", type=int, default=4)
    sub_rlvr.add_argument("--lr", type=float, default=3e-7)
    sub_rlvr.add_argument("--max-steps", type=int, default=20)

    # 6. Evaluate
    sub_eval = subparsers.add_parser("evaluate", help="Standardized Evaluation Harness")
    sub_eval.add_argument("--limit-per-bench", type=int, default=10)
    sub_eval.add_argument("--output-html", type=str, default="docs/evaluation_scoreboard.html")

    # 7. Inspect
    sub_insp = subparsers.add_parser("inspect", help="Visual Dataset Inspector")
    sub_insp.add_argument("--file", type=str, default="data/decontaminated/sft_clean.jsonl")
    sub_insp.add_argument("--samples", type=int, default=3)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    console.print(f"[bold green]Micro-TÜLU Pipeline Controller:[/bold green] executing [cyan]{args.command}[/cyan]...")

    if args.command == "decontaminate":
        from src.data_engine.decontaminate import NGramIndex, decontaminate_file
        idx = NGramIndex(n=8)
        idx.load_benchmarks(args.eval_benchmarks)
        decontaminate_file(args.input, args.output, idx, threshold=args.threshold)

    elif args.command == "sft":
        from src.training.train_sft import train_sft
        train_sft(
            model_name_or_path=args.model,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            loss_type_str=args.loss_type,
            learning_rate=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_steps=args.max_steps,
        )

    elif args.command == "cache-logprobs":
        from src.training.cache_logprobs import cache_reference_logprobs
        cache_reference_logprobs(
            model_name_or_path=args.model,
            input_path=args.input,
            output_path=args.output,
            limit=args.limit,
        )

    elif args.command == "dpo":
        from src.training.train_dpo import train_dpo
        train_dpo(
            model_name_or_path=args.model,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            beta=args.beta,
            learning_rate=args.lr,
            max_steps=args.max_steps,
        )

    elif args.command == "rlvr":
        from src.training.train_rlvr import train_rlvr
        train_rlvr(
            model_name_or_path=args.model,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            group_size=args.group_size,
            learning_rate=args.lr,
            max_steps=args.max_steps,
        )

    elif args.command == "evaluate":
        from src.eval.evaluate import run_full_evaluation_progression
        run_full_evaluation_progression(
            limit_per_bench=args.limit_per_bench,
            output_html=args.output_html,
        )

    elif args.command == "inspect":
        from src.data_engine.inspect_data import display_sample, load_jsonl, print_dataset_stats
        records = load_jsonl(args.file)
        print_dataset_stats(records, Path(args.file).name)
        for i in range(min(args.samples, len(records))):
            display_sample(records[i], i)


if __name__ == "__main__":
    main()
