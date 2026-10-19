"""Download and prepare evaluation benchmark prompts for 8-gram decontamination.

TÜLU 3 decontaminates training datasets against evaluation suites (GSM8K, MATH,
IFEval, HumanEval, MMLU, etc.) to prevent data leakage and guarantee that
evaluation gains reflect true generalization rather than memorization.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Generator

from datasets import load_dataset
from rich.console import Console
from rich.table import Table

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
console = Console()


@dataclass
class BenchmarkPrompt:
    id: str
    benchmark: str
    prompt: str
    reference: str = ""


def fetch_gsm8k_test(limit: int | None = None) -> Generator[BenchmarkPrompt, None, None]:
    """Fetch GSM8K test set prompts."""
    logger.info("Loading GSM8K test set...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    for idx, item in enumerate(ds):
        raw_ans = item.get("answer", "").strip()
        # Extract canonical ground truth answer after '####'
        clean_gt = raw_ans.split("####")[-1].strip() if "####" in raw_ans else raw_ans
        yield BenchmarkPrompt(
            id=f"gsm8k_test_{idx}",
            benchmark="gsm8k",
            prompt=item["question"].strip(),
            reference=clean_gt,
        )


def fetch_math500_test(limit: int | None = None) -> Generator[BenchmarkPrompt, None, None]:
    """Fetch MATH-500 test set prompts."""
    logger.info("Loading MATH-500 test set...")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    for idx, item in enumerate(ds):
        yield BenchmarkPrompt(
            id=f"math500_test_{idx}",
            benchmark="math500",
            prompt=item["problem"].strip(),
            reference=item.get("answer", item.get("solution", "")).strip(),
        )


def fetch_ifeval_test(limit: int | None = None) -> Generator[BenchmarkPrompt, None, None]:
    """Fetch IFEval benchmark prompts."""
    logger.info("Loading IFEval benchmark...")
    ds = load_dataset("google/IFEval", split="train")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    for idx, item in enumerate(ds):
        yield BenchmarkPrompt(
            id=f"ifeval_{item.get('key', idx)}",
            benchmark="ifeval",
            prompt=item["prompt"].strip(),
            reference="",
        )


def prepare_all_benchmarks(
    output_path: Path | str = "data/eval/benchmarks.jsonl",
    limit_per_benchmark: int | None = None,
) -> list[BenchmarkPrompt]:
    """Download, aggregate, and save all evaluation benchmarks."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    records: list[BenchmarkPrompt] = []
    counts: dict[str, int] = {}

    fetchers = [
        ("GSM8K (Grade School Math)", fetch_gsm8k_test),
        ("MATH-500 (Competition Math)", fetch_math500_test),
        ("IFEval (Instruction Following)", fetch_ifeval_test),
    ]

    for name, fetcher in fetchers:
        console.print(f"[cyan]Fetching benchmark:[/cyan] [bold]{name}[/bold]")
        items = list(fetcher(limit=limit_per_benchmark))
        records.extend(items)
        benchmark_key = items[0].benchmark if items else name
        counts[benchmark_key] = len(items)

    with open(output_file, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    # Visual Summary Table
    table = Table(title="[bold green]Evaluation Benchmarks Cached for Decontamination[/bold green]")
    table.add_column("Benchmark", style="cyan", no_wrap=True)
    table.add_column("Samples", justify="right", style="magenta")
    table.add_column("Description", style="white")

    table.add_row("gsm8k", str(counts.get("gsm8k", 0)), "Grade-school math multi-step reasoning")
    table.add_row("math500", str(counts.get("math500", 0)), "High-school/Olympiad competition math")
    table.add_row("ifeval", str(counts.get("ifeval", 0)), "Verifiable constraint instruction following")
    table.add_section()
    table.add_row("[bold]Total[/bold]", f"[bold]{len(records)}[/bold]", f"Saved to {output_file}")
    console.print(table)

    return records


if __name__ == "__main__":
    prepare_all_benchmarks()
