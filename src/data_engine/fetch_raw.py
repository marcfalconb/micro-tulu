"""Stream and cache slices of TÜLU 3 raw datasets for local post-training.

Datasets curated by Allen Institute for AI:
1. SFT Mixture: `allenai/tulu-3-sft-mixture` (2,500 samples)
2. Preference Mixture: `allenai/llama-3.1-tulu-3-8b-preference-mixture` (1,000 pairs)
3. RLVR Verifiable Mixture: `allenai/RLVR-GSM-MATH-IF-Mixed-Constraints` (500 prompts)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from datasets import load_dataset
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
console = Console()


def slice_and_save_dataset(
    dataset_name: str,
    output_path: Path | str,
    target_count: int,
    split: str = "train",
    filter_fn: Any = None,
) -> int:
    """Stream a dataset from Hugging Face and save a slice of target_count items to JSONL."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold cyan]Streaming {target_count} samples from[/bold cyan] [yellow]{dataset_name}[/yellow]...")
    ds = load_dataset(dataset_name, split=split, streaming=True)

    saved = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Slicing {output_file.name}", total=target_count)
        with open(output_file, "w", encoding="utf-8") as f:
            for item in ds:
                if filter_fn is not None and not filter_fn(item):
                    continue
                f.write(json.dumps(dict(item), ensure_ascii=False) + "\n")
                saved += 1
                progress.advance(task)
                if saved >= target_count:
                    break

    return saved


def fetch_all_raw_slices(
    sft_count: int = 2500,
    pref_count: int = 1000,
    rlvr_count: int = 500,
) -> dict[str, int]:
    """Fetch all three TÜLU 3 raw dataset slices."""
    results = {}

    # 1. SFT Mixture
    sft_saved = slice_and_save_dataset(
        dataset_name="allenai/tulu-3-sft-mixture",
        output_path="data/raw/sft_raw.jsonl",
        target_count=sft_count,
    )
    results["sft"] = sft_saved

    # 2. Preference Mixture
    pref_saved = slice_and_save_dataset(
        dataset_name="allenai/llama-3.1-tulu-3-8b-preference-mixture",
        output_path="data/raw/preference_raw.jsonl",
        target_count=pref_count,
    )
    results["preference"] = pref_saved

    # 3. RLVR Verifiable Mixture
    rlvr_saved = slice_and_save_dataset(
        dataset_name="allenai/RLVR-GSM-MATH-IF-Mixed-Constraints",
        output_path="data/raw/rlvr_raw.jsonl",
        target_count=rlvr_count,
    )
    results["rlvr"] = rlvr_saved

    # Visual Summary
    table = Table(title="[bold green]TÜLU 3 Raw Data Slices Cached Locally[/bold green]")
    table.add_column("Pipeline Stage", style="cyan")
    table.add_column("Source HF Repo", style="yellow")
    table.add_column("Samples Sliced", justify="right", style="magenta")
    table.add_column("Output File", style="white")

    table.add_row("SFT (Supervised Fine-Tuning)", "allenai/tulu-3-sft-mixture", str(results["sft"]), "data/raw/sft_raw.jsonl")
    table.add_row("DPO (Preference Alignment)", "allenai/llama-3.1-tulu-3-8b-preference-mixture", str(results["preference"]), "data/raw/preference_raw.jsonl")
    table.add_row("RLVR (Verifiable Reasoning)", "allenai/RLVR-GSM-MATH-IF-Mixed-Constraints", str(results["rlvr"]), "data/raw/rlvr_raw.jsonl")
    console.print(table)

    return results


if __name__ == "__main__":
    fetch_all_raw_slices()
