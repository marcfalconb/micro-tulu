"""Visual terminal inspector for Micro-TÜLU datasets.

Inspect dataset composition, token length distribution, sample conversations,
and side-by-side contaminated match comparisons with rich formatting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


def load_jsonl(path: Path | str, limit: int | None = None) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit and idx >= limit:
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def print_dataset_stats(records: list[dict[str, Any]], title: str) -> None:
    sources: dict[str, int] = {}
    lengths: list[int] = []

    for r in records:
        src = r.get("source", r.get("dataset", "unknown"))
        sources[src] = sources.get(src, 0) + 1

        # approximate length in words
        if "messages" in r and isinstance(r["messages"], list):
            text_len = sum(len(m.get("content", "").split()) for m in r["messages"])
        elif "prompt" in r:
            text_len = len(str(r["prompt"]).split())
        elif "question" in r:
            text_len = len(str(r["question"]).split())
        else:
            text_len = 0
        lengths.append(text_len)

    lengths.sort()
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    p50 = lengths[len(lengths) // 2] if lengths else 0
    p90 = lengths[int(len(lengths) * 0.9)] if lengths else 0

    table = Table(title=f"[bold green]📊 Dataset Overview: {title}[/bold green]")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="bold magenta")

    table.add_row("Total Samples", f"{len(records):,}")
    table.add_row("Avg Words / Sample", f"{avg_len:.1f}")
    table.add_row("Median Words (p50)", f"{p50}")
    table.add_row("90th Percentile Words (p90)", f"{p90}")
    table.add_row("Distinct Sources", f"{len(sources)}")
    console.print(table)

    if len(sources) > 1:
        src_table = Table(title="[bold blue]Sub-dataset Mixture Breakdown[/bold blue]")
        src_table.add_column("Source", style="cyan")
        src_table.add_column("Count", justify="right", style="magenta")
        src_table.add_column("Proportion", justify="right", style="yellow")

        for src, cnt in sorted(sources.items(), key=lambda x: x[1], reverse=True)[:10]:
            pct = cnt / len(records) * 100
            src_table.add_row(src, f"{cnt:,}", f"{pct:.1f}%")
        console.print(src_table)


def display_sample(record: dict[str, Any], index: int) -> None:
    sample_id = record.get("id", record.get("unique_id", f"sample_{index}"))
    source = record.get("source", record.get("dataset", "unknown"))

    console.print(f"\n[bold yellow]── Sample {index + 1} | ID: {sample_id} | Source: {source} ──[/bold yellow]")

    if "messages" in record and isinstance(record["messages"], list):
        for msg in record["messages"]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if role in ("user", "human"):
                style = "bold cyan"
                title = "👤 User"
            elif role in ("assistant", "gpt"):
                style = "bold green"
                title = "🤖 Assistant"
            else:
                style = "bold magenta"
                title = f"⚙️ {role.capitalize()}"

            panel = Panel(Text(content), title=title, border_style=style)
            console.print(panel)

    elif "prompt" in record and "chosen" in record and "rejected" in record:
        console.print(Panel(Text(record["prompt"]), title="👤 Prompt", border_style="cyan"))
        console.print(Panel(Text(str(record["chosen"])), title="✅ Chosen Response", border_style="green"))
        console.print(Panel(Text(str(record["rejected"])), title="❌ Rejected Response", border_style="red"))

    elif "problem" in record or "question" in record:
        q = record.get("problem", record.get("question", ""))
        ans = record.get("solution", record.get("answer", record.get("ground_truth", "")))
        console.print(Panel(Text(str(q)), title="❓ Problem / Prompt", border_style="cyan"))
        console.print(Panel(Text(str(ans)), title="🎯 Ground Truth", border_style="green"))

    else:
        console.print(Panel(json.dumps(record, indent=2, ensure_ascii=False), title="Raw Data"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Micro-TÜLU Visual Dataset Inspector")
    parser.add_argument("--file", type=str, required=True, help="Path to JSONL dataset to inspect")
    parser.add_argument("--samples", type=int, default=3, help="Number of samples to visualize")
    parser.add_argument("--start", type=int, default=0, help="Starting index of samples to visualize")
    args = parser.parse_args()

    records = load_jsonl(args.file)
    print_dataset_stats(records, Path(args.file).name)

    end_idx = min(args.start + args.samples, len(records))
    for i in range(args.start, end_idx):
        display_sample(records[i], i)


if __name__ == "__main__":
    main()
