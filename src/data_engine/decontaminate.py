"""8-Gram Decontamination Scanner according to TÜLU 3 Methodology.

TÜLU 3 Reference (Section 3.2):
- Analyzes prompts (user turns) alone.
- Extracts 8-grams from evaluation benchmarks into an inverted index.
- Computes token coverage: if > 50% of test tokens share 8-grams with a training prompt,
  the training instance is marked as contaminated and decontaminated (removed).
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
console = Console()

TOKEN_REGEX = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize_text(text: str) -> list[str]:
    """Normalize and tokenize text into words and punctuation tokens."""
    return TOKEN_REGEX.findall(text.lower())


def extract_ngrams(tokens: Sequence[str], n: int = 8) -> list[tuple[str, ...]]:
    """Extract contiguous n-grams from a sequence of tokens."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


@dataclass
class ContaminationMatch:
    train_id: str
    eval_id: str
    benchmark: str
    overlap_ratio: float
    matched_tokens_count: int
    total_eval_tokens: int
    sample_matching_ngrams: list[str]
    train_prompt_preview: str
    eval_prompt_preview: str


class NGramIndex:
    """Inverted index mapping 8-grams to benchmark evaluation prompts and token spans."""

    def __init__(self, n: int = 8) -> None:
        self.n = n
        # Map: 8-gram tuple -> list of (eval_id, benchmark, start_pos)
        self.index: dict[tuple[str, ...], list[tuple[str, str, int]]] = defaultdict(list)
        # Store metadata for test instances: eval_id -> (benchmark, prompt_text, tokens, total_tokens)
        self.eval_instances: dict[str, dict[str, Any]] = {}

    def add_instance(self, eval_id: str, benchmark: str, prompt: str) -> None:
        tokens = tokenize_text(prompt)
        self.eval_instances[eval_id] = {
            "eval_id": eval_id,
            "benchmark": benchmark,
            "prompt": prompt,
            "tokens": tokens,
            "length": len(tokens),
        }
        for pos, ngram in enumerate(extract_ngrams(tokens, self.n)):
            self.index[ngram].append((eval_id, benchmark, pos))

    def load_benchmarks(self, benchmark_file: Path | str) -> int:
        """Load benchmark instances from a JSONL file."""
        file_path = Path(benchmark_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {file_path}")

        count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                self.add_instance(
                    eval_id=record["id"],
                    benchmark=record.get("benchmark", "eval"),
                    prompt=record["prompt"],
                )
                count += 1
        logger.info("Loaded %d benchmark prompts into %d-gram index (%d unique %d-grams).",
                    count, self.n, len(self.index), self.n)
        return count


class DecontaminationScanner:
    """Scans candidate training prompts against the 8-gram index."""

    def __init__(self, index: NGramIndex, threshold: float = 0.50) -> None:
        self.index = index
        self.threshold = threshold
        self.n = index.n

    @staticmethod
    def extract_prompt_from_record(record: dict[str, Any]) -> tuple[str, str]:
        """Extract prompt text and ID from SFT, DPO, or RLVR records."""
        rec_id = str(record.get("id", record.get("unique_id", "unknown")))

        # SFT format (messages list)
        if "messages" in record and isinstance(record["messages"], list):
            user_parts = [
                m["content"] for m in record["messages"] if m.get("role") in ("user", "human")
            ]
            return rec_id, "\n\n".join(user_parts)

        # DPO / RLVR format (prompt string or user prompt)
        if "prompt" in record and isinstance(record["prompt"], str):
            return rec_id, record["prompt"]

        if "question" in record and isinstance(record["question"], str):
            return rec_id, record["question"]

        if "problem" in record and isinstance(record["problem"], str):
            return rec_id, record["problem"]

        return rec_id, ""

    def scan_prompt(self, train_id: str, prompt_text: str) -> list[ContaminationMatch]:
        """Check if a training prompt contaminates any benchmark prompt (>50% token overlap)."""
        train_tokens = tokenize_text(prompt_text)
        train_ngrams = set(extract_ngrams(train_tokens, self.n))
        if not train_ngrams:
            return []

        # Find intersecting 8-grams and group by eval_id
        matched_eval_positions: dict[str, set[int]] = defaultdict(set)
        matched_eval_ngrams: dict[str, list[str]] = defaultdict(list)

        for ngram in train_ngrams:
            if ngram in self.index.index:
                for eval_id, _benchmark, pos in self.index.index[ngram]:
                    # Cover all tokens spanned by this 8-gram
                    for i in range(pos, pos + self.n):
                        matched_eval_positions[eval_id].add(i)
                    if len(matched_eval_ngrams[eval_id]) < 5:
                        matched_eval_ngrams[eval_id].append(" ".join(ngram))

        matches: list[ContaminationMatch] = []
        for eval_id, covered_positions in matched_eval_positions.items():
            eval_info = self.index.eval_instances[eval_id]
            total_eval_tokens = eval_info["length"]
            if total_eval_tokens == 0:
                continue

            overlap_ratio = len(covered_positions) / total_eval_tokens
            if overlap_ratio > self.threshold:
                matches.append(
                    ContaminationMatch(
                        train_id=train_id,
                        eval_id=eval_id,
                        benchmark=eval_info["benchmark"],
                        overlap_ratio=overlap_ratio,
                        matched_tokens_count=len(covered_positions),
                        total_eval_tokens=total_eval_tokens,
                        sample_matching_ngrams=matched_eval_ngrams[eval_id],
                        train_prompt_preview=prompt_text[:300],
                        eval_prompt_preview=eval_info["prompt"][:300],
                    )
                )

        return matches


def generate_html_report(
    report_path: Path | str,
    total_processed: int,
    clean_count: int,
    contaminated_count: int,
    matches: list[ContaminationMatch],
    dataset_name: str,
) -> None:
    """Generate a self-contained, interactive HTML decontamination report."""
    output_file = Path(report_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    rejection_pct = (contaminated_count / total_processed * 100) if total_processed else 0.0

    cards_html = []
    for m in matches:
        cards_html.append(f"""
        <div class="match-card">
            <div class="card-header">
                <span class="badge badge-danger">Overlap: {m.overlap_ratio*100:.1f}%</span>
                <span class="badge badge-bench">{html.escape(m.benchmark)}</span>
                <span class="meta">Train ID: <code>{html.escape(m.train_id)}</code> | Eval ID: <code>{html.escape(m.eval_id)}</code> ({m.matched_tokens_count}/{m.total_eval_tokens} tokens)</span>
            </div>
            <div class="comparison">
                <div class="prompt-box">
                    <h4>Training Prompt Preview</h4>
                    <pre>{html.escape(m.train_prompt_preview)}</pre>
                </div>
                <div class="prompt-box">
                    <h4>Benchmark Evaluation Prompt</h4>
                    <pre>{html.escape(m.eval_prompt_preview)}</pre>
                </div>
            </div>
            <div class="ngrams">
                <strong>Shared 8-grams:</strong> <code>{html.escape(", ".join(m.sample_matching_ngrams[:3]))}</code>
            </div>
        </div>
        """)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Micro-TÜLU Decontamination Report - {html.escape(dataset_name)}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
        h1 {{ margin-top: 0; color: #38bdf8; }}
        .kpi-container {{ display: flex; gap: 16px; margin-bottom: 24px; }}
        .kpi-card {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 16px 20px; flex: 1; }}
        .kpi-val {{ font-size: 28px; font-weight: bold; margin-top: 4px; }}
        .val-total {{ color: #38bdf8; }}
        .val-clean {{ color: #4ade80; }}
        .val-contam {{ color: #f87171; }}
        .val-pct {{ color: #fbbf24; }}
        .match-card {{ background: #1e293b; border: 1px solid #475569; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
        .card-header {{ display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }}
        .badge {{ padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 600; text-transform: uppercase; }}
        .badge-danger {{ background: #ef4444; color: #ffffff; }}
        .badge-bench {{ background: #6366f1; color: #ffffff; }}
        .meta {{ font-size: 13px; color: #94a3b8; }}
        .comparison {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
        .prompt-box {{ background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 12px; }}
        .prompt-box h4 {{ margin: 0 0 8px 0; font-size: 13px; color: #94a3b8; }}
        .prompt-box pre {{ margin: 0; font-size: 12px; white-space: pre-wrap; word-break: break-word; color: #e2e8f0; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
        .ngrams {{ margin-top: 12px; font-size: 13px; color: #cbd5e1; }}
        code {{ background: #334155; padding: 2px 6px; border-radius: 4px; font-size: 12px; }}
    </style>
</head>
<body>
    <h1>🔬 Micro-TÜLU Decontamination Report: {html.escape(dataset_name)}</h1>
    <p>Methodology: 8-gram Token Coverage Matching (&gt; 50% overlap against evaluation test sets).</p>

    <div class="kpi-container">
        <div class="kpi-card"><div>Total Examined</div><div class="kpi-val val-total">{total_processed:,}</div></div>
        <div class="kpi-card"><div>Clean Instances Kept</div><div class="kpi-val val-clean">{clean_count:,}</div></div>
        <div class="kpi-card"><div>Contaminated Filtered</div><div class="kpi-val val-contam">{contaminated_count:,}</div></div>
        <div class="kpi-card"><div>Rejection Rate</div><div class="kpi-val val-pct">{rejection_pct:.2f}%</div></div>
    </div>

    <h2>Contaminated Sample Matches ({len(matches)})</h2>
    {"".join(cards_html) if cards_html else "<p>✅ No benchmark contamination detected!</p>"}
</body>
</html>
"""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info("Saved interactive HTML visual report to %s", output_file)


def decontaminate_file(
    input_path: Path | str,
    output_path: Path | str,
    benchmark_index: NGramIndex,
    log_path: Path | str | None = None,
    html_report_path: Path | str | None = None,
    threshold: float = 0.50,
) -> tuple[int, int, list[ContaminationMatch]]:
    """Scan and decontaminate a JSONL dataset."""
    input_file = Path(input_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    scanner = DecontaminationScanner(benchmark_index, threshold=threshold)

    total_count = 0
    clean_count = 0
    all_matches: list[ContaminationMatch] = []
    contaminated_records: list[dict[str, Any]] = []

    with open(input_file, "r", encoding="utf-8") as fin:
        lines = fin.readlines()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Scanning {input_file.name}", total=len(lines))

        with open(output_file, "w", encoding="utf-8") as fout:
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                total_count += 1

                train_id, prompt = scanner.extract_prompt_from_record(record)
                matches = scanner.scan_prompt(train_id, prompt)

                if matches:
                    all_matches.extend(matches)
                    record["contamination_matches"] = [asdict(m) for m in matches]
                    contaminated_records.append(record)
                else:
                    fout.write(line + "\n")
                    clean_count += 1

                progress.advance(task)

    contaminated_count = total_count - clean_count

    # Save contaminated log if requested
    if log_path:
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w", encoding="utf-8") as f:
            for rec in contaminated_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Generate HTML report if requested
    if html_report_path:
        generate_html_report(
            report_path=html_report_path,
            total_processed=total_count,
            clean_count=clean_count,
            contaminated_count=contaminated_count,
            matches=all_matches,
            dataset_name=input_file.name,
        )

    # Print Visual Summary Table
    table = Table(title=f"[bold green]Decontamination Results: {input_file.name}[/bold green]")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="bold magenta")
    table.add_column("Note", style="white")

    table.add_row("Total Processed", str(total_count), "Raw samples evaluated")
    table.add_row("Clean Retained", str(clean_count), f"Saved to {output_file}")
    table.add_row(
        "Contaminated Removed",
        str(contaminated_count),
        f"Overlap > {threshold*100:.0f}% with evaluation sets",
    )
    rejection_pct = (contaminated_count / total_count * 100) if total_count else 0.0
    table.add_row("Rejection Rate", f"{rejection_pct:.2f}%", "benchmark leakage removed")
    console.print(table)

    # Preview sample contaminated matches in terminal if any
    if all_matches:
        sample_m = all_matches[0]
        panel = Panel(
            f"[bold red]Contaminated Sample Detected![/bold red]\n"
            f"[bold]Train ID:[/bold] {sample_m.train_id}  |  [bold]Benchmark:[/bold] {sample_m.benchmark} ({sample_m.eval_id})\n"
            f"[bold]Overlap Ratio:[/bold] [yellow]{sample_m.overlap_ratio*100:.1f}%[/yellow] ({sample_m.matched_tokens_count}/{sample_m.total_eval_tokens} tokens)\n\n"
            f"[bold cyan]Train Prompt Preview:[/bold cyan]\n{sample_m.train_prompt_preview}\n\n"
            f"[bold magenta]Eval Prompt Preview:[/bold magenta]\n{sample_m.eval_prompt_preview}\n\n"
            f"[bold green]Shared 8-Grams:[/bold green] {', '.join(sample_m.sample_matching_ngrams[:2])}",
            title="[red]Sample Decontamination Detail[/red]",
            border_style="red",
        )
        console.print(panel)

    return total_count, clean_count, all_matches


def main() -> None:
    parser = argparse.ArgumentParser(description="TÜLU 3 8-Gram Decontamination Scanner")
    parser.add_argument("--input", type=str, default="data/raw/sft_raw.jsonl", help="Input raw JSONL file")
    parser.add_argument("--eval-benchmarks", type=str, default="data/eval/benchmarks.jsonl", help="Cached eval benchmarks JSONL")
    parser.add_argument("--output", type=str, default="data/decontaminated/sft_clean.jsonl", help="Output decontaminated JSONL")
    parser.add_argument("--log-output", type=str, default="data/decontaminated/sft_contaminated.jsonl", help="Contaminated log JSONL")
    parser.add_argument("--html-report", type=str, default="data/decontaminated/sft_report.html", help="HTML visual report")
    parser.add_argument("--threshold", type=float, default=0.50, help="Token overlap threshold (default: 0.50)")
    parser.add_argument("--n", type=int, default=8, help="N-gram length (default: 8)")
    args = parser.parse_args()

    index = NGramIndex(n=args.n)
    index.load_benchmarks(args.eval_benchmarks)

    decontaminate_file(
        input_path=args.input,
        output_path=args.output,
        benchmark_index=index,
        log_path=args.log_output,
        html_report_path=args.html_report,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
