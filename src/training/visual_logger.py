"""Visual Logger and Interactive Dashboard for SFT Training.

Provides:
1. Rich terminal dashboard with live step loss, token throughput, and memory stats.
2. Sparkline loss trajectory in terminal.
3. Pre- vs. Post-training generation comparison cards.
4. Standalone HTML visual training report with embedded interactive SVG loss curve.
"""

from __future__ import annotations

import html
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@dataclass
class StepMetric:
    step: int
    epoch: float
    loss: float
    sum_loss: float
    mean_loss: float
    learning_rate: float
    active_tokens: int
    tokens_per_sec: float
    timestamp: float


class VisualTrainingLogger:
    """Tracks training progress, prints rich terminal metrics, and generates HTML reports."""

    def __init__(self, output_dir: Path | str, run_name: str = "sft_micro_tulu") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.metrics: list[StepMetric] = []
        self.start_time = time.time()
        self.step_start_time = time.time()
        self.generation_samples: list[dict[str, str]] = []

    def log_step(
        self,
        step: int,
        epoch: float,
        loss: float,
        sum_loss: float,
        mean_loss: float,
        lr: float,
        active_tokens: int,
    ) -> None:
        now = time.time()
        elapsed = max(1e-4, now - self.step_start_time)
        tokens_per_sec = active_tokens / elapsed
        self.step_start_time = now

        metric = StepMetric(
            step=step,
            epoch=epoch,
            loss=loss,
            sum_loss=sum_loss,
            mean_loss=mean_loss,
            learning_rate=lr,
            active_tokens=active_tokens,
            tokens_per_sec=tokens_per_sec,
            timestamp=now - self.start_time,
        )
        self.metrics.append(metric)

    def print_step_summary(self, step: int, total_steps: int) -> None:
        if not self.metrics:
            return
        m = self.metrics[-1]

        # Calculate sparkline string of recent losses
        recent_losses = [x.loss for x in self.metrics[-15:]]
        min_l, max_l = min(recent_losses), max(recent_losses)
        diff = max_l - min_l if max_l != min_l else 1.0
        sparks = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
        sparkline = "".join(sparks[min(7, int((val - min_l) / diff * 7))] for val in recent_losses)

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="cyan")
        table.add_column("Val", style="bold white")
        table.add_column("Key2", style="cyan")
        table.add_column("Val2", style="bold white")

        table.add_row(
            "Step", f"{step}/{total_steps} (Epoch {m.epoch:.2f})",
            "Loss (Selected)", f"[bold green]{m.loss:.4f}[/bold green]",
        )
        table.add_row(
            "Sum-Loss (TÜLU 3)", f"{m.sum_loss:.4f}",
            "Mean-Loss (Std)", f"{m.mean_loss:.4f}",
        )
        table.add_row(
            "Active Tokens", f"{m.active_tokens:,}",
            "Throughput", f"{m.tokens_per_sec:,.0f} tok/s",
        )
        table.add_row(
            "Learning Rate", f"{m.learning_rate:.2e}",
            "Loss Sparkline", f"[yellow]{sparkline}[/yellow]",
        )

        panel = Panel(table, title=f"[bold blue]Step {step}/{total_steps}[/bold blue]", border_style="blue")
        console.print(panel)

    def log_generation_sample(self, prompt: str, before: str, after: str) -> None:
        self.generation_samples.append({
            "prompt": prompt,
            "before": before,
            "after": after,
        })

    def print_generation_comparison(self, prompt: str, before: str, after: str) -> None:
        table = Table(title="[bold green]🤖 Qualitative Generation Check[/bold green]")
        table.add_column("Stage", style="cyan", no_wrap=True)
        table.add_column("Response", style="white")

        table.add_row("[bold cyan]Prompt[/bold cyan]", prompt)
        table.add_row("[bold red]Pre-trained Base[/bold red]", before)
        table.add_row("[bold green]TÜLU 3 SFT Fine-tuned[/bold green]", after)
        console.print(table)

    def save_metrics(self) -> Path:
        out_json = self.output_dir / "metrics.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump([asdict(m) for m in self.metrics], f, indent=2)
        return out_json

    def generate_html_report(self, hyperparams: dict[str, Any]) -> Path:
        report_file = self.output_dir / "training_report.html"
        self.save_metrics()

        # Build SVG loss trajectory curve
        steps = [m.step for m in self.metrics]
        losses = [m.loss for m in self.metrics]

        if losses:
            min_l = min(losses)
            max_l = max(losses)
            span_l = max_l - min_l if max_l != min_l else 1.0

            width = 700
            height = 240
            padding = 40

            points = []
            for idx, l in enumerate(losses):
                x = padding + (idx / max(1, len(losses) - 1)) * (width - 2 * padding)
                y = height - padding - ((l - min_l) / span_l) * (height - 2 * padding)
                points.append(f"{x:.1f},{y:.1f}")
            polyline_points = " ".join(points)
        else:
            polyline_points = ""
            min_l, max_l = 0.0, 0.0
            width, height = 700, 240

        # Build generation rows
        gen_rows = []
        for s in self.generation_samples:
            gen_rows.append(f"""
            <div class="gen-card">
                <div class="gen-prompt"><strong>Prompt:</strong> {html.escape(s['prompt'])}</div>
                <div class="gen-grid">
                    <div class="gen-box box-before">
                        <h5>Base Model (Pre-training)</h5>
                        <pre>{html.escape(s['before'])}</pre>
                    </div>
                    <div class="gen-box box-after">
                        <h5>Micro-TÜLU Model (SFT LoRA)</h5>
                        <pre>{html.escape(s['after'])}</pre>
                    </div>
                </div>
            </div>
            """)

        # Build hyperparams table
        hp_rows = "".join(
            f"<tr><td><code>{html.escape(k)}</code></td><td><strong>{html.escape(str(v))}</strong></td></tr>"
            for k, v in hyperparams.items()
        )

        initial_loss_str = f"{losses[0]:.4f}" if losses else "N/A"
        final_loss_str = f"{losses[-1]:.4f}" if losses else "N/A"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Micro-TÜLU SFT Training Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0b0f19; color: #f1f5f9; margin: 0; padding: 24px; }}
        h1 {{ color: #38bdf8; margin-top: 0; }}
        .kpi-container {{ display: flex; gap: 16px; margin-bottom: 24px; }}
        .kpi-card {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 16px 20px; flex: 1; }}
        .kpi-val {{ font-size: 26px; font-weight: bold; margin-top: 4px; }}
        .chart-card {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 20px; margin-bottom: 24px; }}
        svg text {{ fill: #94a3b8; font-size: 11px; }}
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th, td {{ padding: 8px 12px; border-bottom: 1px solid #334155; text-align: left; }}
        th {{ color: #94a3b8; font-weight: 600; }}
        code {{ background: #0f172a; padding: 2px 6px; border-radius: 4px; color: #38bdf8; }}
        .gen-card {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
        .gen-prompt {{ margin-bottom: 12px; color: #fbbf24; font-size: 14px; }}
        .gen-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
        .gen-box {{ background: #0f172a; border-radius: 6px; padding: 12px; border: 1px solid #334155; }}
        .box-before {{ border-left: 3px solid #f87171; }}
        .box-after {{ border-left: 3px solid #4ade80; }}
        .gen-box h5 {{ margin: 0 0 8px 0; color: #94a3b8; font-size: 12px; text-transform: uppercase; }}
        .gen-box pre {{ margin: 0; font-size: 13px; white-space: pre-wrap; word-break: break-word; color: #e2e8f0; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
    </style>
</head>
<body>
    <h1>🚀 Micro-TÜLU: SFT Training Report</h1>
    <p>Supervised Fine-Tuning on Apple Silicon MPS with PEFT LoRA and TÜLU 3 Equal Token-Weighting (Sum-Loss).</p>

    <div class="kpi-container">
        <div class="kpi-card"><div>Total Steps</div><div class="kpi-val" style="color:#38bdf8">{len(self.metrics)}</div></div>
        <div class="kpi-card"><div>Initial Loss</div><div class="kpi-val" style="color:#f87171">{initial_loss_str}</div></div>
        <div class="kpi-card"><div>Final Loss</div><div class="kpi-val" style="color:#4ade80">{final_loss_str}</div></div>
        <div class="kpi-card"><div>Total Trained Tokens</div><div class="kpi-val" style="color:#fbbf24">{sum(m.active_tokens for m in self.metrics):,}</div></div>
    </div>

    <div class="chart-card">
        <h3>📈 Training Loss Trajectory (Sum-Loss)</h3>
        <svg width="100%" height="{height}" viewBox="0 0 {width} {height}">
            <!-- Axes -->
            <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height-padding}" stroke="#475569" stroke-width="1"/>
            <line x1="{padding}" y1="{height-padding}" x2="{width-padding}" y2="{height-padding}" stroke="#475569" stroke-width="1"/>
            <text x="{padding}" y="{padding-10}">Loss: {max_l:.3f}</text>
            <text x="{padding}" y="{height-padding+15}">Loss: {min_l:.3f}</text>
            <text x="{width-padding-50}" y="{height-padding+15}">Step {steps[-1] if steps else 0}</text>

            <!-- Curve -->
            <polyline fill="none" stroke="#38bdf8" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" points="{polyline_points}"/>
        </svg>
    </div>

    <div class="grid-2">
        <div>
            <h3>⚙️ Training Hyperparameters</h3>
            <div class="kpi-card" style="padding:0;">
                <table>
                    <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
                    <tbody>{hp_rows}</tbody>
                </table>
            </div>
        </div>
        <div>
            <h3>🔬 Qualitative Inference Comparisons</h3>
            {"".join(gen_rows) if gen_rows else "<p>No generation checks recorded.</p>"}
        </div>
    </div>
</body>
</html>
"""
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        console.print(f"[bold green]Saved visual training report to[/bold green] [cyan]{report_file}[/cyan]")
        return report_file
