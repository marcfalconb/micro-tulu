"""Visual Logger and Interactive HTML Report for Length-Normalized DPO Training.

Provides:
1. Live terminal dashboard tracking DPO loss, reward margin, and accuracy.
2. Unicode sparkline trajectory for reward margin progression.
3. Standalone HTML visual report with dual SVG trajectory charts (Loss & Reward Margin).
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
class DPOStepMetric:
    step: int
    epoch: float
    loss: float
    chosen_reward: float
    rejected_reward: float
    reward_margin: float
    accuracy: float
    learning_rate: float
    timestamp: float


class DPOVisualLogger:
    """Tracks DPO training progress, logs metrics, and generates visual reports."""

    def __init__(self, output_dir: Path | str, run_name: str = "micro_tulu_dpo") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.metrics: list[DPOStepMetric] = []
        self.start_time = time.time()

    def log_step(
        self,
        step: int,
        epoch: float,
        loss: float,
        chosen_reward: float,
        rejected_reward: float,
        reward_margin: float,
        accuracy: float,
        lr: float,
    ) -> None:
        metric = DPOStepMetric(
            step=step,
            epoch=epoch,
            loss=loss,
            chosen_reward=chosen_reward,
            rejected_reward=rejected_reward,
            reward_margin=reward_margin,
            accuracy=accuracy,
            learning_rate=lr,
            timestamp=time.time() - self.start_time,
        )
        self.metrics.append(metric)

    def print_step_summary(self, step: int, total_steps: int) -> None:
        if not self.metrics:
            return
        m = self.metrics[-1]

        # Calculate sparkline of recent reward margins
        recent_margins = [x.reward_margin for x in self.metrics[-15:]]
        min_m, max_m = min(recent_margins), max(recent_margins)
        diff = max_m - min_m if max_m != min_m else 1.0
        sparks = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
        sparkline = "".join(sparks[min(7, int((val - min_m) / diff * 7))] for val in recent_margins)

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="cyan")
        table.add_column("Val", style="bold white")
        table.add_column("Key2", style="cyan")
        table.add_column("Val2", style="bold white")

        table.add_row(
            "Step", f"{step}/{total_steps} (Epoch {m.epoch:.2f})",
            "DPO Loss", f"[bold red]{m.loss:.4f}[/bold red]",
        )
        table.add_row(
            "Reward Margin", f"[bold green]{m.reward_margin:+.4f}[/bold green]",
            "Pref Accuracy", f"[bold yellow]{m.accuracy*100:.1f}%[/bold yellow]",
        )
        table.add_row(
            "Chosen Reward", f"{m.chosen_reward:+.4f}",
            "Rejected Reward", f"{m.rejected_reward:+.4f}",
        )
        table.add_row(
            "Learning Rate", f"{m.learning_rate:.2e}",
            "Margin Sparkline", f"[cyan]{sparkline}[/cyan]",
        )

        panel = Panel(table, title=f"[bold blue]DPO Step {step}/{total_steps}[/bold blue]", border_style="blue")
        console.print(panel)

    def save_metrics(self) -> Path:
        out_json = self.output_dir / "dpo_metrics.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump([asdict(m) for m in self.metrics], f, indent=2)
        return out_json

    def generate_html_report(self, hyperparams: dict[str, Any]) -> Path:
        report_file = self.output_dir / "training_report.html"
        self.save_metrics()

        steps = [m.step for m in self.metrics]
        losses = [m.loss for m in self.metrics]
        margins = [m.reward_margin for m in self.metrics]
        accuracies = [m.accuracy for m in self.metrics]

        def build_svg_curve(values: list[float], stroke_color: str, width: int = 700, height: int = 200, padding: int = 35) -> str:
            if not values:
                return ""
            min_v, max_v = min(values), max(values)
            span_v = max_v - min_v if max_v != min_v else 1.0
            points = []
            for idx, val in enumerate(values):
                x = padding + (idx / max(1, len(values) - 1)) * (width - 2 * padding)
                y = height - padding - ((val - min_v) / span_v) * (height - 2 * padding)
                points.append(f"{x:.1f},{y:.1f}")
            pts_str = " ".join(points)
            return f"""
            <svg width="100%" height="{height}" viewBox="0 0 {width} {height}">
                <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height-padding}" stroke="#475569" stroke-width="1"/>
                <line x1="{padding}" y1="{height-padding}" x2="{width-padding}" y2="{height-padding}" stroke="#475569" stroke-width="1"/>
                <text x="{padding}" y="{padding-8}" fill="#94a3b8" font-size="11">{max_v:.3f}</text>
                <text x="{padding}" y="{height-padding+15}" fill="#94a3b8" font-size="11">{min_v:.3f}</text>
                <polyline fill="none" stroke="{stroke_color}" stroke-width="2.5" stroke-linecap="round" points="{pts_str}"/>
            </svg>
            """

        loss_svg = build_svg_curve(losses, "#f87171")
        margin_svg = build_svg_curve(margins, "#4ade80")

        hp_rows = "".join(
            f"<tr><td><code>{html.escape(k)}</code></td><td><strong>{html.escape(str(v))}</strong></td></tr>"
            for k, v in hyperparams.items()
        )

        final_loss = f"{losses[-1]:.4f}" if losses else "N/A"
        final_margin = f"{margins[-1]:+.4f}" if margins else "N/A"
        final_acc = f"{accuracies[-1]*100:.1f}%" if accuracies else "N/A"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Micro-TÜLU DPO Training Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0b0f19; color: #f1f5f9; margin: 0; padding: 24px; }}
        h1 {{ color: #38bdf8; margin-top: 0; }}
        .kpi-container {{ display: flex; gap: 16px; margin-bottom: 24px; }}
        .kpi-card {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 16px 20px; flex: 1; }}
        .kpi-val {{ font-size: 26px; font-weight: bold; margin-top: 4px; }}
        .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }}
        .chart-card {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 20px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th, td {{ padding: 8px 12px; border-bottom: 1px solid #334155; text-align: left; }}
        th {{ color: #94a3b8; font-weight: 600; }}
        code {{ background: #0f172a; padding: 2px 6px; border-radius: 4px; color: #38bdf8; }}
    </style>
</head>
<body>
    <h1>🎯 Micro-TÜLU: Length-Normalized DPO Report</h1>
    <p>Offline Direct Preference Optimization on Apple Silicon MPS with split forward passes and precomputed reference log-probabilities.</p>

    <div class="kpi-container">
        <div class="kpi-card"><div>Total Steps</div><div class="kpi-val" style="color:#38bdf8">{len(self.metrics)}</div></div>
        <div class="kpi-card"><div>Final DPO Loss</div><div class="kpi-val" style="color:#f87171">{final_loss}</div></div>
        <div class="kpi-card"><div>Reward Margin (R_c - R_r)</div><div class="kpi-val" style="color:#4ade80">{final_margin}</div></div>
        <div class="kpi-card"><div>Preference Accuracy</div><div class="kpi-val" style="color:#fbbf24">{final_acc}</div></div>
    </div>

    <div class="chart-grid">
        <div class="chart-card">
            <h3>📉 DPO Loss</h3>
            {loss_svg}
        </div>
        <div class="chart-card">
            <h3>📈 Implicit Reward Margin</h3>
            {margin_svg}
        </div>
    </div>

    <h3>⚙️ DPO Hyperparameters</h3>
    <div class="kpi-card" style="padding:0; max-width: 600px;">
        <table>
            <thead><tr><th>Hyperparameter</th><th>Value</th></tr></thead>
            <tbody>{hp_rows}</tbody>
        </table>
    </div>
</body>
</html>
"""
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        console.print(f"[bold green]Saved visual DPO report to[/bold green] [cyan]{report_file}[/cyan]")
        return report_file
