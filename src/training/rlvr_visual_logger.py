"""Visual Logger and Interactive HTML Report for RLVR Training.

Provides:
1. Live terminal dashboard tracking verifiable rewards, verifier accuracy, and response lengths.
2. Unicode sparkline trajectory for verifiable reward progression.
3. Qualitative verification cards displaying generated reasoning chains and SymPy/IFEval verdicts.
4. Standalone HTML visual report with interactive SVG trajectory charts.
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
class RLVRStepMetric:
    step: int
    mean_reward: float
    accuracy: float
    loss: float
    kl_div: float
    avg_response_len: float
    learning_rate: float
    timestamp: float


class RLVRVisualLogger:
    """Tracks RLVR training, logs step metrics, and generates visual reports."""

    def __init__(self, output_dir: Path | str, run_name: str = "micro_tulu_rlvr") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.metrics: list[RLVRStepMetric] = []
        self.rollout_samples: list[dict[str, Any]] = []
        self.start_time = time.time()

    def log_step(
        self,
        step: int,
        mean_reward: float,
        accuracy: float,
        loss: float,
        kl_div: float,
        avg_response_len: float,
        lr: float,
    ) -> None:
        metric = RLVRStepMetric(
            step=step,
            mean_reward=mean_reward,
            accuracy=accuracy,
            loss=loss,
            kl_div=kl_div,
            avg_response_len=avg_response_len,
            learning_rate=lr,
            timestamp=time.time() - self.start_time,
        )
        self.metrics.append(metric)

    def print_step_summary(self, step: int, total_steps: int) -> None:
        if not self.metrics:
            return
        m = self.metrics[-1]

        # Sparkline of recent verifiable rewards
        recent_rewards = [x.mean_reward for x in self.metrics[-15:]]
        min_r, max_r = min(recent_rewards), max(recent_rewards)
        diff = max_r - min_r if max_r != min_r else 1.0
        sparks = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
        sparkline = "".join(sparks[min(7, int((val - min_r) / diff * 7))] for val in recent_rewards)

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="cyan")
        table.add_column("Val", style="bold white")
        table.add_column("Key2", style="cyan")
        table.add_column("Val2", style="bold white")

        table.add_row(
            "Step", f"{step}/{total_steps}",
            "Verifiable Reward", f"[bold green]{m.mean_reward:+.3f}[/bold green]",
        )
        table.add_row(
            "Verifier Accuracy", f"[bold yellow]{m.accuracy*100:.1f}%[/bold yellow]",
            "RL Policy Loss", f"[bold red]{m.loss:.4f}[/bold red]",
        )
        table.add_row(
            "KL Divergence", f"{m.kl_div:.4f}",
            "Avg Response Len", f"{m.avg_response_len:.1f} tokens",
        )
        table.add_row(
            "Learning Rate", f"{m.learning_rate:.2e}",
            "Reward Sparkline", f"[cyan]{sparkline}[/cyan]",
        )

        panel = Panel(table, title=f"[bold blue]RLVR Step {step}/{total_steps}[/bold blue]", border_style="blue")
        console.print(panel)

    def log_rollout_sample(
        self,
        prompt: str,
        completion: str,
        extracted_answer: str,
        ground_truth: str,
        is_correct: bool,
        reward: float,
        details: str,
    ) -> None:
        self.rollout_samples.append({
            "prompt": prompt,
            "completion": completion,
            "extracted_answer": extracted_answer,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "reward": reward,
            "details": details,
        })

    def print_rollout_sample(self, s: dict[str, Any]) -> None:
        status_color = "green" if s["is_correct"] else "red"
        status_icon = "✅ Correct" if s["is_correct"] else "❌ Incorrect"

        panel = Panel(
            f"[bold cyan]Prompt:[/bold cyan] {s['prompt']}\n\n"
            f"[bold white]Generated Completion:[/bold white]\n{s['completion']}\n\n"
            f"[bold]Extracted Answer:[/bold] [yellow]{s['extracted_answer']}[/yellow]  |  "
            f"[bold]Ground Truth:[/bold] [magenta]{s['ground_truth']}[/magenta]\n"
            f"[bold]Verifier Verdict:[/bold] [{status_color}]{status_icon} (Reward: {s['reward']}) - {s['details']}[/{status_color}]",
            title=f"[{status_color}]Deterministic Verifier Rollout[/{status_color}]",
            border_style=status_color,
        )
        console.print(panel)

    def save_metrics(self) -> Path:
        out_json = self.output_dir / "rlvr_metrics.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump([asdict(m) for m in self.metrics], f, indent=2)
        return out_json

    def generate_html_report(self, hyperparams: dict[str, Any]) -> Path:
        report_file = self.output_dir / "training_report.html"
        self.save_metrics()

        rewards = [m.mean_reward for m in self.metrics]
        lengths = [m.avg_response_len for m in self.metrics]
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

        reward_svg = build_svg_curve(rewards, "#4ade80")
        length_svg = build_svg_curve(lengths, "#38bdf8")

        rollout_cards = []
        for s in self.rollout_samples:
            badge_class = "badge-pass" if s["is_correct"] else "badge-fail"
            badge_text = "PASSED" if s["is_correct"] else "FAILED"
            rollout_cards.append(f"""
            <div class="rollout-card">
                <div class="card-header">
                    <span class="badge {badge_class}">{badge_text}</span>
                    <span class="meta">Reward: <strong>{s['reward']}</strong> | Verifier: {html.escape(s['details'])}</span>
                </div>
                <div class="prompt-text"><strong>Prompt:</strong> {html.escape(s['prompt'])}</div>
                <pre class="comp-text">{html.escape(s['completion'])}</pre>
                <div class="ans-meta">
                    Extracted: <code>{html.escape(s['extracted_answer'])}</code> | Ground Truth: <code>{html.escape(s['ground_truth'])}</code>
                </div>
            </div>
            """)

        hp_rows = "".join(
            f"<tr><td><code>{html.escape(k)}</code></td><td><strong>{html.escape(str(v))}</strong></td></tr>"
            for k, v in hyperparams.items()
        )

        final_reward = f"{rewards[-1]:+.3f}" if rewards else "N/A"
        final_acc = f"{accuracies[-1]*100:.1f}%" if accuracies else "N/A"
        final_len = f"{lengths[-1]:.1f}" if lengths else "N/A"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Micro-TÜLU RLVR Training Report</title>
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
        .rollout-card {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
        .badge {{ padding: 4px 10px; border-radius: 9999px; font-size: 11px; font-weight: bold; }}
        .badge-pass {{ background: #22c55e; color: white; }}
        .badge-fail {{ background: #ef4444; color: white; }}
        .meta {{ font-size: 13px; color: #94a3b8; margin-left: 10px; }}
        .prompt-text {{ margin: 10px 0; font-size: 14px; color: #fbbf24; }}
        .comp-text {{ background: #0f172a; padding: 12px; border-radius: 6px; font-size: 12px; white-space: pre-wrap; word-break: break-word; color: #e2e8f0; border: 1px solid #334155; font-family: ui-monospace, monospace; }}
        .ans-meta {{ margin-top: 10px; font-size: 13px; color: #cbd5e1; }}
    </style>
</head>
<body>
    <h1>🔬 Micro-TÜLU: RLVR Training Report</h1>
    <p>Reinforcement Learning with Verifiable Rewards using SymPy symbolic math equivalence and deterministic IFEval rule verifiers on Apple Silicon MPS.</p>

    <div class="kpi-container">
        <div class="kpi-card"><div>Total Steps</div><div class="kpi-val" style="color:#38bdf8">{len(self.metrics)}</div></div>
        <div class="kpi-card"><div>Final Verifiable Reward</div><div class="kpi-val" style="color:#4ade80">{final_reward}</div></div>
        <div class="kpi-card"><div>Verifier Accuracy</div><div class="kpi-val" style="color:#fbbf24">{final_acc}</div></div>
        <div class="kpi-card"><div>Avg Generation Length</div><div class="kpi-val" style="color:#a855f7">{final_len} tok</div></div>
    </div>

    <div class="chart-grid">
        <div class="chart-card">
            <h3>📈 Verifiable Reward Trajectory</h3>
            {reward_svg}
        </div>
        <div class="chart-card">
            <h3>📏 Response Length Trajectory</h3>
            {length_svg}
        </div>
    </div>

    <div style="display: grid; grid-template-columns: 1fr 2fr; gap: 20px;">
        <div>
            <h3>⚙️ RLVR Hyperparameters</h3>
            <div class="kpi-card" style="padding:0;">
                <table>
                    <thead><tr><th>Hyperparameter</th><th>Value</th></tr></thead>
                    <tbody>{hp_rows}</tbody>
                </table>
            </div>
        </div>
        <div>
            <h3>🎯 Sample Rollouts & Verifier Diagnostics</h3>
            {"".join(rollout_cards) if rollout_cards else "<p>No rollouts logged.</p>"}
        </div>
    </div>
</body>
</html>
"""
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        console.print(f"[bold green]Saved visual RLVR report to[/bold green] [cyan]{report_file}[/cyan]")
        return report_file
