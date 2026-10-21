"""OlmoRL & Reasoning Effort Visual Report Generator.

Visualizes:
1. Vanilla GRPO vs OlmoRL Unnormalized Advantage:
   Demonstrates how dividing by standard deviation (1/sigma) artificially inflates
   gradient magnitudes on rare-success outlier questions (difficulty bias), whereas
   OlmoRL unnormalized advantage scales proportionally to reward delta.
2. Asymmetric PPO Clip-Higher Dynamics (eps_low=0.20, eps_high=0.28).
3. Kimi K3 Reasoning Token Budget Penalties.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np


def generate_olmorl_comparison_report(output_dir: str | Path = "reports") -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "olmorl_comparison.html"

    # Synthetic simulation data for comparison
    np.random.seed(42)

    # Simulation 1: Rare-success question (1 success out of 8 completions)
    rewards_rare = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    mu_rare = rewards_rare.mean()
    sigma_rare = rewards_rare.std()
    adv_grpo_rare = (rewards_rare - mu_rare) / (sigma_rare + 1e-8)
    adv_olmo_rare = rewards_rare - mu_rare

    # Simulation 2: Balanced question (4 successes, 4 failures)
    rewards_bal = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    mu_bal = rewards_bal.mean()
    sigma_bal = rewards_bal.std()
    adv_grpo_bal = (rewards_bal - mu_bal) / (sigma_bal + 1e-8)
    adv_olmo_bal = rewards_bal - mu_bal

    # Simulation 3: Reasoning Token Lengths & Budget Penalties
    lengths = np.random.normal(110, 35, 100).clip(30, 220).astype(int)
    budget = 130
    penalties = [(-1.0 if l > budget else 1.0) for l in lengths]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Micro-TÜLU: OlmoRL & Reasoning Effort Analysis</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #0d1117;
            color: #c9d1d9;
            margin: 0;
            padding: 24px;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        h1, h2, h3 {{
            color: #58a6ff;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            background: #238636;
            color: #fff;
            margin-right: 8px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }}
        .card {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            text-align: left;
            padding: 10px;
            border-bottom: 1px solid #21262d;
        }}
        th {{
            color: #8b949e;
        }}
        .highlight {{
            color: #f85149;
            font-weight: bold;
        }}
        .success {{
            color: #3fb950;
            font-weight: bold;
        }}
        svg {{
            width: 100%;
            height: auto;
            background: #0d1117;
            border-radius: 6px;
            border: 1px solid #30363d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Micro-TÜLU: OlmoRL & Reasoning Effort Analysis</h1>
        <div>
            <span class="badge">OLMo 3 Algorithmic Upgrades</span>
            <span class="badge">Kimi K3 Budget Control</span>
            <span class="badge">Apple Silicon MPS</span>
        </div>

        <div class="grid">
            <div class="card">
                <h3>1. Advantage Inflation (Difficulty Bias)</h3>
                <p>In vanilla GRPO, dividing by standard deviation &sigma; makes a rare success on an outlier question explode to <strong>+{adv_grpo_rare[-1]:.2f}</strong>, dominating policy gradients.</p>
                <p>OlmoRL unnormalized advantage scales strictly with reward delta (<strong>+{adv_olmo_rare[-1]:.2f}</strong>), eliminating difficulty bias.</p>
                <table>
                    <tr><th>Question Type</th><th>GRPO Advantage (w/ &sigma;)</th><th>OlmoRL Advantage (w/o &sigma;)</th></tr>
                    <tr><td>Rare Success (1/8 pass)</td><td class="highlight">+{adv_grpo_rare[-1]:.2f} (Exploded!)</td><td class="success">+{adv_olmo_rare[-1]:.2f} (Stable)</td></tr>
                    <tr><td>Balanced (4/8 pass)</td><td>+{adv_grpo_bal[-1]:.2f}</td><td>+{adv_olmo_bal[-1]:.2f}</td></tr>
                </table>
            </div>

            <div class="card">
                <h3>2. Asymmetric PPO Clip-Higher</h3>
                <p>Standard PPO clips policy ratios symmetrically at [&plusmn;0.20]. OlmoRL applies <strong>clip-higher</strong> (&epsilon;<sub>low</sub>=0.20, &epsilon;<sub>high</sub>=0.28):</p>
                <ul>
                    <li>Lower bound: <code>1.0 - 0.20 = 0.80</code> (guards against policy collapse).</li>
                    <li>Upper bound: <code>1.0 + 0.28 = 1.28</code> (allows larger policy gains along verified reasoning paths).</li>
                </ul>
            </div>
        </div>

        <div class="card" style="margin-top: 20px;">
            <h3>3. Visual Simulation: Advantage Scaling & Token Budget Enforcement</h3>
            <svg viewBox="0 0 900 320" xmlns="http://www.w3.org/2000/svg">
                <!-- Grid lines -->
                <line x1="50" y1="260" x2="850" y2="260" stroke="#30363d" stroke-width="2"/>
                <line x1="50" y1="40" x2="50" y2="260" stroke="#30363d" stroke-width="2"/>
                <text x="50" y="30" fill="#8b949e" font-size="12">Advantage Magnitude</text>
                <text x="800" y="285" fill="#8b949e" font-size="12">Rollout ID</text>

                <!-- Zero line -->
                <line x1="50" y1="180" x2="850" y2="180" stroke="#484f58" stroke-dasharray="4,4"/>
                <text x="25" y="185" fill="#8b949e" font-size="11">0.0</text>

                <!-- GRPO Rare Bar -->
                <rect x="120" y="60" width="80" height="120" fill="#f85149" rx="4"/>
                <text x="125" y="50" fill="#f85149" font-size="13" font-weight="bold">GRPO (+2.65)</text>
                <text x="120" y="200" fill="#8b949e" font-size="11">Rare Success</text>

                <!-- OlmoRL Rare Bar -->
                <rect x="230" y="140" width="80" height="40" fill="#3fb950" rx="4"/>
                <text x="235" y="130" fill="#3fb950" font-size="13" font-weight="bold">OlmoRL (+0.88)</text>
                <text x="230" y="200" fill="#8b949e" font-size="11">Rare Success</text>

                <!-- GRPO Balanced Bar -->
                <rect x="420" y="130" width="80" height="50" fill="#58a6ff" rx="4"/>
                <text x="425" y="120" fill="#58a6ff" font-size="13">GRPO (+1.00)</text>
                <text x="420" y="200" fill="#8b949e" font-size="11">Balanced</text>

                <!-- OlmoRL Balanced Bar -->
                <rect x="530" y="155" width="80" height="25" fill="#3fb950" rx="4"/>
                <text x="535" y="145" fill="#3fb950" font-size="13">OlmoRL (+0.50)</text>
                <text x="530" y="200" fill="#8b949e" font-size="11">Balanced</text>

                <!-- Budget Penalty Indicator -->
                <rect x="700" y="180" width="80" height="60" fill="#d29922" rx="4"/>
                <text x="700" y="255" fill="#d29922" font-size="13" font-weight="bold">Penalty (-1.0)</text>
                <text x="690" y="275" fill="#8b949e" font-size="11">Over-Budget (>130t)</text>
            </svg>
        </div>
    </div>
</body>
</html>
"""
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Also generate a standalone SVG version
    svg_file = out_dir / "olmorl_advantage_comparison.svg"
    svg_content = f"""<svg viewBox="0 0 800 280" xmlns="http://www.w3.org/2000/svg" style="background:#0d1117; font-family:sans-serif;">
        <text x="30" y="35" fill="#58a6ff" font-size="16" font-weight="bold">OlmoRL Unnormalized Advantage vs Vanilla GRPO</text>
        <line x1="60" y1="220" x2="740" y2="220" stroke="#30363d" stroke-width="2"/>
        <line x1="60" y1="50" x2="60" y2="220" stroke="#30363d" stroke-width="2"/>
        <line x1="60" y1="170" x2="740" y2="170" stroke="#484f58" stroke-dasharray="4,4"/>
        <text x="35" y="175" fill="#8b949e" font-size="11">0.0</text>

        <!-- GRPO Rare -->
        <rect x="140" y="70" width="70" height="100" fill="#f85149" rx="4"/>
        <text x="135" y="60" fill="#f85149" font-size="12" font-weight="bold">GRPO (+2.65)</text>
        <text x="130" y="190" fill="#8b949e" font-size="11">Outlier Question</text>

        <!-- OlmoRL Rare -->
        <rect x="240" y="135" width="70" height="35" fill="#3fb950" rx="4"/>
        <text x="235" y="125" fill="#3fb950" font-size="12" font-weight="bold">OlmoRL (+0.88)</text>
        <text x="240" y="190" fill="#8b949e" font-size="11">Unnormalized</text>

        <!-- GRPO Balanced -->
        <rect x="420" y="120" width="70" height="50" fill="#58a6ff" rx="4"/>
        <text x="415" y="110" fill="#58a6ff" font-size="12">GRPO (+1.00)</text>
        <text x="420" y="190" fill="#8b949e" font-size="11">Balanced 50/50</text>

        <!-- OlmoRL Balanced -->
        <rect x="520" y="145" width="70" height="25" fill="#3fb950" rx="4"/>
        <text x="515" y="135" fill="#3fb950" font-size="12">OlmoRL (+0.50)</text>
        <text x="520" y="190" fill="#8b949e" font-size="11">Unnormalized</text>
    </svg>"""
    with open(svg_file, "w", encoding="utf-8") as f:
        f.write(svg_content)

    return report_file


if __name__ == "__main__":
    p = generate_olmorl_comparison_report()
    print(f"OlmoRL visualization report generated at: {p}")
