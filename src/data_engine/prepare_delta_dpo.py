"""Delta Learning Preference Dataset Preparation (OLMo 3 / Dolci Think DPO).

Reference:
OLMo 3 Technical Report (AI2, 2026 - Section 4.3):
"Supervised fine-tuning on high-quality external thinking traces quickly saturates...
We introduce Delta Learning for preference tuning: pairing strong model thinking
traces (Qwen-3-32B-Think) with deliberately weaker model thinking traces (Qwen-3-0.6B-Think)
on identical prompts to supply a massive capability delta for contrastive optimization."
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


THINK_REGEX = re.compile(r"<think>\s*(.*?)\s*</think>\s*(.*)", re.DOTALL)


def format_delta_dpo_pair(
    prompt: str,
    chosen_thought: str,
    chosen_answer: str,
    rejected_thought: str,
    rejected_answer: str,
    source: str = "delta_math_reasoning",
) -> dict[str, Any]:
    """Format a Delta Learning preference record with explicit <think> tags.

    Args:
        prompt: User question / problem statement.
        chosen_thought: Multi-step reasoning trace from high-capability model.
        chosen_answer: Accurate final solution.
        rejected_thought: Flawed / superficial reasoning trace from low-capability model.
        rejected_answer: Flawed final solution.
        source: Dataset origin label.

    Returns:
        Standardized DPO JSON record with 'prompt', 'chosen', 'rejected', 'source'.
    """
    chosen_text = f"<think>\n{chosen_thought.strip()}\n</think>\n{chosen_answer.strip()}"
    rejected_text = f"<think>\n{rejected_thought.strip()}\n</think>\n{rejected_answer.strip()}"

    return {
        "prompt": prompt.strip(),
        "chosen": chosen_text,
        "rejected": rejected_text,
        "source": source,
    }


def validate_think_tags(text: str) -> bool:
    """Validate that text starts with <think> and contains closing </think>."""
    if not text.strip().startswith("<think>"):
        return False
    if "</think>" not in text:
        return False
    # Verify open and close count
    return text.count("<think>") == 1 and text.count("</think>") == 1


def extract_think_and_answer(text: str) -> tuple[str, str]:
    """Extract thought trace and final answer from formatted response.

    Returns:
        (thought, answer) tuple, or ("", text) if tags are absent.
    """
    match = THINK_REGEX.search(text.strip())
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", text.strip()


# Seed reasoning problems for synthetic delta generation
SYNTHETIC_DELTA_SEEDS = [
    {
        "prompt": "Solve for x: 3x + 15 = 45.",
        "chosen_thought": "Subtract 15 from both sides: 3x = 45 - 15 = 30. Then divide both sides by 3: x = 30 / 3 = 10. Check: 3(10) + 15 = 30 + 15 = 45.",
        "chosen_answer": "x = 10",
        "rejected_thought": "Subtract 3 from 45: 42. Divide by 15: x = 42 / 15 = 2.8.",
        "rejected_answer": "x = 2.8",
    },
    {
        "prompt": "A train travels 180 km in 3 hours. What is its speed in km/h and m/s?",
        "chosen_thought": "First calculate speed in km/h: Speed = Distance / Time = 180 km / 3 h = 60 km/h. To convert km/h to m/s, multiply by (1000 m / 3600 s) = 5/18: 60 * 5 / 18 = 300 / 18 = 16.67 m/s.",
        "chosen_answer": "60 km/h and 16.67 m/s",
        "rejected_thought": "180 times 3 is 540 km/h. Convert to m/s by multiplying by 60: 32400 m/s.",
        "rejected_answer": "540 km/h and 32,400 m/s",
    },
    {
        "prompt": "If f(x) = 2x^2 - 4x + 7, find the minimum value of f(x).",
        "chosen_thought": "This is a quadratic parabola opening upwards (a = 2 > 0). The vertex occurs at x = -b / (2a) = -(-4) / (2 * 2) = 4 / 4 = 1. Plug x = 1 into f(x): f(1) = 2(1)^2 - 4(1) + 7 = 2 - 4 + 7 = 5.",
        "chosen_answer": "The minimum value is 5 at x = 1.",
        "rejected_thought": "The minimum is at x = 0: f(0) = 7. So minimum is 7.",
        "rejected_answer": "7",
    },
    {
        "prompt": "Write a Python function to check if a string is a palindrome ignoring casing and spaces.",
        "chosen_thought": "Clean the string by removing spaces and converting to lowercase: cleaned = ''.join(c.lower() for c in s if c.isalnum()). Then check if cleaned equals its reverse: cleaned == cleaned[::-1].",
        "chosen_answer": "def is_palindrome(s: str) -> bool:\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]",
        "rejected_thought": "Check if s equals s[::-1] directly without cleaning.",
        "rejected_answer": "def is_palindrome(s: str) -> bool:\n    return s == s[::-1]",
    },
    {
        "prompt": "What is the probability of rolling a sum of 7 with two fair 6-sided dice?",
        "chosen_thought": "Total possible outcomes: 6 * 6 = 36. Outcomes that sum to 7: (1,6), (2,5), (3,4), (4,3), (5,2), (6,1). Total favorable outcomes: 6. Probability = 6 / 36 = 1/6.",
        "chosen_answer": "1/6 (or ~16.67%)",
        "rejected_thought": "Dice numbers go up to 12. 7 is in the middle, so probability is 1/12.",
        "rejected_answer": "1/12",
    },
]


def build_synthetic_delta_dataset(output_path: str | Path, num_samples: int = 100) -> int:
    """Generate a synthetic Delta DPO preference dataset with <think> reasoning traces."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    records = []
    num_seeds = len(SYNTHETIC_DELTA_SEEDS)

    for i in range(num_samples):
        seed = SYNTHETIC_DELTA_SEEDS[i % num_seeds]
        # Slight variation on index for variety
        rec = format_delta_dpo_pair(
            prompt=f"{seed['prompt']} (Case #{i+1})",
            chosen_thought=seed["chosen_thought"],
            chosen_answer=seed["chosen_answer"],
            rejected_thought=seed["rejected_thought"],
            rejected_answer=seed["rejected_answer"],
            source="olmo3_synthetic_delta",
        )
        records.append(rec)

    with open(out_p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Delta Learning DPO Dataset")
    parser.add_argument("--output", type=str, default="data/raw/delta_dpo_train.jsonl", help="Output JSONL path")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to generate")
    args = parser.parse_args()

    count = build_synthetic_delta_dataset(args.output, args.samples)
    print(f"✅ Generated {count} Delta DPO preference pairs at: {args.output}")


if __name__ == "__main__":
    main()
