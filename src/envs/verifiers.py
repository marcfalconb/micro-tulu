"""Deterministic Rule Verifiers for Reinforcement Learning with Verifiable Rewards (RLVR).

TÜLU 3 Reference (Section 6 & Section 6.1):
Replaces unstable, prone-to-hacking neural reward models with deterministic
verification functions:
1. Math Verifier: extracts final answers (boxed or numeric) and assesses symbolic
   equivalence via SymPy.
2. IFEval Verifier: verifies strict, verifiable instruction constraints
   (e.g., word count limits, case formatting, keyword inclusion).
3. Non-EOS Penalty: applies -10.0 penalty if generated response fails to emit EOS.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import sympy
from sympy.parsing.sympy_parser import parse_expr

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    reward: float
    is_correct: bool
    predicted_answer: str
    ground_truth: str
    details: str


def extract_boxed_content(text: str) -> str | None:
    """Extract content from LaTeX \\boxed{...}, handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None

    start = idx + len("\\boxed{")
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None


def extract_math_answer(completion: str) -> str:
    """Extract the candidate mathematical answer from a generated completion."""
    # 1. Look for LaTeX \boxed{...}
    boxed = extract_boxed_content(completion)
    if boxed is not None:
        return boxed.strip()

    # 2. Look for GSM8K standard format: #### 1234
    hash_match = re.search(r"####\s*([^\n\r]+)", completion)
    if hash_match:
        return hash_match.group(1).strip()

    # 3. Look for explicit answer declaration: The answer is X / Answer: X
    decl_match = re.search(r"(?:the\s+answer\s+is|answer\s*[:=])\s*([^\n\.\$]+)", completion, re.IGNORECASE)
    if decl_match:
        return decl_match.group(1).strip()

    # 4. Fallback: extract the last numeric or algebraic token on the last non-empty line
    lines = [line.strip() for line in completion.strip().split("\n") if line.strip()]
    if lines:
        last_line = lines[-1]
        numbers = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", last_line)
        if numbers:
            return numbers[-1]
        return last_line

    return ""


def clean_math_string(s: str) -> str:
    """Normalize a mathematical string by stripping LaTeX formatting artifacts."""
    s = s.strip()
    # Strip dollar signs, text wrappers, and whitespace
    s = re.sub(r"\$+", "", s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = s.replace("\\%", "%").replace("%", "")
    s = s.replace(",", "")  # e.g., 1,000 -> 1000
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("\\le", "<=").replace("\\ge", ">=")
    s = s.replace("\\div", "/")

    # Convert \frac{a}{b} to (a)/(b)
    frac_pattern = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    while frac_pattern.search(s):
        s = frac_pattern.sub(r"(\1)/(\2)", s)

    return s.strip()


def check_math_equivalence(pred_str: str, gt_str: str) -> bool:
    """Check mathematical equivalence using SymPy symbolic comparison with fallback."""
    p_clean = clean_math_string(pred_str)
    g_clean = clean_math_string(gt_str)

    # 1. Exact string match
    if p_clean.lower() == g_clean.lower():
        return True

    # 2. Numeric float equivalence
    try:
        if abs(float(p_clean) - float(g_clean)) < 1e-6:
            return True
    except ValueError:
        pass

    # 3. Symbolic SymPy expression simplification
    try:
        e_pred = parse_expr(p_clean)
        e_gt = parse_expr(g_clean)
        diff = sympy.simplify(e_pred - e_gt)
        if diff == 0:
            return True
    except Exception:
        pass

    return False


def verify_math_completion(completion: str, ground_truth: str) -> tuple[bool, str]:
    """Verify a mathematical completion against ground truth."""
    pred_answer = extract_math_answer(completion)
    is_correct = check_math_equivalence(pred_answer, ground_truth)
    return is_correct, pred_answer


def verify_ifeval_constraints(completion: str, record: dict[str, Any]) -> tuple[bool, str]:
    """Verify instruction-following constraints deterministically."""
    text = completion.strip()
    constraint_type = str(record.get("constraint_type", "")).lower()
    constraint_spec = record.get("constraint", "")
    prompt = record.get("prompt", "")

    # 1. All lowercase (no capital letters allowed)
    if "lowercase" in constraint_type or "no capital letters" in str(prompt).lower():
        has_upper = any(c.isupper() for c in text)
        return (not has_upper), "no capital letters allowed"

    # 2. All uppercase
    if "uppercase" in constraint_type or "all capital letters" in str(prompt).lower():
        has_lower = any(c.islower() for c in text)
        return (not has_lower), "all uppercase required"

    # 3. Word count constraints
    words = text.split()
    word_count = len(words)

    max_word_match = re.search(r"(?:at most|maximum of|fewer than)\s+(\d+)\s+words", str(prompt), re.IGNORECASE)
    if max_word_match:
        max_words = int(max_word_match.group(1))
        return word_count <= max_words, f"word count {word_count} <= {max_words}"

    min_word_match = re.search(r"(?:at least|minimum of|more than)\s+(\d+)\s+words", str(prompt), re.IGNORECASE)
    if min_word_match:
        min_words = int(min_word_match.group(1))
        return word_count >= min_words, f"word count {word_count} >= {min_words}"

    exact_word_match = re.search(r"(?:exactly)\s+(\d+)\s+words", str(prompt), re.IGNORECASE)
    if exact_word_match:
        exact_words = int(exact_word_match.group(1))
        return word_count == exact_words, f"word count {word_count} == {exact_words}"

    # 4. Keyword / delimiter inclusion (e.g., 6 asterisk symbols: ******)
    delim_match = re.search(r"([*#=_~-]{4,})", str(prompt))
    if delim_match:
        delim = delim_match.group(1)
        return (delim in text), f"required delimiter '{delim}' present"

    # 5. Sentence count
    sent_match = re.search(r"exactly\s+(\d+)\s+sentences", str(prompt), re.IGNORECASE)
    if sent_match:
        target_sents = int(sent_match.group(1))
        # Split on sentence terminals . ! ?
        sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        return len(sents) == target_sents, f"sentence count {len(sents)} == {target_sents}"

    # Default fallback: check if ground truth is in text
    gt = str(record.get("ground_truth", "")).strip()
    if gt and gt.lower() in text.lower():
        return True, "ground truth substring match"

    return True, "default passed"


class DeterministicVerifier:
    """Unified reward verifier for RLVR according to TÜLU 3 Section 6."""

    def __init__(self, correct_reward: float = 1.0, non_eos_penalty: float = -10.0) -> None:
        self.correct_reward = correct_reward
        self.non_eos_penalty = non_eos_penalty

    def verify(
        self,
        completion: str,
        record: dict[str, Any],
        has_eos: bool = True,
    ) -> VerificationResult:
        """Evaluate a generated completion against deterministic domain rules."""
        # Check non-EOS penalty
        if not has_eos:
            return VerificationResult(
                reward=self.non_eos_penalty,
                is_correct=False,
                predicted_answer="[NO_EOS]",
                ground_truth=str(record.get("ground_truth", "")),
                details=f"Non-EOS penalty ({self.non_eos_penalty}) applied",
            )

        dataset_name = str(record.get("dataset", "")).lower()
        ground_truth = str(record.get("ground_truth", "")).strip()

        # Math domain (GSM8K, MATH)
        if "math" in dataset_name or "gsm" in dataset_name or "gsm8k" in dataset_name:
            is_correct, pred_ans = verify_math_completion(completion, ground_truth)
            reward = self.correct_reward if is_correct else 0.0
            return VerificationResult(
                reward=reward,
                is_correct=is_correct,
                predicted_answer=pred_ans,
                ground_truth=ground_truth,
                details="SymPy math equivalence verified" if is_correct else "Math answer mismatch",
            )

        # IFEval constraint domain
        if "ifeval" in dataset_name or record.get("constraint_type") is not None:
            is_correct, details = verify_ifeval_constraints(completion, record)
            reward = self.correct_reward if is_correct else 0.0
            return VerificationResult(
                reward=reward,
                is_correct=is_correct,
                predicted_answer="[CONSTRAINT_OUTPUT]",
                ground_truth=ground_truth,
                details=details,
            )

        # Generic math fallback if ground_truth is provided
        if ground_truth:
            is_correct, pred_ans = verify_math_completion(completion, ground_truth)
            reward = self.correct_reward if is_correct else 0.0
            return VerificationResult(
                reward=reward,
                is_correct=is_correct,
                predicted_answer=pred_ans,
                ground_truth=ground_truth,
                details="Generic ground truth verified",
            )

        return VerificationResult(
            reward=0.0,
            is_correct=False,
            predicted_answer="",
            ground_truth="",
            details="No verifier matched",
        )
