"""Unit tests for Phase 5: Standardized Evaluation Harness."""

from pathlib import Path
from src.eval.evaluate import (
    BenchmarkItem,
    EvaluationSample,
    StageResult,
    extract_mmlu_choice,
    generate_scoreboard_html,
    load_eval_suite,
)


def test_extract_mmlu_choice():
    assert extract_mmlu_choice("The answer is (B).") == "B"
    assert extract_mmlu_choice("Therefore, the correct answer is D.") == "D"
    assert extract_mmlu_choice("A) Option text") == "A"
    assert extract_mmlu_choice("C") == "C"
    assert extract_mmlu_choice("(A)") == "A"


def test_generate_scoreboard_html(tmp_path: Path):
    sample = EvaluationSample(
        benchmark="gsm8k",
        prompt="Janet has 16 eggs...",
        generated_text="4 eggs left.",
        extracted_answer="4",
        ground_truth="4",
        is_correct=True,
        details="SymPy verified",
    )

    stage_base = StageResult(
        stage_name="Base Model",
        checkpoint_path="unsloth/Llama-3.2-1B",
        gsm8k_accuracy=10.0,
        math_accuracy=5.0,
        ifeval_accuracy=20.0,
        mmlu_accuracy=35.0,
        average_score=17.5,
        development_average=11.6,
        unseen_average=35.0,
        samples=[sample],
    )

    stage_sft = StageResult(
        stage_name="Stage 1: SFT",
        checkpoint_path="checkpoints/sft",
        gsm8k_accuracy=30.0,
        math_accuracy=15.0,
        ifeval_accuracy=50.0,
        mmlu_accuracy=40.0,
        average_score=33.7,
        development_average=31.6,
        unseen_average=40.0,
        samples=[sample],
    )

    out_file = tmp_path / "scoreboard.html"
    res_path = generate_scoreboard_html([stage_base, stage_sft], output_html_path=out_file)

    assert res_path.exists()
    content = res_path.read_text()
    assert "Micro-TÜLU: Post-Training Progression Scoreboard" in content
    assert "Base Model" in content
    assert "Stage 1: SFT" in content
    assert "<svg" in content
    assert "Janet has 16 eggs..." in content


def test_load_eval_suite_structure():
    items = load_eval_suite(benchmarks_path="data/eval/benchmarks.jsonl", limit_per_bench=3)
    assert len(items) > 0
    # Check that development benchmarks exist
    dev_benches = {item.benchmark for item in items if item.suite == "development"}
    assert "gsm8k" in dev_benches or "math500" in dev_benches or "ifeval" in dev_benches
