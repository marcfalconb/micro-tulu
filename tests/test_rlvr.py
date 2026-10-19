"""Unit tests for Phase 4: Reinforcement Learning with Verifiable Rewards (RLVR)."""

from pathlib import Path
from src.envs.verifiers import (
    DeterministicVerifier,
    check_math_equivalence,
    extract_boxed_content,
    extract_math_answer,
    verify_ifeval_constraints,
    verify_math_completion,
)
from src.training.rlvr_visual_logger import RLVRVisualLogger


def test_extract_boxed_content():
    text_simple = "The solution is \\boxed{42}."
    assert extract_boxed_content(text_simple) == "42"

    text_nested = "Hence, the domain is \\boxed{\\frac{3}{4}}."
    assert extract_boxed_content(text_nested) == "\\frac{3}{4}"

    text_none = "The answer is 12."
    assert extract_boxed_content(text_none) is None


def test_extract_math_answer():
    assert extract_math_answer("The final answer is \\boxed{2x + 1}") == "2x + 1"
    assert extract_math_answer("Some working out... #### 1104\n") == "1104"
    assert extract_math_answer("Therefore, the answer is 25.") == "25"


def test_check_math_equivalence_sympy():
    # Fractions vs decimal
    assert check_math_equivalence("\\frac{1}{2}", "0.5") is True
    assert check_math_equivalence("2/4", "1/2") is True

    # Algebraic simplification
    assert check_math_equivalence("3*x + 7", "7 + 3*x") is True

    # Numeric integer
    assert check_math_equivalence("1,104", "1104") is True
    assert check_math_equivalence("42.0", "42") is True

    # Differing values
    assert check_math_equivalence("10", "11") is False


def test_verify_ifeval_constraints():
    rec_lower = {"constraint_type": "lowercase", "prompt": "Explain gravity in all lowercase letters."}
    assert verify_ifeval_constraints("gravity is a force", rec_lower)[0] is True
    assert verify_ifeval_constraints("Gravity is a force", rec_lower)[0] is False

    rec_word_count = {"prompt": "Write at most 5 words about space."}
    assert verify_ifeval_constraints("space is very big", rec_word_count)[0] is True
    assert verify_ifeval_constraints("space is very big and dark and empty", rec_word_count)[0] is False

    rec_delim = {"prompt": "Separate responses with 6 asterisks: ******"}
    assert verify_ifeval_constraints("Part 1 ****** Part 2", rec_delim)[0] is True
    assert verify_ifeval_constraints("Part 1 and Part 2", rec_delim)[0] is False


def test_deterministic_verifier_non_eos_penalty():
    verifier = DeterministicVerifier(correct_reward=1.0, non_eos_penalty=-10.0)
    record = {"dataset": "gsm8k", "ground_truth": "42"}

    # Normal correct completion
    res_correct = verifier.verify("#### 42", record, has_eos=True)
    assert res_correct.is_correct is True
    assert res_correct.reward == 1.0

    # Cut off without EOS
    res_no_eos = verifier.verify("#### 42", record, has_eos=False)
    assert res_no_eos.is_correct is False
    assert res_no_eos.reward == -10.0


def test_rlvr_visual_logger_and_report(tmp_path: Path):
    logger = RLVRVisualLogger(output_dir=tmp_path, run_name="test_rlvr")
    logger.log_step(
        step=1, mean_reward=0.25, accuracy=0.25, loss=0.15, kl_div=0.01, avg_response_len=45.0, lr=3e-7
    )
    logger.log_step(
        step=2, mean_reward=0.75, accuracy=0.75, loss=0.08, kl_div=0.02, avg_response_len=52.0, lr=2.7e-7
    )
    logger.log_rollout_sample(
        prompt="What is 5*5?",
        completion="5*5 is 25 #### 25",
        extracted_answer="25",
        ground_truth="25",
        is_correct=True,
        reward=1.0,
        details="SymPy math equivalence verified",
    )

    report_path = logger.generate_html_report({"learning_rate": 3e-7, "group_size": 4})
    assert report_path.exists()
    content = report_path.read_text()
    assert "Micro-TÜLU: RLVR Training Report" in content
    assert "What is 5*5?" in content
    assert "PASSED" in content
