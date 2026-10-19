"""Unit tests for 8-Gram Decontamination Engine."""

import json
from pathlib import Path

import pytest
from src.data_engine.decontaminate import (
    DecontaminationScanner,
    NGramIndex,
    decontaminate_file,
    extract_ngrams,
    generate_html_report,
    tokenize_text,
)


def test_tokenize_text():
    text = "Hello world! This is a test: 123 tokens."
    tokens = tokenize_text(text)
    assert "hello" in tokens
    assert "world" in tokens
    assert "!" in tokens
    assert "123" in tokens


def test_extract_ngrams_length_checks():
    short_text = ["only", "three", "tokens"]
    assert extract_ngrams(short_text, n=8) == []

    exact_eight = ["one", "two", "three", "four", "five", "six", "seven", "eight"]
    ngrams = extract_ngrams(exact_eight, n=8)
    assert len(ngrams) == 1
    assert ngrams[0] == tuple(exact_eight)

    nine_tokens = exact_eight + ["nine"]
    ngrams = extract_ngrams(nine_tokens, n=8)
    assert len(ngrams) == 2


def test_exact_overlap_flagged_contaminated():
    index = NGramIndex(n=8)
    prompt = "Janet has 16 eggs. She bakes 3 cakes using 4 eggs each. How many remain?"
    index.add_instance(eval_id="gsm8k_1", benchmark="gsm8k", prompt=prompt)

    scanner = DecontaminationScanner(index, threshold=0.50)
    # Identical training prompt
    matches = scanner.scan_prompt(train_id="train_1", prompt_text=prompt)
    assert len(matches) == 1
    assert matches[0].eval_id == "gsm8k_1"
    assert matches[0].overlap_ratio > 0.50


def test_sub_threshold_overlap_not_flagged():
    index = NGramIndex(n=8)
    # Long benchmark prompt
    prompt = (
        "Solve the following polynomial equation carefully: "
        "find all real solutions x for x^4 - 5x^2 + 4 = 0, "
        "and show your work step by step with intermediate factorizations."
    )
    index.add_instance(eval_id="math_1", benchmark="math500", prompt=prompt)

    scanner = DecontaminationScanner(index, threshold=0.50)
    # Training prompt only shares one generic phrase: "and show your work step by step with intermediate"
    partial_train = (
        "Write an essay about Shakespeare's Hamlet and show your work step by step with intermediate drafts."
    )
    matches = scanner.scan_prompt(train_id="train_safe", prompt_text=partial_train)
    # The overlap ratio on the long math benchmark should be well below 50%
    assert len(matches) == 0


def test_extract_prompt_from_record():
    index = NGramIndex(n=8)
    scanner = DecontaminationScanner(index)

    # Test SFT format
    sft_rec = {
        "id": "sft_123",
        "messages": [
            {"role": "user", "content": "What is gravity?"},
            {"role": "assistant", "content": "Gravity is a fundamental force..."},
        ],
    }
    rec_id, text = scanner.extract_prompt_from_record(sft_rec)
    assert rec_id == "sft_123"
    assert text == "What is gravity?"

    # Test DPO format
    dpo_rec = {
        "id": "dpo_456",
        "prompt": "Explain photosynthesis.",
        "chosen": "Photosynthesis is...",
        "rejected": "It makes plants green.",
    }
    rec_id, text = scanner.extract_prompt_from_record(dpo_rec)
    assert rec_id == "dpo_456"
    assert text == "Explain photosynthesis."


def test_decontaminate_file_end_to_end(tmp_path: Path):
    # Setup test benchmark
    bench_file = tmp_path / "benchmarks.jsonl"
    eval_prompt = "A farmer has 15 sheep and all but 8 die. How many are left?"
    bench_file.write_text(
        json.dumps({"id": "eval_sheep", "benchmark": "gsm8k", "prompt": eval_prompt}) + "\n"
    )

    index = NGramIndex(n=8)
    index.load_benchmarks(bench_file)

    # Setup train records: 1 clean, 1 contaminated
    train_file = tmp_path / "raw_train.jsonl"
    train_records = [
        {
            "id": "clean_1",
            "messages": [
                {"role": "user", "content": "Write a python function to compute fibonacci numbers."},
                {"role": "assistant", "content": "def fib(n): return n if n <= 1 else fib(n-1)+fib(n-2)"},
            ],
        },
        {
            "id": "contam_1",
            "messages": [
                {"role": "user", "content": eval_prompt},
                {"role": "assistant", "content": "There are 8 sheep left."},
            ],
        },
    ]
    with open(train_file, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")

    clean_out = tmp_path / "clean.jsonl"
    contam_out = tmp_path / "contam.jsonl"
    html_out = tmp_path / "report.html"

    total, clean_cnt, matches = decontaminate_file(
        input_path=train_file,
        output_path=clean_out,
        benchmark_index=index,
        log_path=contam_out,
        html_report_path=html_out,
        threshold=0.50,
    )

    assert total == 2
    assert clean_cnt == 1
    assert len(matches) == 1
    assert matches[0].train_id == "contam_1"
    assert clean_out.exists()
    assert contam_out.exists()
    assert html_out.exists()

    # Verify clean file has only clean_1
    clean_lines = [json.loads(l) for l in clean_out.read_text().strip().split("\n")]
    assert len(clean_lines) == 1
    assert clean_lines[0]["id"] == "clean_1"
