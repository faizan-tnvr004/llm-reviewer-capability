"""Unit tests for LiveCodeBench task loader, filters, decoder, and ID persistence.

Verifies Task T3 acceptance criteria:
  1. Same seed gives identical task IDs; problem_ids round-trips.
  2. No functional-type or special-judge problems pass the filter in a 200-problem sample.
  3. Decoding works for both public tests (JSON) and private tests (base64+zlib+pickle+JSON).
  4. Task metadata and prompt formatting match specification.
"""

from __future__ import annotations

import base64
import json
import os
import pickle
import sys
import tempfile
import zlib
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tasks.livecodebench_tasks import (
    load_lcb_tasks,
    save_problem_ids,
    load_problem_ids,
    decode_test_cases,
    extract_program,
    normalize_code,
    grade_lcb,
    is_special_judge_statement,
    SPECIAL_JUDGE_PHRASES,
)
from core import Task
from prompts_code import CODE_ANSWER_INSTRUCTION


RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "run_code_data", "raw")


# ---------------------------------------------------------------------------
# Test case decoding tests
# ---------------------------------------------------------------------------

def test_decode_test_cases_json_string():
    """Verify decoding direct JSON string (as used in public_test_cases)."""
    cases = [{"input": "1 2\n", "output": "3\n", "testtype": "stdin"}]
    encoded = json.dumps(cases)
    decoded = decode_test_cases(encoded)
    assert len(decoded) == 1
    assert decoded[0]["input"] == "1 2\n"
    assert decoded[0]["expected_output"] == "3\n"
    assert decoded[0]["testtype"] == "stdin"


def test_decode_test_cases_pickle_compressed():
    """Verify decoding base64 -> zlib -> pickle -> JSON (private_test_cases)."""
    cases = [{"input": "10\n", "output": "20\n", "testtype": "stdin"}]
    json_str = json.dumps(cases)
    pickled = pickle.dumps(json_str)
    compressed = zlib.compress(pickled)
    b64 = base64.b64encode(compressed).decode("ascii")

    decoded = decode_test_cases(b64)
    assert len(decoded) == 1
    assert decoded[0]["input"] == "10\n"
    assert decoded[0]["expected_output"] == "20\n"


def test_decode_test_cases_empty_and_invalid():
    """Verify decoder gracefully handles empty or invalid inputs."""
    assert decode_test_cases("") == []
    assert decode_test_cases(None) == []
    assert decode_test_cases("not valid base64 or json") == []


# ---------------------------------------------------------------------------
# Filter and special judge tests
# ---------------------------------------------------------------------------

def test_is_special_judge_statement():
    """Verify special-judge phrase detection."""
    assert is_special_judge_statement("Print any of them if multiple exist.")
    assert is_special_judge_statement("Output with absolute or relative error at most 1e-6.")
    assert is_special_judge_statement("If there are multiple answers, print any.")
    assert not is_special_judge_statement("Print the single integer representing the minimum cost.")


# ---------------------------------------------------------------------------
# Loader, sampling, and round-trip tests (requires raw data)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(RAW_DIR), reason="Raw dataset not downloaded yet")
def test_same_seed_gives_same_ids():
    """Acceptance criterion: Same seed gives the same ids."""
    tasks_a = load_lcb_tasks(n=20, seed=42, raw_dir=RAW_DIR)
    tasks_b = load_lcb_tasks(n=20, seed=42, raw_dir=RAW_DIR)
    assert len(tasks_a) == 20
    assert len(tasks_b) == 20
    ids_a = [t.task_id for t in tasks_a]
    ids_b = [t.task_id for t in tasks_b]
    assert ids_a == ids_b


@pytest.mark.skipif(not os.path.exists(RAW_DIR), reason="Raw dataset not downloaded yet")
def test_different_seeds_give_different_orders():
    """Verify different seeds produce different permutations."""
    tasks_a = load_lcb_tasks(n=30, seed=1, raw_dir=RAW_DIR)
    tasks_b = load_lcb_tasks(n=30, seed=2, raw_dir=RAW_DIR)
    ids_a = [t.task_id for t in tasks_a]
    ids_b = [t.task_id for t in tasks_b]
    assert ids_a != ids_b


@pytest.mark.skipif(not os.path.exists(RAW_DIR), reason="Raw dataset not downloaded yet")
def test_problem_ids_round_trip():
    """Acceptance criterion: problem_ids round-trips correctly."""
    tasks = load_lcb_tasks(n=15, seed=123, raw_dir=RAW_DIR)
    assert len(tasks) == 15
    original_ids = [t.task_id for t in tasks]

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "test_problem_ids.json")
        save_problem_ids(path, tasks)

        loaded_ids = load_problem_ids(path)
        assert loaded_ids == original_ids

        # Load tasks using the pre-registered IDs: order and content must match exactly
        reloaded_tasks = load_lcb_tasks(problem_ids=loaded_ids, raw_dir=RAW_DIR)
        assert len(reloaded_tasks) == len(tasks)
        assert [t.task_id for t in reloaded_tasks] == original_ids
        for t_orig, t_reload in zip(tasks, reloaded_tasks):
            assert t_orig.task_id == t_reload.task_id
            assert t_orig.meta["question_id"] == t_reload.meta["question_id"]
            assert t_orig.prompt == t_reload.prompt


@pytest.mark.skipif(not os.path.exists(RAW_DIR), reason="Raw dataset not downloaded yet")
def test_no_functional_or_special_judge_in_sample():
    """Acceptance criterion: No functional-type or special-judge problem passes the filter in a 200-problem sample."""
    sample = load_lcb_tasks(n=200, seed=0, raw_dir=RAW_DIR)
    assert len(sample) > 0

    for t in sample:
        # Check task attributes
        assert t.family == "code_lcb"
        assert t.ground_truth == "PASS"
        assert t.prompt.endswith(CODE_ANSWER_INSTRUCTION.strip())

        # Check required metadata keys
        meta = t.meta
        for k in ("statement", "public_tests", "hidden_tests", "contest_date",
                  "difficulty", "platform", "question_id"):
            assert k in meta, f"Missing meta key {k} in task {t.task_id}"

        # 1. Platform must be stdin-based (AtCoder / Codeforces), never LeetCode
        platform = meta["platform"].lower()
        assert platform in ("atcoder", "codeforces"), f"Unexpected platform: {platform}"

        # 2. No functional starter code
        assert "class solution:" not in meta["statement"].lower()

        # 3. All tests must be stdin testtype
        for tc in meta["public_tests"]:
            if isinstance(tc, dict) and "testtype" in tc:
                assert tc["testtype"] == "stdin", f"Non-stdin test in public_tests for {t.task_id}"
        for tc in meta["hidden_tests"]:
            if isinstance(tc, dict) and "testtype" in tc:
                assert tc["testtype"] == "stdin", f"Non-stdin test in hidden_tests for {t.task_id}"

        # 4. No special-judge phrases in statement
        content = meta["statement"].lower()
        for phrase in SPECIAL_JUDGE_PHRASES:
            assert phrase not in content, f"Special judge phrase '{phrase}' found in {t.task_id}"


# ---------------------------------------------------------------------------
# Code normalization and extraction tests
# ---------------------------------------------------------------------------

def test_extract_program():
    """Verify extracting the last python code block."""
    resp = "Here is some code:\n```python\nprint('first')\n```\nWait, better:\n```python\nprint('final')\n```"
    extracted = extract_program(resp)
    assert extracted == "print('final')"

    no_code = "Just plain text with no fences."
    assert extract_program(no_code) is None


def test_normalize_code():
    """Verify normalizing line endings, trailing whitespace, and blank lines."""
    raw = "def foo():   \r\n    return 42  \r\n\r\n"
    norm = normalize_code(raw)
    assert norm == "def foo():\n    return 42"


def test_grade_lcb_fail_fingerprint_semantics(monkeypatch):
    """Verify that for every FAIL:k/n:
    - n == len(hidden_tests) (TOTAL hidden tests)
    - tests_run == k (1-indexed first failing test)
    """
    from unittest.mock import patch

    # Mock Task with 5 hidden tests
    hidden_tests = [
        {"input": f"{i}", "expected_output": f"{i}", "output": f"{i}", "testtype": "stdin"}
        for i in range(1, 6)
    ]
    task = Task(
        task_id="test_semantics_p1",
        family="code_lcb",
        prompt="Solve problem",
        ground_truth="PASS",
        meta={"hidden_tests": hidden_tests},
    )

    # Case 1: First test fails immediately (k=1)
    with patch("tasks.livecodebench_tasks.run_tests") as mock_run:
        mock_run.return_value = [
            {"passed": False, "expected": "1", "actual": "wrong", "kind": "wrong", "stderr_tail": ""}
        ]
        grade = grade_lcb(task, "```python\nprint(0)\n```")
        assert not grade["correct"]
        assert grade["fingerprint"] == "FAIL:1/5"
        assert grade["tests_run"] == 1
        fp_parts = grade["fingerprint"].split(":")[-1].split("/")
        k, n = int(fp_parts[0]), int(fp_parts[1])
        assert n == len(hidden_tests)
        assert grade["tests_run"] == k

    # Case 2: Tests 1, 2 pass; test 3 fails (k=3)
    with patch("tasks.livecodebench_tasks.run_tests") as mock_run:
        mock_run.return_value = [
            {"passed": True, "expected": "1", "actual": "1", "kind": "pass", "stderr_tail": ""},
            {"passed": True, "expected": "2", "actual": "2", "kind": "pass", "stderr_tail": ""},
            {"passed": False, "expected": "3", "actual": "wrong", "kind": "wrong", "stderr_tail": ""},
        ]
        grade = grade_lcb(task, "```python\nprint(x)\n```")
        assert not grade["correct"]
        assert grade["fingerprint"] == "FAIL:3/5"
        assert grade["tests_run"] == 3
        fp_parts = grade["fingerprint"].split(":")[-1].split("/")
        k, n = int(fp_parts[0]), int(fp_parts[1])
        assert n == len(hidden_tests)
        assert grade["tests_run"] == k

