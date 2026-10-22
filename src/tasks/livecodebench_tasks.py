"""LiveCodeBench task loader and grader for the code extension.

Loads problems from the LiveCodeBench-Lite dataset on HuggingFace, decodes
test cases, and provides grading + public-test checking.

Dataset: livecodebench/code_generation_lite
Test cases are base64-encoded, then zlib-compressed JSON.

Filters applied:
  - stdin/stdout IO only (no function-call problems)
  - medium and hard difficulty (configurable)
  - exclude special-judge problems (no automated grading possible)

Grader contract (matches rej_quality.py expectations):
  grade_lcb returns:
    correct      : bool            — True iff all hidden tests pass
    fingerprint  : str             — "PASS" | "FAIL:k/n" | "TIMEOUT" | "CRASH" | "NO_CODE"
    extracted    : Optional[str]   — first 16 hex of sha256(normalized_code), or None

  The 'extracted' field is a code hash, so final_extracted == a0_extracted
  evaluates program identity — rej_quality.py's classify() works unchanged.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import zlib
from typing import Any, Dict, List, Optional, Tuple

from core import Task
from prompts_code import CODE_ANSWER_INSTRUCTION
from code_exec import run_tests, normalize_output, code_hash

# ---------------------------------------------------------------------------
# Test case decoding
# ---------------------------------------------------------------------------

def decode_test_cases(encoded: str) -> List[dict]:
    """Decode LiveCodeBench's base64 -> zlib -> JSON test case encoding.

    Returns a list of dicts with keys: input, expected_output.
    Handles both the pickle-encoded and JSON-encoded variants.
    """
    raw = base64.b64decode(encoded)
    decompressed = zlib.decompress(raw)

    # Try JSON first (the lite version uses JSON)
    try:
        data = json.loads(decompressed.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Fallback: pickle (older dataset versions)
        import pickle
        data = pickle.loads(decompressed)

    # Normalize to list of {input, expected_output}
    cases = []
    if isinstance(data, dict):
        inputs = data.get("inputs", data.get("input", []))
        outputs = data.get("outputs", data.get("output", []))
        for inp, out in zip(inputs, outputs):
            cases.append({"input": str(inp), "expected_output": str(out)})
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                cases.append({
                    "input": str(item.get("input", "")),
                    "expected_output": str(item.get("output", item.get("expected_output", ""))),
                })
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                cases.append({"input": str(item[0]), "expected_output": str(item[1])})
    return cases


# ---------------------------------------------------------------------------
# Code extraction from model responses
# ---------------------------------------------------------------------------

_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_program(response: str) -> Optional[str]:
    """Extract the last Python code block from a model response.

    Priority: the LAST ```python ... ``` block (models often iterate).
    Returns None if no code block is found.
    """
    blocks = _CODE_FENCE.findall(response)
    if blocks:
        return blocks[-1].strip()
    return None


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def grade_lcb(task: Task, response: str) -> dict:
    """Grade a LiveCodeBench task by running the extracted program against
    hidden test cases.

    Returns dict matching the grader contract:
      correct     : bool
      fingerprint : str — "PASS" | "FAIL:k/n" | "TIMEOUT" | "CRASH" | "NO_CODE"
      extracted   : Optional[str] — 16-char hex hash of normalized code, or None
    """
    code = extract_program(response)
    if code is None:
        return {"correct": False, "fingerprint": "NO_CODE", "extracted": None}

    chash = code_hash(code)
    test_cases = task.meta["test_cases"]

    results = run_tests(code, test_cases)
    if not results:
        return {"correct": False, "fingerprint": "CRASH", "extracted": chash}

    n_total = len(results)
    n_pass = sum(1 for r in results if r["passed"])
    n_timeout = sum(1 for r in results if r["kind"] == "timeout")
    n_crash = sum(1 for r in results if r["kind"] == "crash")

    if n_pass == n_total:
        return {"correct": True, "fingerprint": "PASS", "extracted": chash}

    if n_timeout > 0 and n_pass == 0:
        return {"correct": False, "fingerprint": "TIMEOUT", "extracted": chash}

    if n_crash > 0 and n_pass == 0:
        return {"correct": False, "fingerprint": "CRASH", "extracted": chash}

    n_fail = n_total - n_pass
    return {"correct": False, "fingerprint": f"FAIL:{n_fail}/{n_total}",
            "extracted": chash}


def public_check(task: Task, code: str) -> dict:
    """Run code against the task's PUBLIC test cases (sample tests from the
    problem statement).

    Returns:
      all_pass       : bool
      first_failure  : Optional[dict] with keys:
        input, expected, actual, kind ("wrong"|"crash"|"timeout")
    """
    public_tests = task.meta.get("public_tests", [])
    if not public_tests:
        return {"all_pass": True, "first_failure": None}

    results = run_tests(code, public_tests)
    for r in results:
        if not r["passed"]:
            return {
                "all_pass": False,
                "first_failure": {
                    "input": r.get("expected", ""),  # we'll fix below
                    "expected": r["expected"],
                    "actual": r["actual"],
                    "kind": r["kind"],
                    "stderr_tail": r.get("stderr_tail", ""),
                },
            }
    return {"all_pass": True, "first_failure": None}


def _fix_public_check_result(result: dict, public_tests: List[dict],
                             run_results: List[dict]) -> dict:
    """Attach the original input text to a public_check failure."""
    if result["first_failure"] is None:
        return result
    for tc, r in zip(public_tests, run_results):
        if not r["passed"]:
            result["first_failure"]["input"] = tc["input"]
            break
    return result


def public_check_with_input(task: Task, code: str) -> dict:
    """Like public_check but ensures first_failure.input is the actual test input."""
    public_tests = task.meta.get("public_tests", [])
    if not public_tests:
        return {"all_pass": True, "first_failure": None}

    results = run_tests(code, public_tests)
    for tc, r in zip(public_tests, results):
        if not r["passed"]:
            return {
                "all_pass": False,
                "first_failure": {
                    "input": tc["input"],
                    "expected": r["expected"],
                    "actual": r["actual"],
                    "kind": r["kind"],
                    "stderr_tail": r.get("stderr_tail", ""),
                },
            }
    return {"all_pass": True, "first_failure": None}


# ---------------------------------------------------------------------------
# Task loader
# ---------------------------------------------------------------------------

def load_lcb_tasks(
    n: int | None = None,
    difficulty: str | List[str] = "medium",
    seed: int = 0,
    exclude_ids: List[str] | None = None,
    raw_dir: str = "run_code_data/raw",
    report: bool = True,
) -> List[Task]:
    """Load LiveCodeBench-Lite problems, filtered and converted to Task objects.

    Args:
        n: Maximum number of tasks to return (None = all passing filters).
        difficulty: One of "easy", "medium", "hard", or a list thereof.
        seed: RNG seed for deterministic shuffling.
        exclude_ids: Problem IDs to exclude (e.g. calibration tasks).
        raw_dir: Directory for cached dataset files.
        report: If True, print a summary of filtering.

    Returns a list of Task objects with meta containing:
        test_cases, public_tests, difficulty, source, contest, question_id
    """
    from datasets import load_dataset
    import random

    ds = load_dataset("livecodebench/code_generation_lite",
                      split="test", trust_remote_code=True)

    if isinstance(difficulty, str):
        difficulty = [difficulty]
    difficulty_set = set(d.lower() for d in difficulty)
    exclude_set = set(exclude_ids or [])

    kept = []
    n_diff, n_io, n_judge, n_exclude, n_decode_err = 0, 0, 0, 0, 0

    for row in ds:
        # Difficulty filter
        row_diff = str(row.get("difficulty", "")).lower().strip()
        if row_diff not in difficulty_set:
            n_diff += 1
            continue

        # IO type filter: stdin/stdout only
        io_type = str(row.get("question_type", row.get("io_type", ""))).lower()
        if io_type and io_type not in ("stdin", "stdio", "standard"):
            # Only exclude if the field explicitly says something other than stdin
            pass  # we'll handle this below

        # Exclude special-judge problems
        if row.get("is_special_judge", False) or row.get("special_judge", False):
            n_judge += 1
            continue

        # Exclude specific IDs
        qid = str(row.get("question_id", row.get("id", "")))
        if qid in exclude_set:
            n_exclude += 1
            continue

        # Decode test cases
        encoded_tests = row.get("test_cases", row.get("hidden_test_cases", ""))
        if not encoded_tests:
            n_decode_err += 1
            continue
        try:
            test_cases = decode_test_cases(encoded_tests)
        except Exception:
            n_decode_err += 1
            continue
        if not test_cases:
            n_decode_err += 1
            continue

        # Decode public test cases (sample tests from the problem statement)
        public_encoded = row.get("public_test_cases", row.get("sample_test_cases", ""))
        public_tests = []
        if public_encoded:
            try:
                public_tests = decode_test_cases(public_encoded)
            except Exception:
                pass  # no public tests is fine, just means C4 can't fire

        kept.append({
            "row": row,
            "test_cases": test_cases,
            "public_tests": public_tests,
            "question_id": qid,
        })

    rng = random.Random(seed)
    rng.shuffle(kept)
    if n is not None:
        kept = kept[:n]

    if report:
        print(f"[lcb] difficulties={difficulty_set}: {len(kept)} kept "
              f"(excluded: {n_diff} difficulty, {n_judge} special-judge, "
              f"{n_exclude} by-id, {n_decode_err} decode-error, {n_io} io-type). "
              f"Returning {len(kept)}.")

    tasks: List[Task] = []
    for i, item in enumerate(kept):
        row = item["row"]
        problem_text = row.get("question_content", row.get("problem", ""))

        prompt = (
            f"Solve the following programming problem.\n\n"
            f"{problem_text}"
            + CODE_ANSWER_INSTRUCTION
        )

        tasks.append(Task(
            task_id=f"lcb_{item['question_id']}",
            family="livecodebench",
            prompt=prompt,
            ground_truth="PASS",
            meta={
                "test_cases": item["test_cases"],
                "public_tests": item["public_tests"],
                "difficulty": row.get("difficulty", ""),
                "source": row.get("platform", row.get("source", "")),
                "contest": row.get("contest", ""),
                "question_id": item["question_id"],
            },
        ))

    return tasks


def save_problem_ids(tasks: List[Task], path: str) -> None:
    """Save the selected problem IDs to a JSON file for pre-registration."""
    ids = [t.meta["question_id"] for t in tasks]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"problem_ids": ids, "n": len(ids)}, f, indent=2)
    print(f"[lcb] saved {len(ids)} problem IDs to {path}")


def load_problem_ids(path: str) -> List[str]:
    """Load problem IDs from a pre-registration file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["problem_ids"]
