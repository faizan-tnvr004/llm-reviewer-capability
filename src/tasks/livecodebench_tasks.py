"""LiveCodeBench task loader and grader for the code extension.

Loads problems from the LiveCodeBench-Lite dataset on HuggingFace, decodes
test cases, and provides grading + public-test checking.

Dataset: livecodebench/code_generation_lite
Expected fields:
  question_title, question_content, platform, question_id, contest_id,
  contest_date, starter_code, difficulty, public_test_cases,
  private_test_cases, metadata.

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
import json
import os
import pickle
import random
import re
import sys
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from core import Task
from prompts_code import CODE_ANSWER_INSTRUCTION
from code_exec import run_tests, normalize_output, code_hash

SPECIAL_JUDGE_PHRASES = (
    "any of them",
    "print any",
    "if there are multiple",
    "absolute or relative error",
    "relative error",
    "absolute error",
)

_CACHED_ROWS: Optional[List[dict]] = None


# ---------------------------------------------------------------------------
# Test case decoding
# ---------------------------------------------------------------------------

def decode_test_cases(encoded: Any) -> List[dict]:
    """Decode LiveCodeBench test case encoding.

    Handles:
      - Raw list of dicts
      - Direct JSON string (public_test_cases)
      - base64 -> zlib -> pickle -> JSON (private_test_cases)
      - base64 -> zlib -> JSON string / object
    Returns normalized list of dicts with keys:
      'input', 'expected_output', 'output', 'testtype'
    """
    if not encoded:
        return []
    if isinstance(encoded, list):
        data = encoded
    elif isinstance(encoded, str):
        s = encoded.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                data = json.loads(s)
            except Exception:
                data = None
        else:
            data = None

        if data is None:
            try:
                raw = base64.b64decode(encoded)
            except Exception:
                return []
            try:
                decomp = zlib.decompress(raw)
            except Exception:
                decomp = raw

            obj = None
            try:
                obj = pickle.loads(decomp)
            except Exception:
                pass

            if obj is None:
                try:
                    obj = decomp.decode("utf-8", errors="replace")
                except Exception:
                    pass

            if isinstance(obj, str):
                try:
                    obj = json.loads(obj)
                except Exception:
                    pass
            data = obj
    elif isinstance(encoded, (bytes, bytearray)):
        try:
            decomp = zlib.decompress(encoded)
        except Exception:
            decomp = encoded
        obj = None
        try:
            obj = pickle.loads(decomp)
        except Exception:
            pass
        if obj is None:
            try:
                obj = decomp.decode("utf-8", errors="replace")
            except Exception:
                pass
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except Exception:
                pass
        data = obj
    else:
        return []

    # Normalize to list of test case dicts
    cases: List[dict] = []
    if isinstance(data, dict):
        inputs = data.get("inputs", data.get("input", []))
        outputs = data.get("outputs", data.get("output", []))
        for inp, out in zip(inputs, outputs):
            cases.append({
                "input": str(inp),
                "expected_output": str(out),
                "output": str(out),
                "testtype": "stdin",
            })
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                out = str(item.get("output", item.get("expected_output", "")))
                cases.append({
                    "input": str(item.get("input", "")),
                    "expected_output": out,
                    "output": out,
                    "testtype": str(item.get("testtype", "stdin")),
                })
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                out = str(item[1])
                cases.append({
                    "input": str(item[0]),
                    "expected_output": out,
                    "output": out,
                    "testtype": "stdin",
                })
    return cases


# ---------------------------------------------------------------------------
# Code extraction and normalization
# ---------------------------------------------------------------------------

_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_program(response: str) -> Optional[str]:
    """Extract the last Python code block from a model response.

    Priority: the LAST ```python ... ``` block (models often iterate).
    Falls back to any fenced code block, otherwise None.
    """
    if not response:
        return None
    blocks = _CODE_FENCE.findall(response)
    if blocks:
        return blocks[-1].strip()
    generic_blocks = re.findall(r"```\s*\n(.*?)```", response, re.DOTALL)
    if generic_blocks:
        return generic_blocks[-1].strip()
    return None


def normalize_code(code: str) -> str:
    """Normalize code: unify line endings, strip trailing whitespace per line,
    drop trailing blank lines.
    """
    if not code:
        return ""
    lines = code.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [l.rstrip() for l in lines]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Grading and Public Test Checks
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

    norm_code = normalize_code(code)
    chash = code_hash(norm_code)
    test_cases = task.meta.get("hidden_tests", task.meta.get("test_cases", []))

    if not test_cases:
        return {"correct": True, "fingerprint": "PASS", "extracted": chash}

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
    return {
        "correct": False,
        "fingerprint": f"FAIL:{n_fail}/{n_total}",
        "extracted": chash,
    }


def public_check(task: Task, code: str) -> dict:
    """Run code against the task's PUBLIC test cases (sample tests from the
    problem statement).

    Returns:
      all_pass       : bool
      first_failure  : Optional[dict] with keys:
        input, expected, actual, kind ("wrong"|"crash"|"timeout"), stderr_tail
    """
    public_tests = task.meta.get("public_tests", [])
    if not public_tests:
        return {"all_pass": True, "first_failure": None}

    results = run_tests(code, public_tests)
    for tc, r in zip(public_tests, results):
        if not r["passed"]:
            return {
                "all_pass": False,
                "first_failure": {
                    "input": tc.get("input", ""),
                    "expected": r.get("expected", tc.get("expected_output", "")),
                    "actual": r.get("actual", ""),
                    "kind": r.get("kind", "wrong"),
                    "stderr_tail": r.get("stderr_tail", ""),
                },
            }
    return {"all_pass": True, "first_failure": None}


# Compatibility alias for test_reviewer.py
public_check_with_input = public_check


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def is_special_judge_statement(content: str) -> bool:
    """Check if problem statement indicates a special judge."""
    c = (content or "").lower()
    for phrase in SPECIAL_JUDGE_PHRASES:
        if phrase in c:
            return True
    return False


# ---------------------------------------------------------------------------
# Task loader
# ---------------------------------------------------------------------------

def _load_all_rows(raw_dir: str) -> List[dict]:
    """Load and deduplicate all rows across available JSONL files with in-memory caching."""
    global _CACHED_ROWS
    if _CACHED_ROWS is not None:
        return _CACHED_ROWS

    if not os.path.exists(raw_dir):
        return []

    file_order = [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
        "test6.jsonl",
    ]
    present_files = [
        os.path.join(raw_dir, f) for f in file_order
        if os.path.exists(os.path.join(raw_dir, f))
    ]
    for f in os.listdir(raw_dir):
        if f.endswith(".jsonl") and f not in file_order:
            present_files.append(os.path.join(raw_dir, f))

    seen_ids = set()
    all_rows = []
    for p in present_files:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                row = json.loads(line_str)
                qid = str(row.get("question_id", ""))
                if qid and qid not in seen_ids:
                    seen_ids.add(qid)
                    all_rows.append(row)
    _CACHED_ROWS = all_rows
    return _CACHED_ROWS


def load_lcb_tasks(
    n: Optional[int] = None,
    difficulties: Sequence[str] = ("medium", "hard"),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    seed: int = 0,
    problem_ids: Optional[List[str]] = None,
    raw_dir: str = os.path.join("run_code_data", "raw"),
    difficulty: Optional[Union[str, Sequence[str]]] = None,
) -> List[Task]:
    """Load LiveCodeBench-Lite problems, filtered and converted to Task objects.

    Steps (from section 6 Task T3):
      1. Apply Section 4 filters:
         - Keep only problems where every test has testtype == 'stdin' (AtCoder / Codeforces).
           Drop functional (LeetCode) problems.
         - Keep difficulties medium and hard (configurable).
         - Date range filtering (date_from, date_to).
         - Drop special-judge problems (is_special_judge or phrasing like 'any of them',
           'print any', 'if there are multiple', 'absolute or relative error').
      2. Deterministic sampling:
         - If problem_ids is given, load exactly those, in that order.
         - Otherwise, sort candidates and shuffle with random.Random(seed), taking first n.
      3. Build Task(task_id=question_id, family="code_lcb",
                   prompt=statement + CODE_ANSWER_INSTRUCTION, ground_truth="PASS",
                   meta={statement, public_tests, hidden_tests, contest_date,
                         difficulty, platform, question_id})
    """
    if difficulty is not None:
        difficulties = [difficulty] if isinstance(difficulty, str) else difficulty
    all_rows = _load_all_rows(raw_dir)
    diff_set = set(d.lower().strip() for d in difficulties)

    rows_by_id = {str(r.get("question_id", "")): r for r in all_rows}

    # If problem_ids specified, load exactly those in order
    if problem_ids is not None:
        tasks: List[Task] = []
        for pid in problem_ids:
            row = rows_by_id.get(pid)
            if not row:
                continue
            tasks.append(_build_task_from_row(row))
        return tasks

    candidates = []
    for row in all_rows:
        qid = str(row.get("question_id", ""))
        platform = str(row.get("platform", "")).lower().strip()
        difficulty = str(row.get("difficulty", "")).lower().strip()
        contest_date = str(row.get("contest_date", "")).strip()
        statement = row.get("question_content", "")
        starter_code = row.get("starter_code", "")

        # 1. Platform & functional check: only AtCoder / Codeforces, no LeetCode
        if platform not in ("atcoder", "codeforces"):
            continue
        if "class solution:" in starter_code.lower():
            continue

        # 2. Difficulty filter
        if difficulty not in diff_set:
            continue

        # 3. Contest date window filter
        if date_from and contest_date and contest_date < date_from:
            continue
        if date_to and contest_date and contest_date > date_to:
            continue

        # 4. Special judge check
        if row.get("is_special_judge", False) or row.get("special_judge", False):
            continue
        if is_special_judge_statement(statement):
            continue

        # 5. Must have test cases present
        if not row.get("private_test_cases"):
            continue

        candidates.append((contest_date, qid, row))

    # Deterministic sorting prior to shuffling ensures identical behavior across OS/filesystems
    candidates.sort(key=lambda c: (c[0], c[1]))

    rng = random.Random(seed)
    rng.shuffle(candidates)

    if n is not None:
        candidates = candidates[:n]

    tasks = []
    for cdate, qid, row in candidates:
        tasks.append(_build_task_from_row(row))

    return tasks


def _build_task_from_row(
    row: dict,
    public_tests: Optional[List[dict]] = None,
    hidden_tests: Optional[List[dict]] = None,
) -> Task:
    """Construct a Task object from a row dict according to T3 spec."""
    qid = str(row.get("question_id", ""))
    statement = row.get("question_content", "")
    difficulty = str(row.get("difficulty", ""))
    platform = str(row.get("platform", ""))
    contest_date = str(row.get("contest_date", ""))

    if public_tests is None:
        public_tests = decode_test_cases(row.get("public_test_cases"))
    if hidden_tests is None:
        hidden_tests = decode_test_cases(row.get("private_test_cases"))

    prompt = statement + CODE_ANSWER_INSTRUCTION

    meta = {
        "statement": statement,
        "public_tests": public_tests,
        "hidden_tests": hidden_tests,
        "contest_date": contest_date,
        "difficulty": difficulty,
        "platform": platform,
        "question_id": qid,
        "test_cases": hidden_tests,  # backwards compatibility
    }

    return Task(
        task_id=qid,
        family="code_lcb",
        prompt=prompt,
        ground_truth="PASS",
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Problem ID persistence
# ---------------------------------------------------------------------------

def save_problem_ids(arg1: Any, arg2: Any) -> None:
    """Save selected problem IDs and metadata to JSON.

    Supports both (path, tasks) as specified in T3 and (tasks, path).
    """
    if isinstance(arg1, str):
        path, tasks = arg1, arg2
    else:
        tasks, path = arg1, arg2

    ids = [t.meta["question_id"] for t in tasks]
    metadata = [
        {
            "question_id": t.meta["question_id"],
            "contest_date": t.meta.get("contest_date", ""),
            "difficulty": t.meta.get("difficulty", ""),
            "platform": t.meta.get("platform", ""),
        }
        for t in tasks
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"problem_ids": ids, "metadata": metadata, "n": len(ids)}, f, indent=2)
    print(f"[lcb] saved {len(ids)} problem IDs to {path}")


def load_problem_ids(path: str) -> List[str]:
    """Load problem IDs from a pre-registration JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("problem_ids", [])
