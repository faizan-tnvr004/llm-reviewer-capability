"""Code-debugging task family.

TASK: the model is shown a buggy Python function plus a docstring describing
intended behaviour, and must return a corrected version. We run the model's
code against a hidden test suite. Correct = all tests pass.

WHY THIS FAMILY: real code with subtle bugs produces *conceptual* errors even
in strong models, and the answer is discrete (which tests pass). The
FINGERPRINT is the set of failing tests, so two models whose fixes fail the
SAME tests share a fingerprint — the correlated-error signal D_across needs.

GROUND TRUTH: each task is built from a hand-written CORRECT reference function
plus its test suite. We verify the reference passes all tests, then inject a
bug and verify the buggy version fails at least one test. This guarantees the
task is well-formed (fixable, and the fix is checkable).

SAFETY: model code is executed in a separate subprocess with a hard timeout,
in a temp directory, with no arguments. This is pragmatic isolation, not a
true sandbox — adequate for small-model output in a research setting, but
noted as a limitation. Do NOT run this on untrusted adversarial input.
"""

from __future__ import annotations

import random
import re
import subprocess
import sys
import tempfile
import textwrap
import os
from typing import List, Optional, Tuple

from core import Task

EXEC_TIMEOUT = 8  # seconds per model-code test run


# ---------------------------------------------------------------------------
# Reference problems: (name, correct_source, tests_source)
# Each `tests` is a list of (call_expression, expected_repr) pairs that will be
# turned into assertions. Keeping tests as data lets us compute the failing-test
# SET precisely (the fingerprint).
# ---------------------------------------------------------------------------

def _problems():
    return [
        {
            "name": "count_vowels",
            "docstring": "Return the number of vowels (a, e, i, o, u, case-insensitive) in s.",
            "correct": '''
def count_vowels(s):
    return sum(1 for c in s.lower() if c in "aeiou")
''',
            "signature": "count_vowels(s)",
            "tests": [
                ('count_vowels("hello")', "2"),
                ('count_vowels("AEIOU")', "5"),
                ('count_vowels("xyz")', "0"),
                ('count_vowels("")', "0"),
                ('count_vowels("Programming")', "3"),
            ],
        },
        {
            "name": "second_largest",
            "docstring": "Return the second-largest distinct value in the list nums (len>=2, may have duplicates).",
            "correct": '''
def second_largest(nums):
    distinct = sorted(set(nums))
    return distinct[-2]
''',
            "signature": "second_largest(nums)",
            "tests": [
                ('second_largest([1, 2, 3])', "2"),
                ('second_largest([5, 5, 4])', "4"),
                ('second_largest([10, 20])', "10"),
                ('second_largest([3, 3, 3, 1])', "1"),
                ('second_largest([-1, -2, -3])', "-2"),
            ],
        },
        {
            "name": "is_palindrome",
            "docstring": "Return True if s is a palindrome ignoring case and non-alphanumeric chars.",
            "correct": '''
def is_palindrome(s):
    cleaned = [c.lower() for c in s if c.isalnum()]
    return cleaned == cleaned[::-1]
''',
            "signature": "is_palindrome(s)",
            "tests": [
                ('is_palindrome("A man a plan a canal Panama")', "True"),
                ('is_palindrome("racecar")', "True"),
                ('is_palindrome("hello")', "False"),
                ('is_palindrome("")', "True"),
                ('is_palindrome("Ab_a")', "True"),
            ],
        },
        {
            "name": "running_max",
            "docstring": "Return a list where element i is the max of nums[0..i].",
            "correct": '''
def running_max(nums):
    out, cur = [], nums[0]
    for x in nums:
        cur = max(cur, x)
        out.append(cur)
    return out
''',
            "signature": "running_max(nums)",
            "tests": [
                ('running_max([1, 3, 2, 5, 4])', "[1, 3, 3, 5, 5]"),
                ('running_max([5, 4, 3])', "[5, 5, 5]"),
                ('running_max([1])', "[1]"),
                ('running_max([-2, -5, -1])', "[-2, -2, -1]"),
                ('running_max([2, 2, 2])', "[2, 2, 2]"),
            ],
        },
        {
            "name": "sum_even",
            "docstring": "Return the sum of even numbers in nums.",
            "correct": '''
def sum_even(nums):
    return sum(x for x in nums if x % 2 == 0)
''',
            "signature": "sum_even(nums)",
            "tests": [
                ('sum_even([1, 2, 3, 4])', "6"),
                ('sum_even([2, 4, 6])', "12"),
                ('sum_even([1, 3, 5])', "0"),
                ('sum_even([])', "0"),
                ('sum_even([-2, -4, 1])', "-6"),
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Bug injections: each takes correct source and returns buggy source.
# Each bug is designed to break at least one test but keep the code runnable.
# ---------------------------------------------------------------------------

def _inject_bugs(name, src):
    """Return a list of (bug_label, buggy_src) variants for this problem."""
    variants = []
    if name == "count_vowels":
        variants.append(("missing_u", src.replace('"aeiou"', '"aeio"')))
        variants.append(("no_lowercase", src.replace("s.lower()", "s")))
    elif name == "second_largest":
        variants.append(("off_by_one", src.replace("distinct[-2]", "distinct[-1]")))
        variants.append(("no_dedup", src.replace("sorted(set(nums))", "sorted(nums)")))
    elif name == "is_palindrome":
        variants.append(("no_isalnum", src.replace("if c.isalnum()", "if True")))
        variants.append(("wrong_reverse", src.replace("cleaned[::-1]", "cleaned")))
    elif name == "running_max":
        variants.append(("min_instead", src.replace("max(cur, x)", "min(cur, x)")))
        variants.append(("skip_first", src.replace("for x in nums:", "for x in nums[1:]:")))
    elif name == "sum_even":
        variants.append(("odd_instead", src.replace("x % 2 == 0", "x % 2 == 1")))
        variants.append(("wrong_op", src.replace("x % 2 == 0", "x % 2 != 0")))
    return variants


# ---------------------------------------------------------------------------
# Execution + grading
# ---------------------------------------------------------------------------

def _run_tests(func_src: str, func_name: str, tests: List[Tuple[str, str]]) -> Optional[List[bool]]:
    """Run func_src against tests in a subprocess. Returns a list of per-test
    pass/fail booleans, or None if the code failed to even run (syntax error,
    exception at import, timeout)."""
    # Build a script that defines the function, then prints PASS/FAIL per test.
    lines = [func_src, "import json", "results = []"]
    for i, (call, expected) in enumerate(tests):
        lines.append(
            f"try:\n"
            f"    _got = {call}\n"
            f"    results.append(_got == ({expected}))\n"
            f"except Exception:\n"
            f"    results.append(False)"
        )
    lines.append("print(json.dumps(results))")
    script = "\n".join(lines)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "run.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(script)
        try:
            proc = subprocess.run(
                [sys.executable, path],
                capture_output=True, text=True,
                timeout=EXEC_TIMEOUT, cwd=td,
            )
        except subprocess.TimeoutExpired:
            return None
        if proc.returncode != 0:
            return None
        import json
        try:
            out = json.loads(proc.stdout.strip().splitlines()[-1])
            if isinstance(out, list) and len(out) == len(tests):
                return [bool(x) for x in out]
        except Exception:
            return None
    return None


# Extract a Python function from a model response (```python ... ``` or bare def).
_CODE_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(response: str, func_name: str) -> Optional[str]:
    """Pull the function definition from the model's response."""
    # Prefer a fenced code block containing the function.
    for block in _CODE_FENCE.findall(response):
        if f"def {func_name}" in block:
            return textwrap.dedent(block).strip()
    # Fallback: from the first 'def func_name' to end of response.
    idx = response.find(f"def {func_name}")
    if idx != -1:
        return textwrap.dedent(response[idx:]).strip()
    return None


def grade_code(task: Task, response: str) -> dict:
    """Grade by running the model's fix against the test suite.

    fingerprint = failing-test set, e.g. 'FAIL:0,3' (tests 0 and 3 failed),
    'PASS' if all pass, 'NO_CODE' if no function found, 'CRASH' if the code
    didn't run. Two wrong fixes failing the same tests share a fingerprint.
    """
    func_name = task.meta["func_name"]
    tests = task.meta["tests"]

    code = extract_code(response, func_name)
    if code is None:
        return {"correct": False, "fingerprint": "NO_CODE", "extracted": None}

    results = _run_tests(code, func_name, tests)
    if results is None:
        return {"correct": False, "fingerprint": "CRASH", "extracted": code[:200]}

    failing = [i for i, ok in enumerate(results) if not ok]
    if not failing:
        return {"correct": True, "fingerprint": "PASS", "extracted": code[:200]}
    fp = "FAIL:" + ",".join(str(i) for i in failing)
    return {"correct": False, "fingerprint": fp, "extracted": code[:200]}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_code_tasks(n: int = 25, difficulty: int = 2, seed: int = 0) -> List[Task]:
    """Generate n code-debugging tasks by pairing reference problems with
    injected bugs. Each task is validated at generation time: the reference
    must pass all tests and the buggy version must fail at least one."""
    rng = random.Random(seed)
    problems = _problems()

    # Build all valid (problem, bug) pairs, validating each.
    catalog = []
    for p in problems:
        # sanity: reference passes everything
        ref_results = _run_tests(textwrap.dedent(p["correct"]).strip(), p["name"], p["tests"])
        assert ref_results == [True] * len(p["tests"]), \
            f"reference {p['name']} does not pass its own tests: {ref_results}"
        for label, buggy in _inject_bugs(p["name"], p["correct"]):
            buggy_src = textwrap.dedent(buggy).strip()
            bug_results = _run_tests(buggy_src, p["name"], p["tests"])
            # buggy version must RUN but FAIL at least one test
            if bug_results is not None and not all(bug_results):
                catalog.append((p, label, buggy_src))

    if not catalog:
        raise RuntimeError("no valid buggy variants generated")

    rng.shuffle(catalog)
    tasks: List[Task] = []
    for i in range(n):
        p, label, buggy_src = catalog[i % len(catalog)]
        prompt = (
            f"The following Python function is supposed to do this:\n"
            f'"""{p["docstring"]}"""\n\n'
            f"But it contains a bug:\n\n```python\n{buggy_src}\n```\n\n"
            f"Return a corrected version of the function. Put the complete fixed "
            f"function in a single ```python code block. Keep the same function "
            f"name `{p['name']}`."
        )
        tasks.append(Task(
            f"code_{i:03d}", "code", prompt, "PASS",
            meta={"func_name": p["name"], "tests": p["tests"],
                  "bug": label, "difficulty": difficulty},
        ))
    return tasks
