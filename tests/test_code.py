"""Tests for the code-debugging family.

Run:  python3 tests/test_code.py

Critical tests:
  - every reference function passes its own suite (ground-truth integrity)
  - every injected bug actually breaks >=1 test (task is non-trivial)
  - grader: correct fix -> PASS, buggy code -> failing-test-set fingerprint
  - fingerprint distinguishes different failure patterns, matches identical ones
  - sandbox: infinite loops time out (return CRASH, not hang), malicious/
    broken code doesn't crash the grader
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tasks.code_tasks import (generate_code_tasks, grade_code, extract_code,
                              _run_tests, _problems, _inject_bugs)
import textwrap


def run():
    # ------------------------------------------------------------------
    # 1. Ground-truth integrity: references pass, bugs break
    # ------------------------------------------------------------------
    for p in _problems():
        ref = textwrap.dedent(p["correct"]).strip()
        res = _run_tests(ref, p["name"], p["tests"])
        assert res == [True] * len(p["tests"]), \
            f"reference {p['name']} fails own tests: {res}"
        variants = _inject_bugs(p["name"], p["correct"])
        assert variants, f"no bugs defined for {p['name']}"
        for label, buggy in variants:
            r = _run_tests(textwrap.dedent(buggy).strip(), p["name"], p["tests"])
            assert r is not None, f"buggy {p['name']}/{label} failed to run at all"
            assert not all(r), f"buggy {p['name']}/{label} passes all tests (not a real bug)"
    print("[ok] references pass their suites; every injected bug breaks >=1 test")

    # ------------------------------------------------------------------
    # 2. Generator produces valid tasks
    # ------------------------------------------------------------------
    tasks = generate_code_tasks(n=10, difficulty=2, seed=0)
    assert len(tasks) == 10
    for t in tasks:
        assert t.meta["func_name"] in t.prompt
        assert "```python" in t.prompt
        assert t.ground_truth == "PASS"
    print(f"[ok] generated {len(tasks)} valid code tasks")

    # ------------------------------------------------------------------
    # 3. Grader: a correct fix passes
    # ------------------------------------------------------------------
    t = tasks[0]
    p = next(pp for pp in _problems() if pp["name"] == t.meta["func_name"])
    correct_fix = f"```python\n{textwrap.dedent(p['correct']).strip()}\n```"
    r = grade_code(t, correct_fix)
    assert r["correct"] and r["fingerprint"] == "PASS", f"correct fix not graded PASS: {r}"
    print("[ok] correct fix -> PASS")

    # ------------------------------------------------------------------
    # 4. Grader: a still-buggy answer -> failing-test-set fingerprint
    # ------------------------------------------------------------------
    # Return the ORIGINAL buggy code (no fix) -> should fail some tests
    # Reconstruct the buggy source from the prompt's code block.
    import re
    buggy_block = re.search(r"```python\s*(.*?)```", t.prompt, re.DOTALL).group(1)
    r = grade_code(t, f"```python\n{buggy_block}\n```")
    assert not r["correct"], "unfixed buggy code graded correct"
    assert r["fingerprint"].startswith("FAIL:"), f"expected FAIL fingerprint: {r}"
    print(f"[ok] unfixed code -> {r['fingerprint']}")

    # ------------------------------------------------------------------
    # 5. Fingerprint semantics for D_across
    # ------------------------------------------------------------------
    # Two identical wrong fixes -> same fingerprint
    r2 = grade_code(t, f"```python\n{buggy_block}\n```")
    assert r["fingerprint"] == r2["fingerprint"]
    print("[ok] identical failures share a fingerprint")

    # ------------------------------------------------------------------
    # 6. Robustness: no code, crash, timeout
    # ------------------------------------------------------------------
    assert grade_code(t, "I think the bug is on line 2.")["fingerprint"] == "NO_CODE"

    # code that raises at import time -> CRASH (doesn't break grader)
    fn = t.meta["func_name"]
    broken = f"```python\ndef {fn}(*a, **k):\n    raise ImportError('boom')\n```"
    # note: exception is inside the call, caught per-test -> becomes FAIL not CRASH.
    r_broken = grade_code(t, broken)
    assert not r_broken["correct"]

    # infinite loop -> must TIME OUT and return CRASH, not hang the suite
    loop = f"```python\ndef {fn}(*a, **k):\n    while True:\n        pass\n```"
    r_loop = grade_code(t, loop)
    assert r_loop["fingerprint"] == "CRASH", f"infinite loop not handled: {r_loop}"
    print("[ok] robustness: NO_CODE, exceptions, and infinite-loop timeout handled")

    # ------------------------------------------------------------------
    # 7. Code extraction variants
    # ------------------------------------------------------------------
    fn = t.meta["func_name"]
    fenced = f"Here is my fix:\n```python\ndef {fn}(x):\n    return x\n```\nDone."
    assert extract_code(fenced, fn) is not None
    bare = f"def {fn}(x):\n    return x"
    assert extract_code(bare, fn) is not None
    assert extract_code("no function here", fn) is None
    print("[ok] code extraction (fenced + bare + none)")

    print("\nALL CODE TESTS PASSED")


if __name__ == "__main__":
    run()


def test_all():
    run()

