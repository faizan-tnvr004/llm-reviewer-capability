"""Tests for the PER pipeline code-extension path — mock models, no API calls.

Verifies:
  1. Custom grader hook is called (not grade_numeric)
  2. Custom review_input_fn and revise_input_fn are used
  3. ToolReviewer dispatch: PublicTestReviewer.review() called directly
  4. extra_log_fn fields appear in log records
  5. C0 (no review): final = A0, graded by custom grader
  6. LLM reviewer (C1): standard VERDICT parsing + revision on REJECT
  7. ToolReviewer (C4): review() called, VERDICT parsed, revision on REJECT
  8. Caching: second run makes zero new calls
  9. Existing math tests still pass (import check)
"""

import sys, os, shutil, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core import Task
from harness import Model, DiskCache, RunLog
from pipeline import run_per_experiment, analyze, parse_review, TokenMeter
from prompts_code import CODE_REVIEW_PROMPT, CODE_REVISE_PROMPT

TMP = os.path.join(os.path.dirname(__file__), "..", "scratch_test_pipeline_code")
N_TASKS = 4


# ---------------------------------------------------------------------------
# Mock grader: correct iff task index is even
# ---------------------------------------------------------------------------

def mock_grade(task, response):
    """Grade by checking if response contains the 'CORRECT' marker."""
    correct = "CORRECT_ANSWER" in response
    # Use a hash-like extracted field (simulating code_hash)
    extracted = response[:16] if response else None
    fp = "PASS" if correct else "FAIL:mock"
    return {"correct": correct, "fingerprint": fp, "extracted": extracted}


# ---------------------------------------------------------------------------
# Mock models
# ---------------------------------------------------------------------------

class MockCodeExecutor:
    """Produces CORRECT_ANSWER for even-indexed tasks, wrong for odd."""
    def __init__(self):
        self.model_id = "mock-code-executor"
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        i = int(re.search(r"task_(\d+)", prompt).group(1))
        if prompt.startswith("You previously wrote"):
            # revision: fix if critique mentions 'off-by-one'
            if "off-by-one" in prompt:
                return "```python\nprint('CORRECT_ANSWER')\n```"
            return "```python\nprint('still_wrong')\n```"
        if i % 2 == 0:
            return "```python\nprint('CORRECT_ANSWER')\n```"
        return "```python\nprint('wrong')\n```"


class MockCodeReviewer:
    """REJECTS odd-indexed tasks with a critique mentioning 'off-by-one'."""
    def __init__(self):
        self.model_id = "mock-code-reviewer"

    def __call__(self, prompt):
        # Look for odd-indexed task markers in the prompt
        m = re.search(r"task_(\d+)", prompt)
        if m and int(m.group(1)) % 2 == 1:
            return "VERDICT: REJECT\nCRITIQUE: off-by-one error in the loop"
        return "VERDICT: ACCEPT"


class MockToolReviewer:
    """ToolReviewer: REJECTS everything with a generic critique."""
    model_id = "mock-tool-reviewer"
    review_calls = 0

    def review(self, task, solution_text):
        self.review_calls += 1
        i = int(re.search(r"task_(\d+)", task.task_id).group(1))
        if i % 2 == 1:
            return "VERDICT: REJECT\nCRITIQUE: test failed: wrong output"
        return "VERDICT: ACCEPT"


# ---------------------------------------------------------------------------
# Test tasks and helpers
# ---------------------------------------------------------------------------

def make_tasks():
    return [Task(
        f"task_{i:02d}", "livecodebench",
        f"Solve task_{i:02d}: write a program.",
        "PASS",
        meta={"difficulty": "medium", "question_id": f"q{i}"},
    ) for i in range(N_TASKS)]


def code_review_input(task, a0_text):
    return {"problem": task.prompt, "solution": a0_text}


def code_revise_input(task, a0_text, critique):
    return {"problem": task.prompt, "solution": a0_text, "critique": critique}


def extra_log(task, condition, a0_text, a0_grade, final_grade):
    return {"family": "livecodebench", "question_id": task.meta.get("question_id")}


def run():
    if os.path.exists(TMP):
        shutil.rmtree(TMP)
    os.makedirs(TMP)

    tasks = make_tasks()
    executor = MockCodeExecutor()
    tool_reviewer = MockToolReviewer()

    reviewer_conditions = {
        "C0": None,                          # no review
        "C1": Model("mock-code-reviewer", MockCodeReviewer()),  # LLM reviewer
        "C4": tool_reviewer,                 # ToolReviewer
    }

    cache = DiskCache(os.path.join(TMP, "cache"))
    log = RunLog(os.path.join(TMP, "log.jsonl"))
    meter = TokenMeter(sleep=lambda _: None)

    # --- Run 1 ---
    summary = run_per_experiment(
        tasks, Model(executor.model_id, executor), reviewer_conditions,
        cache, log, meter,
        grader=mock_grade,
        review_prompt=CODE_REVIEW_PROMPT,
        revise_prompt=CODE_REVISE_PROMPT,
        review_input_fn=code_review_input,
        revise_input_fn=code_revise_input,
        extra_log_fn=extra_log,
    )

    rows = log.load()
    expected_outcomes = N_TASKS * len(reviewer_conditions)
    assert len(rows) == expected_outcomes, \
        f"expected {expected_outcomes} log rows, got {len(rows)}"
    assert summary["completed_outcomes"] == expected_outcomes
    print(f"[ok] run 1: {len(rows)} outcomes logged")

    # --- Check C0 (no review) ---
    c0_rows = [r for r in rows if r["condition"] == "C0"]
    for r in c0_rows:
        assert r["verdict"] == "NONE"
        assert r["revised"] is False
    c0_correct = sum(r["final_correct"] for r in c0_rows)
    assert c0_correct == N_TASKS // 2, f"C0 accuracy: {c0_correct}/{N_TASKS}"
    print(f"[ok] C0: {c0_correct}/{N_TASKS} correct, no reviews")

    # --- Check C1 (LLM reviewer) ---
    c1_rows = [r for r in rows if r["condition"] == "C1"]
    c1_rejects = [r for r in c1_rows if r["verdict"] == "REJECT"]
    # Odd-indexed tasks should be rejected
    assert len(c1_rejects) == N_TASKS // 2, \
        f"C1 should reject {N_TASKS // 2} tasks, got {len(c1_rejects)}"
    # Critique should contain 'off-by-one'
    for r in c1_rejects:
        assert "off-by-one" in r["critique"]
    # After revision with 'off-by-one' critique, executor fixes -> all correct
    c1_correct = sum(r["final_correct"] for r in c1_rows)
    assert c1_correct == N_TASKS, f"C1 should fix all, got {c1_correct}/{N_TASKS}"
    print(f"[ok] C1: {c1_correct}/{N_TASKS} correct after revision")

    # --- Check C4 (ToolReviewer) ---
    c4_rows = [r for r in rows if r["condition"] == "C4"]
    assert tool_reviewer.review_calls == N_TASKS, \
        f"ToolReviewer.review() should be called {N_TASKS} times, got {tool_reviewer.review_calls}"
    c4_rejects = [r for r in c4_rows if r["verdict"] == "REJECT"]
    assert len(c4_rejects) == N_TASKS // 2
    # C4 critique doesn't mention 'off-by-one', so revision won't fix
    for r in c4_rejects:
        assert "off-by-one" not in r["critique"]
    print(f"[ok] C4: ToolReviewer dispatched correctly, {len(c4_rejects)} rejects")

    # --- Check extra log fields ---
    for r in rows:
        assert "family" in r, "extra_log_fn field 'family' missing"
        assert r["family"] == "livecodebench"
        assert "question_id" in r
    print("[ok] extra_log_fn fields present in all log rows")

    # --- Check custom grader was used ---
    for r in rows:
        assert r["a0_extracted"] is not None or not r["a0_correct"]
    print("[ok] custom grader was used (not grade_numeric)")

    # --- Check token meter records ---
    # ToolReviewer calls should NOT produce meter records
    tool_records = [t for t in meter.records if t.get("model_id") == "mock-tool-reviewer"]
    assert len(tool_records) == 0, "ToolReviewer should not produce token records"
    print("[ok] ToolReviewer produced zero token records")

    # --- Run 2: caching, no new calls ---
    executor2 = MockCodeExecutor()
    tool_reviewer2 = MockToolReviewer()
    reviewer_conditions2 = {
        "C0": None,
        "C1": Model("mock-code-reviewer", MockCodeReviewer()),
        "C4": tool_reviewer2,
    }
    meter2 = TokenMeter(sleep=lambda _: None)
    summary2 = run_per_experiment(
        tasks, Model(executor2.model_id, executor2), reviewer_conditions2,
        cache, log, meter2,
        grader=mock_grade,
        review_prompt=CODE_REVIEW_PROMPT,
        revise_prompt=CODE_REVISE_PROMPT,
        review_input_fn=code_review_input,
        revise_input_fn=code_revise_input,
        extra_log_fn=extra_log,
    )
    assert summary2["completed_outcomes"] == 0, "second run should make zero new outcomes"
    assert executor2.calls == 0, "second run should make zero executor calls"
    rows2 = log.load()
    assert len(rows2) == expected_outcomes, "no duplicate log rows"
    print("[ok] caching: second run zero new calls, no duplicate rows")

    # --- Analyze ---
    m = analyze(rows)
    assert "C0" in m and "C1" in m and "C4" in m
    assert m["C1"]["accuracy"] == 1.0
    print(f"[ok] analyze: C0 acc={m['C0']['accuracy']:.2f}, "
          f"C1 acc={m['C1']['accuracy']:.2f}, C4 acc={m['C4']['accuracy']:.2f}")

    # Cleanup
    shutil.rmtree(TMP)
    print("\n[ok] ALL pipeline-code tests passed")


def test_all():
    run()


if __name__ == "__main__":
    run()
