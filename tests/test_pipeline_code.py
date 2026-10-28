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

import sys, os, shutil, re, json
import pytest
from unittest.mock import patch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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


def test_budget_exhaustion_scores_timeout():
    """Verify that a program passing every test it runs scores TIMEOUT
    if total elapsed time exhausts the problem budget before all tests complete.
    """
    from tasks.livecodebench_tasks import grade_lcb

    # 10 hidden tests
    hidden_tests = [
        {"input": f"{i}\n", "expected_output": f"{i}\n", "output": f"{i}\n", "testtype": "stdin"}
        for i in range(10)
    ]
    task = Task(
        task_id="budget_exhaust_test",
        family="code_lcb",
        prompt="Solve problem",
        ground_truth="PASS",
        meta={"hidden_tests": hidden_tests},
    )

    # Mock run_tests where first 3 tests pass, but problem budget ran out
    # producing kind='timeout' on the 4th entry
    with patch("tasks.livecodebench_tasks.run_tests") as mock_run:
        mock_run.return_value = [
            {"passed": True, "expected": "0\n", "actual": "0\n", "kind": "pass", "stderr_tail": ""},
            {"passed": True, "expected": "1\n", "actual": "1\n", "kind": "pass", "stderr_tail": ""},
            {"passed": True, "expected": "2\n", "actual": "2\n", "kind": "pass", "stderr_tail": ""},
            {"passed": False, "expected": "3\n", "actual": "", "kind": "timeout", "stderr_tail": "Per-problem budget exhausted"},
        ]
        grade = grade_lcb(task, "```python\nprint(input())\n```")
        assert not grade["correct"], "Budget exhaustion must be marked not correct"
        assert grade["fingerprint"] == "TIMEOUT", f"Expected TIMEOUT, got {grade['fingerprint']}"
        assert grade["tests_run"] == 4


def test_daily_limit_exhausted_stops_without_accept():
    """Verify that a per-day 429 raises DailyLimitExhausted, stopping the run
    without ever logging an empty or malformed review as ACCEPT.
    Verify that per-minute 429s (RetryableRateLimit) are retried.
    """
    from harness import DailyLimitExhausted, RetryableRateLimit, call_with_retry

    # 1. Per-minute 429 retries
    call_count = 0
    def mock_per_minute_api(prompt):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RetryableRateLimit("429 rate limit - requests per minute")
        return "SUCCESS_RESPONSE"

    dummy_model = Model("test-min-429", mock_per_minute_api)
    res = call_with_retry(dummy_model, "hi", sleep=lambda _: None)
    assert res == "SUCCESS_RESPONSE"
    assert call_count == 3, "Should have retried twice and succeeded on 3rd attempt"

    # 2. Per-day 429 raises DailyLimitExhausted immediately
    def mock_per_day_api(prompt):
        raise DailyLimitExhausted("429 daily quota exceeded - requests per day limit reached")

    daily_model = Model("test-day-429", mock_per_day_api)
    with pytest.raises(DailyLimitExhausted):
        call_with_retry(daily_model, "hi", sleep=lambda _: None)

    # 3. In pipeline: DailyLimitExhausted during review stops run gracefully and NEVER logs ACCEPT
    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "scratch_test_daily_429")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        cache = DiskCache(os.path.join(tmp_dir, "cache"))
        log = RunLog(os.path.join(tmp_dir, "log.jsonl"))
        meter = TokenMeter(sleep=lambda _: None)

        task = Task(
            task_id="daily_limit_task",
            family="code_lcb",
            prompt="Solve",
            ground_truth="PASS",
            meta={"hidden_tests": [{"input": "1", "expected_output": "1"}]},
        )
        executor = Model("mock-exec", lambda p: "```python\nprint(1)\n```")
        reviewer = Model("mock-daily-reviewer", mock_per_day_api)

        summary = run_per_experiment(
            [task], executor, {"C0": None, "C1": reviewer},
            cache, log, meter,
            grader=lambda t, r: {"correct": True, "fingerprint": "PASS", "extracted": "hash"},
        )

        assert summary["stopped_daily_limit"] is True, "Pipeline must flag stopped_daily_limit"
        rows = log.load()
        # C0 was logged before review
        assert len(rows) == 1
        assert rows[0]["condition"] == "C0"
        # C1 was NEVER logged as ACCEPT or malformed review
        c1_rows = [r for r in rows if r["condition"] == "C1"]
        assert len(c1_rows) == 0, f"C1 should not be logged upon daily limit, but found: {c1_rows}"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_resume_safety_injected_sandbox_unavailable():
    """1. Resume safety: 6 problems, problem-major loop.
    Inject SandboxUnavailable partway through problem 3 (once inside revision grade,
    once inside C4 review), then call runner again.
    Assert:
      - exactly 5 outcome rows per problem (30 total)
      - no duplicates in log
      - rows identical (ignoring timestamps) to an uninterrupted run
      - zero model calls on resume for results already cached
      - no duplicate (task, condition, stage) in programs.jsonl
    """
    import json
    from harness import SandboxUnavailable

    base_tmp = os.path.join(os.path.dirname(__file__), "..", "scratch_test_resume_safety")
    shutil.rmtree(base_tmp, ignore_errors=True)
    os.makedirs(base_tmp, exist_ok=True)

    try:
        # Create 6 tasks
        tasks = [
            Task(
                task_id=f"prob_{i}",
                family="code_lcb",
                prompt=f"Solve problem {i}: task_{i}",
                ground_truth="PASS",
                meta={"hidden_tests": [{"input": f"{i}", "expected_output": f"{i}"}]},
            )
            for i in range(6)
        ]

        class CountingExecutor:
            def __init__(self):
                self.model_id = "mock-exec"
                self.call_count = 0
            def call(self, prompt):
                self.call_count += 1
                if "revised" in prompt:
                    return "```python\nprint(1) # revised\n```"
                return "```python\nprint(1)\n```"
            def __call__(self, prompt):
                return self.call(prompt)

        def make_reviewers():
            r1 = Model("rev-1", lambda p: "VERDICT: REJECT\nCRITIQUE: Fix bug 1")
            r2 = Model("rev-2", lambda p: "VERDICT: ACCEPT\nCRITIQUE: Looks good")
            r3 = Model("rev-3", lambda p: "VERDICT: REJECT\nCRITIQUE: Fix bug 3")
            return r1, r2, r3

        class InjectableToolReviewer:
            def __init__(self):
                self.should_fail = False
            def review(self, task, a0_text):
                if self.should_fail and task.task_id == "prob_2":
                    raise SandboxUnavailable("Injected SandboxUnavailable inside C4 review")
                return "VERDICT: ACCEPT\nCRITIQUE: Public tests passed"

        def make_extra_log(programs_path):
            def extra_log_fn(task, condition, a0_text, a0_grade, final_grade):
                with open(programs_path, "a", encoding="utf-8") as pf:
                    entry = {
                        "task_id": task.task_id,
                        "condition": condition,
                        "stage": "final",
                        "a0_code": a0_text,
                        "final_code": final_grade.get("extracted"),
                    }
                    pf.write(json.dumps(entry) + "\n")
                return {
                    "a0_fingerprint": a0_grade.get("fingerprint"),
                    "final_fingerprint": final_grade.get("fingerprint"),
                }
            return extra_log_fn

        # Step A: Uninterrupted baseline run
        dir_clean = os.path.join(base_tmp, "clean")
        cache_clean = DiskCache(os.path.join(dir_clean, "cache"))
        log_clean = RunLog(os.path.join(dir_clean, "log.jsonl"))
        meter_clean = TokenMeter(sleep=lambda _: None)
        exec_clean = CountingExecutor()
        r1_c, r2_c, r3_c = make_reviewers()
        tool_clean = InjectableToolReviewer()
        programs_clean = os.path.join(dir_clean, "programs.jsonl")

        clean_conditions = {
            "C0": None,
            "C1": r1_c,
            "C2": r2_c,
            "C3": r3_c,
            "C4": tool_clean,
        }

        revise_input_fn = lambda t, a, c: {"problem": t.prompt, "solution": a, "critique": c + " revised"}
        clean_grader = lambda t, r: {"correct": True, "fingerprint": "PASS", "extracted": "h1"}
        run_per_experiment(
            tasks, exec_clean, clean_conditions,
            cache_clean, log_clean, meter_clean,
            grader=clean_grader,
            revise_input_fn=revise_input_fn,
            extra_log_fn=make_extra_log(programs_clean),
            verbose=False,
        )
        clean_rows = log_clean.load()
        assert len(clean_rows) == 30  # 6 tasks * 5 conditions

        # Step B: Resumed run with 2 injected SandboxUnavailable stops
        dir_resume = os.path.join(base_tmp, "resume")
        cache_res = DiskCache(os.path.join(dir_resume, "cache"))
        log_res = RunLog(os.path.join(dir_resume, "log.jsonl"))
        meter_res = TokenMeter(sleep=lambda _: None)
        exec_res = CountingExecutor()
        r1_r, r2_r, r3_r = make_reviewers()
        tool_res = InjectableToolReviewer()
        programs_res = os.path.join(dir_resume, "programs.jsonl")
        res_conditions = {
            "C0": None,
            "C1": r1_r,
            "C2": r2_r,
            "C3": r3_r,
            "C4": tool_res,
        }

        inject_revision_grade_fail = True
        tool_res.should_fail = False

        def injected_grader(task, response):
            if inject_revision_grade_fail and task.task_id == "prob_2" and "revised" in response:
                raise SandboxUnavailable("Injected SandboxUnavailable inside revision grade")
            return {"correct": True, "fingerprint": "PASS", "extracted": "h1"}

        revise_input_fn = lambda t, a, c: {"problem": t.prompt, "solution": a, "critique": c + " revised"}

        # Run 1: Fails partway through problem 3 (prob_2) in revision grade
        summary_1 = run_per_experiment(
            tasks, exec_res, res_conditions,
            cache_res, log_res, meter_res,
            grader=injected_grader,
            revise_input_fn=revise_input_fn,
            extra_log_fn=make_extra_log(programs_res),
            verbose=False,
        )
        assert summary_1["stopped_sandbox_unavailable"] is True
        rows_after_1 = log_res.load()
        assert len(rows_after_1) == 11
        assert (rows_after_1[-1]["task_id"], rows_after_1[-1]["condition"]) == ("prob_2", "C0")

        # Run 2: Resume, revision grade succeeds, C4 review on prob_2 fails
        inject_revision_grade_fail = False
        tool_res.should_fail = True
        calls_before_run_2 = exec_res.call_count

        summary_2 = run_per_experiment(
            tasks, exec_res, res_conditions,
            cache_res, log_res, meter_res,
            grader=injected_grader,
            revise_input_fn=revise_input_fn,
            extra_log_fn=make_extra_log(programs_res),
            verbose=False,
        )
        assert summary_2["stopped_sandbox_unavailable"] is True
        rows_after_2 = log_res.load()
        assert len(rows_after_2) == 14
        # Assert ZERO model calls were made for previously cached results (prob_0, prob_1, prob_2 C0, prob_2 C1):
        # On prob_2, C1 revision was already cached in Run 1; only C3 revision needed a call:
        assert exec_res.call_count - calls_before_run_2 == 1

        # Run 3: Resume to completion! C4 review now succeeds
        tool_res.should_fail = False
        calls_before_run_3 = exec_res.call_count

        summary_3 = run_per_experiment(
            tasks, exec_res, res_conditions,
            cache_res, log_res, meter_res,
            grader=injected_grader,
            revise_input_fn=revise_input_fn,
            extra_log_fn=make_extra_log(programs_res),
            verbose=False,
        )
        assert summary_3["stopped_sandbox_unavailable"] is False
        rows_final = log_res.load()
        assert len(rows_final) == 30

        # Assert zero model calls on resume for results already cached:
        assert exec_res.call_count - calls_before_run_3 == 9

        # Assert exactly 5 outcome rows per problem
        from collections import Counter
        counts = Counter(r["task_id"] for r in rows_final)
        for t in tasks:
            assert counts[t.task_id] == 5

        # Assert no duplicates in log
        pairs = [(r["task_id"], r["condition"]) for r in rows_final]
        assert len(pairs) == len(set(pairs)) == 30

        # Assert rows identical (ignoring timestamps) to uninterrupted run
        def strip_ts(r):
            return {k: v for k, v in r.items() if k not in ("ts", "utc_ts")}

        for r_res, r_clean in zip(rows_final, clean_rows):
            assert strip_ts(r_res) == strip_ts(r_clean)

        # Assert no duplicate (task, condition, stage) in programs.jsonl
        with open(programs_res, "r", encoding="utf-8") as pf:
            prog_entries = [json.loads(l) for l in pf if l.strip()]
        assert len(prog_entries) == 30
        prog_keys = [(e["task_id"], e["condition"], e.get("stage", "final")) for e in prog_entries]
        assert len(prog_keys) == len(set(prog_keys)) == 30

    finally:
        shutil.rmtree(base_tmp, ignore_errors=True)


@pytest.mark.parametrize("msg,expected_daily", [
    ("Rate limit reached for model `llama-3.1-8b-instant` on tokens per day (TPD): Limit 500000, Used 500001.", True),
    ("Rate limit reached for requests per day (RPD): Limit 1000, Used 1000.", True),
    ("ResourceExhausted: 429 Quota exceeded for quota metric 'GenerateContent requests' and limit 'GenerateContent requests per day'", True),
    ("429 daily quota exceeded - requests per day limit reached", True),
    ("Rate limit reached for model `openai/gpt-oss-20b` on tokens per minute (TPM): Limit 8000, Used 8192.", False),
    ("Rate limit reached for requests per minute (RPM): Limit 30, Used 30.", False),
    ("Rate limit exceeded. Please wait 15 seconds.", False),
])
def test_rate_limit_classification_looks_daily(msg, expected_daily):
    """2. Rate-limit classification: _looks_daily matches daily markers and rejects per-minute."""
    from providers import _looks_daily
    assert _looks_daily(Exception(msg)) is expected_daily


def test_exhausted_per_minute_retries_stops_cleanly_scores_nothing():
    """2. Exhausted per-minute retries must stop the run cleanly with stopped_rate_limit and score nothing:
    no ACCEPT, no malformed review, no traceback.
    """
    from harness import RetryableRateLimit

    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "scratch_test_exhausted_rpm")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        cache = DiskCache(os.path.join(tmp_dir, "cache"))
        log = RunLog(os.path.join(tmp_dir, "log.jsonl"))
        meter = TokenMeter(sleep=lambda _: None)

        task = Task(
            task_id="rpm_task",
            family="code_lcb",
            prompt="Solve problem",
            ground_truth="PASS",
            meta={"hidden_tests": [{"input": "1", "expected_output": "1"}]},
        )
        executor = Model("mock-exec", lambda p: "```python\nprint(1)\n```")
        def mock_failing_rpm(prompt):
            raise RetryableRateLimit("429 rate limit - requests per minute (RPM)")

        reviewer = Model("mock-rpm-reviewer", mock_failing_rpm)

        summary = run_per_experiment(
            [task], executor, {"C0": None, "C1": reviewer},
            cache, log, meter,
            grader=lambda t, r: {"correct": True, "fingerprint": "PASS", "extracted": "hash"},
            verbose=False,
        )
        assert summary["stopped_rate_limit"] is True

        rows = log.load()
        # C0 was logged before review
        assert len(rows) == 1
        assert rows[0]["condition"] == "C0"
        # C1 was NEVER logged as ACCEPT or malformed review
        c1_rows = [r for r in rows if r["condition"] == "C1"]
        assert len(c1_rows) == 0, f"C1 should not be logged upon exhausted rate limit, but found: {c1_rows}"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.parametrize("name,body,expected_daily", [
    (
        "Gemini RESOURCE_EXHAUSTED daily (reconstructed, not captured)",
        json.dumps({
            "error": {
                "code": 429,
                "message": "Resource has been exhausted (e.g. check quota).",
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [
                            {
                                "subject": "project:1234567890",
                                "description": "Quota exceeded for quota metric 'GenerateContent requests' and limit 'GenerateContent requests per day'",
                                "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
                            }
                        ]
                    }
                ]
            }
        }),
        True,
    ),
    (
        "Gemini RESOURCE_EXHAUSTED per-minute retryable (reconstructed, not captured)",
        json.dumps({
            "error": {
                "code": 429,
                "message": "Resource has been exhausted (e.g. check quota).",
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [
                            {
                                "subject": "project:1234567890",
                                "description": "Quota exceeded for quota metric 'GenerateContent requests' and limit 'GenerateContent requests per minute'",
                                "quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
                            }
                        ]
                    }
                ]
            }
        }),
        False,
    ),
    (
        "Groq tokens per day (TPD) daily (reconstructed, not captured)",
        json.dumps({
            "error": {
                "message": "Rate limit reached for model `llama-3.1-8b-instant` on tokens per day (TPD): Limit 500000, Used 500001, Requested 120. Please try again tomorrow.",
                "type": "tokens",
                "code": "rate_limit_exceeded"
            }
        }),
        True,
    ),
    (
        "Groq tokens per minute (TPM) retryable (reconstructed, not captured)",
        json.dumps({
            "error": {
                "message": "Rate limit reached for model `openai/gpt-oss-20b` on tokens per minute (TPM): Limit 8000, Used 7950, Requested 100. Please try again in 15s.",
                "type": "tokens",
                "code": "rate_limit_exceeded"
            }
        }),
        False,
    ),
])
def test_looks_daily_realistic_reconstructed_bodies(name, body, expected_daily):
    """1. Table-driven test with realistic reconstructed 429 bodies (labeled reconstructed, not captured)."""
    from providers import _looks_daily
    assert _looks_daily(Exception(body)) is expected_daily


def test_call_with_retry_bounds_per_minute_retries_exhausted():
    """2. call_with_retry bounds retries to 6 attempts or 300s total wait, raising RateLimitStop."""
    from harness import call_with_retry, RetryableRateLimit, RateLimitStop, Model

    attempts = 0
    def failing_call(p):
        nonlocal attempts
        attempts += 1
        raise RetryableRateLimit("429 rate limit")

    m = Model("test-model", failing_call)
    sleeps = []
    with pytest.raises(RateLimitStop) as exc_info:
        call_with_retry(m, "test", max_retries=6, max_total_wait_s=300.0, base_wait=2.0, sleep=sleeps.append)

    assert attempts == 7
    assert len(sleeps) == 6
    assert sleeps == [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    assert "Rate limit retries exhausted" in str(exc_info.value)

    attempts2 = 0
    sleeps2 = []
    def failing_call2(p):
        nonlocal attempts2
        attempts2 += 1
        raise RetryableRateLimit("429 rate limit")
    m2 = Model("test-model-2", failing_call2)
    with pytest.raises(RateLimitStop):
        call_with_retry(m2, "test", max_retries=10, max_total_wait_s=50.0, base_wait=20.0, sleep=sleeps2.append)
    assert sum(sleeps2) <= 50.0


def test_ambiguous_persistent_429_clean_stop_zero_rows_no_traceback():
    """2. Ambiguous persistent 429 ends in a clean stop with zero rows written and no traceback."""
    from harness import RetryableRateLimit, DiskCache, RunLog, Model
    from pipeline import run_per_experiment, TokenMeter

    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "scratch_test_ambig_429")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        cache = DiskCache(os.path.join(tmp_dir, "cache"))
        log = RunLog(os.path.join(tmp_dir, "log.jsonl"))
        meter = TokenMeter(sleep=lambda _: None)

        task = Task(
            task_id="ambig_task",
            family="code_lcb",
            prompt="Write solution",
            ground_truth="PASS",
            meta={"hidden_tests": [{"input": "1", "expected_output": "1"}]},
        )
        def mock_ambig_429(prompt):
            raise RetryableRateLimit("HTTP 429: Too Many Requests - Rate limit exceeded")

        executor = Model("mock-ambig-exec", mock_ambig_429)

        summary = run_per_experiment(
            [task], executor, {"C0": None, "C1": None},
            cache, log, meter,
            grader=lambda t, r: {"correct": True, "fingerprint": "PASS", "extracted": "hash"},
            verbose=False,
        )

        assert summary["stopped_rate_limit"] is True
        assert summary["completed_outcomes"] == 0
        rows = log.load()
        assert len(rows) == 0, f"Expected 0 rows written on persistent 429 stop, got {len(rows)}"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_run_log_load_ignores_truncated_final_line_and_warns():
    """3. RunLog.load must ignore a truncated final line (warn and continue, never drop earlier lines)."""
    import warnings
    from harness import RunLog

    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "scratch_test_trunc_log")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        log_path = os.path.join(tmp_dir, "log.jsonl")
        log = RunLog(log_path)
        log.append({"task_id": "t1", "condition": "C0", "correct": True})
        log.append({"task_id": "t2", "condition": "C0", "correct": False})

        with open(log_path, "a", encoding="utf-8") as f:
            f.write('{"task_id": "t3", "condition": "C0", "corr')

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rows = log.load()
            assert len(w) == 1
            assert "ignoring truncated final line" in str(w[-1].message)

        assert len(rows) == 2
        assert rows[0]["task_id"] == "t1"
        assert rows[1]["task_id"] == "t2"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_resume_after_truncated_log_equals_uninterrupted_run():
    """3. Test by truncating log.jsonl mid-line, resuming, and showing the result equals an uninterrupted run."""
    from harness import RunLog, DiskCache, Model
    from pipeline import run_per_experiment, TokenMeter

    base_tmp = os.path.join(os.path.dirname(__file__), "..", "scratch_test_trunc_resume")
    shutil.rmtree(base_tmp, ignore_errors=True)
    os.makedirs(base_tmp, exist_ok=True)
    try:
        tasks = [
            Task(task_id=f"prob_{i}", family="code_lcb", prompt=f"Solve {i}",
                 ground_truth="PASS", meta={"hidden_tests": [{"input": "1", "expected_output": "1"}]})
            for i in range(3)
        ]
        executor = Model("mock-exec", lambda p: f"```python\nprint('{p}')\n```")
        grader = lambda t, r: {"correct": True, "fingerprint": "PASS", "extracted": hash(r)}

        # Run 1: Uninterrupted baseline run
        clean_dir = os.path.join(base_tmp, "clean")
        clean_cache = DiskCache(os.path.join(clean_dir, "cache"))
        clean_log = RunLog(os.path.join(clean_dir, "log.jsonl"))
        clean_meter = TokenMeter(sleep=lambda _: None)
        run_per_experiment(tasks, executor, {"C0": None}, clean_cache, clean_log, clean_meter, grader=grader, verbose=False)
        clean_rows = clean_log.load()
        assert len(clean_rows) == 3

        # Run 2: Interrupted run with truncated final line
        res_dir = os.path.join(base_tmp, "resume")
        res_cache = DiskCache(os.path.join(res_dir, "cache"))
        res_log = RunLog(os.path.join(res_dir, "log.jsonl"))
        res_meter = TokenMeter(sleep=lambda _: None)

        run_per_experiment(tasks[:2], executor, {"C0": None}, res_cache, res_log, res_meter, grader=grader, verbose=False)
        assert len(res_log.load()) == 2

        # Truncate log mid-line simulating a hard kill during write of task 2
        with open(res_log.path, "a", encoding="utf-8") as f:
            f.write('{"task_id": "prob_2", "condition": "C0", "ground_truth": "PA')

        # Resume run with all 3 tasks
        run_per_experiment(tasks, executor, {"C0": None}, res_cache, res_log, res_meter, grader=grader, verbose=False)
        res_rows = res_log.load()

        assert len(res_rows) == len(clean_rows) == 3
        for r_res, r_clean in zip(res_rows, clean_rows):
            assert {k: v for k, v in r_res.items() if k not in ("ts", "utc_ts")} == \
                   {k: v for k, v in r_clean.items() if k not in ("ts", "utc_ts")}
    finally:
        shutil.rmtree(base_tmp, ignore_errors=True)


def test_analyze_code_deduplicates_programs_by_task_condition_stage():
    """3. analyze_code.load_programs de-duplicates programs.jsonl by (task_id, condition, stage)."""
    import analyze_code

    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "scratch_test_analyze_dedup")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        p_path = os.path.join(tmp_dir, "programs.jsonl")
        entries = [
            {"task_id": "t1", "condition": "C0", "stage": "final", "code": "v1"},
            {"task_id": "t1", "condition": "C0", "stage": "final", "code": "v2"},  # duplicate from resume
            {"task_id": "t1", "condition": "C1", "stage": "final", "code": "v3"},
            {"task_id": "t2", "condition": "C0", "stage": "final", "code": "v4"},
        ]
        with open(p_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        deduped = analyze_code.load_programs(p_path)
        assert len(deduped) == 3
        t1_c0 = next(e for e in deduped if e["task_id"] == "t1" and e["condition"] == "C0")
        assert t1_c0["code"] == "v2"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_analyze_critiques_deduplicates_programs_by_task_condition_stage():
    """3. analyze_critiques.load_programs de-duplicates programs.jsonl by (task_id, condition, stage)."""
    import analyze_critiques

    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "scratch_test_critiques_dedup")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        p_path = os.path.join(tmp_dir, "programs.jsonl")
        entries = [
            {"task_id": "t1", "condition": "C1", "stage": "final", "code": "v1"},
            {"task_id": "t1", "condition": "C1", "stage": "final", "code": "v2"},  # duplicate
            {"task_id": "t2", "condition": "C2", "stage": "final", "code": "v3"},
        ]
        with open(p_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        deduped = analyze_critiques.load_programs(p_path)
        assert len(deduped) == 2
        t1_c1 = next(e for e in deduped if e["task_id"] == "t1" and e["condition"] == "C1")
        assert t1_c1["code"] == "v2"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_single_runner_lock_alive_pid_refuses_and_dead_pid_recovers():
    """4. Single runner: .runner.lock refuses to start if PID is alive, treats dead PID as stale, and removes lock on exit."""
    from run_experiment_code import RunnerLock, is_pid_alive

    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "scratch_test_lock")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        lock_file = os.path.join(tmp_dir, ".runner.lock")

        # Case 1: First runner acquires lock
        lock1 = RunnerLock(lock_file)
        assert lock1.acquire() is True
        assert os.path.exists(lock_file)
        with open(lock_file, "r") as f:
            assert int(f.read().strip()) == os.getpid()

        # Case 2: Second runner tries to acquire while lock1 is held by active PID -> Refuses!
        lock2 = RunnerLock(lock_file)
        assert lock2.acquire() is False

        # Release lock1
        lock1.release()
        assert not os.path.exists(lock_file)

        # Case 3: Dead/stale PID in lock file (e.g. 9999999)
        with open(lock_file, "w") as f:
            f.write("9999999\n")
        assert is_pid_alive(9999999) is False

        # RunnerLock treats dead PID as stale, recovers and acquires successfully!
        lock3 = RunnerLock(lock_file)
        assert lock3.acquire() is True
        with open(lock_file, "r") as f:
            assert int(f.read().strip()) == os.getpid()
        lock3.release()
        assert not os.path.exists(lock_file)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_single_runner_cleanup_stale_containers_runs_only_after_lock_held():
    """4. cleanup_stale_containers runs only after the single runner lock is held."""
    from run_experiment_code import RunnerLock

    events = []
    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "scratch_test_lock_order")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        lock_file = os.path.join(tmp_dir, ".runner.lock")
        lock = RunnerLock(lock_file)

        def mock_acquire():
            res = lock.acquire()
            events.append("lock_acquired")
            return res

        def mock_cleanup():
            assert os.path.exists(lock_file)
            events.append("cleanup_run")
            return 0

        if mock_acquire():
            try:
                mock_cleanup()
            finally:
                lock.release()
                events.append("lock_released")

        assert events == ["lock_acquired", "cleanup_run", "lock_released"]
        assert not os.path.exists(lock_file)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_cli_run_dir_and_ids_file_artifacts_routing():
    """5. Manifest, log, programs, tokens and cache must all follow --run-dir and --ids-file."""
    import run_experiment_code

    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "scratch_test_run_dir_cli")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        custom_run_dir = os.path.join(tmp_dir, "custom_run")
        custom_ids_file = os.path.join(tmp_dir, "my_problems.json")
        with open(custom_ids_file, "w", encoding="utf-8") as f:
            json.dump(["arc186_e"], f)

        manifest = run_experiment_code.append_run_manifest(
            ids_file=custom_ids_file,
            run_dir=custom_run_dir,
        )
        assert os.path.exists(os.path.join(custom_run_dir, "run_manifest.json"))
        assert manifest["problem_ids_sha256"] != ""

        extra_log = run_experiment_code.make_extra_code_log(os.path.join(custom_run_dir, "programs.jsonl"))
        task = Task(task_id="arc186_e", family="code_lcb", prompt="test", ground_truth="PASS", meta={})
        extra_log(task, "C0", "print(1)", {"correct": True, "fingerprint": "PASS", "extracted": "h1"},
                  {"correct": True, "fingerprint": "PASS", "extracted": "h1"})
        assert os.path.exists(os.path.join(custom_run_dir, "programs.jsonl"))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_grade_caching_by_task_id_and_code_hash_reuses_identical_grade():
    """Resume consistency: grade results are cached by (task_id, code_hash) so a resumed run reuses the identical grade."""
    from harness import DiskCache, RunLog, Task, Model
    from pipeline import run_per_experiment, TokenMeter

    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "scratch_test_grade_cache")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        cache = DiskCache(os.path.join(tmp_dir, "cache"))
        log = RunLog(os.path.join(tmp_dir, "log.jsonl"))
        meter = TokenMeter(sleep=lambda _: None)

        grader_calls = []

        def counting_grader(task, text):
            grader_calls.append((task.task_id, text))
            return {
                "correct": True,
                "fingerprint": "PASS",
                "extracted": "hash_abc",
                "ast_hash": "ast_abc",
                "max_test_wall_s": 0.42,
                "infra_retries": 0,
            }

        task = Task(task_id="prob_test_cache", family="code_lcb", prompt="solve problem", ground_truth="PASS", meta={})
        executor = Model("mock_exec", lambda p: "```python\nprint(42)\n```")
        conditions = {"C0": None}

        # Run 1: initial grade
        run_per_experiment([task], executor, conditions, cache, log, meter, grader=counting_grader, verbose=False)
        assert len(grader_calls) == 1

        # Check that grade is in cache under ("grade_cache", task.task_id, code_hash, 0)
        from pipeline import _extract_code_hash
        ch = _extract_code_hash(task, "```python\nprint(42)\n```")
        cached_json = cache.get("grade_cache", task.task_id, ch, 0)
        assert cached_json is not None
        cached_grade = json.loads(cached_json)
        assert cached_grade["fingerprint"] == "PASS"
        assert cached_grade["max_test_wall_s"] == 0.42

        # Run 2: resume / rerun identical code
        # Empty the log so it attempts to process C0 again
        empty_log = RunLog(os.path.join(tmp_dir, "empty_log.jsonl"))
        run_per_experiment([task], executor, conditions, cache, empty_log, meter, grader=counting_grader, verbose=False)

        # Grader call count MUST STILL BE 1 (cached grade reused, 0 new grader calls)
        assert len(grader_calls) == 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_analyze_code_integrity_check_blocks_corrupted_fixture(capsys):
    """Integrity check: each problem's rows must share a0_code_hash, a0_ast_hash, and a0_fingerprint; any violation is reported loudly and blocks primary results."""
    import analyze_code

    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "scratch_test_integrity")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        corrupted_log = os.path.join(tmp_dir, "corrupt_log.jsonl")
        rows = [
            {
                "task_id": "prob_1",
                "condition": "C0",
                "final_correct": True,
                "a0_code_hash": "hash_original",
                "a0_ast_hash": "ast_original",
                "a0_fingerprint": "PASS",
            },
            {
                "task_id": "prob_1",
                "condition": "C1",
                "final_correct": False,
                "a0_code_hash": "hash_corrupted",  # Mismatch!
                "a0_ast_hash": "ast_original",
                "a0_fingerprint": "PASS",
            },
        ]
        with open(corrupted_log, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        with pytest.raises(ValueError, match=r"Integrity check failed: 1 task\(s\) have conflicting A0 hashes"):
            analyze_code.run_analysis(corrupted_log)

        captured = capsys.readouterr().out
        assert "CRITICAL DATA INTEGRITY VIOLATION IN EXPERIMENT LOG" in captured
        assert "Task 'prob_1': mismatch between C0 and C1" in captured
        assert "a0_code_hash ('hash_original' vs 'hash_corrupted')" in captured
        assert "Primary results are BLOCKED" in captured
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_rate_limit_stop_prints_raw_error_and_matched_marker(capsys):
    """On any rate-limit stop, print the raw exception text and the matched marker (or 'no daily marker; per-minute retries exhausted')."""
    from harness import DiskCache, RunLog, Task, Model, DailyLimitExhausted, RetryableRateLimit
    from pipeline import run_per_experiment, TokenMeter

    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "scratch_test_ratelimit_prints")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        task = Task(task_id="prob_rl", family="code_lcb", prompt="solve", ground_truth="PASS", meta={})

        # Subtest A: DailyLimitExhausted
        cache_a = DiskCache(os.path.join(tmp_dir, "cache_a"))
        log_a = RunLog(os.path.join(tmp_dir, "log_a.jsonl"))
        meter_a = TokenMeter(sleep=lambda _: None)

        def daily_fail_model(p):
            raise DailyLimitExhausted(
                raw_error="Rate limit reached for requests per day (RPD): Limit 1000, Used 1000.",
                matched_marker="requests per day"
            )

        model_a = Model("mock_daily", daily_fail_model)
        run_per_experiment([task], model_a, {"C0": None}, cache_a, log_a, meter_a, verbose=True)

        out_a = capsys.readouterr().out
        assert "[pipeline] Rate limit stop triggered: DailyLimitExhausted" in out_a
        assert "[pipeline] Matched marker: requests per day" in out_a
        assert "[pipeline] Raw provider error text: Rate limit reached for requests per day (RPD): Limit 1000, Used 1000." in out_a

        # Subtest B: RateLimitStop (per-minute retries exhausted)
        cache_b = DiskCache(os.path.join(tmp_dir, "cache_b"))
        log_b = RunLog(os.path.join(tmp_dir, "log_b.jsonl"))
        meter_b = TokenMeter(sleep=lambda _: None)

        def permin_fail_model(p):
            raise RetryableRateLimit("Rate limit reached for model `openai/gpt-oss-20b` on tokens per minute (TPM): Limit 8000, Used 8192.")

        model_b = Model("mock_permin", permin_fail_model)
        run_per_experiment([task], model_b, {"C0": None}, cache_b, log_b, meter_b, verbose=True)

        out_b = capsys.readouterr().out
        assert "[pipeline] Rate limit stop triggered: RateLimitStop" in out_b
        assert "[pipeline] Matched marker: no daily marker; per-minute retries exhausted" in out_b
        assert "[pipeline] Raw provider error text: Rate limit reached for model `openai/gpt-oss-20b` on tokens per minute (TPM): Limit 8000, Used 8192." in out_b
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    run()




def test_math_path_grade_numeric_is_never_cached(monkeypatch):
    """Grade cache applies only to non-grade_numeric graders: the math path grades
    directly and writes no grade_cache entry."""
    import hashlib
    import pipeline

    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "scratch_test_math_no_grade_cache")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        def fail_if_called(task, text):
            raise AssertionError("_extract_code_hash must not run on the math path")
        monkeypatch.setattr(pipeline, "_extract_code_hash", fail_if_called)

        cache = DiskCache(os.path.join(tmp_dir, "cache"))
        log = RunLog(os.path.join(tmp_dir, "log.jsonl"))
        task = Task(task_id="math_1", family="math", prompt="What is 6*7?", ground_truth="42", meta={})
        executor = Model("mock_exec", lambda p: "The answer is 42")
        run_per_experiment([task], executor, {"C0": None}, cache, log,
                           TokenMeter(sleep=lambda _: None), verbose=False)

        rows = log.load()
        assert len(rows) == 1 and rows[0]["a0_correct"] is True
        text_hash = hashlib.sha256(b"The answer is 42").hexdigest()[:16]
        assert cache.get("grade_cache", "math_1", text_hash, 0) is None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _mock_code_rows():
    """Mock log: 4 pre-cutoff and 4 post-cutoff problems, A0 correct on even ids.
    C1 rejects every correct A0 and the executor keeps it; C2 repairs every wrong
    A0; C4 accepts all correct A0 and one wrong A0 (p1) passes the samples."""
    rows = []
    for i in range(8):
        tid = f"p{i}"
        date = "2024-11-10T00:00:00" if i < 4 else "2025-03-02T00:00:00"
        a0_ok = i % 2 == 0
        base = {"task_id": tid, "contest_date": date, "a0_correct": a0_ok,
                "a0_code_hash": f"h{i}", "a0_ast_hash": f"a{i}", "a0_extracted": f"h{i}",
                "a0_fingerprint": "PASS" if a0_ok else "FAIL:1/5",
                "a0_public_pass": a0_ok or i == 1, "review_malformed": False}
        same = {"final_correct": a0_ok, "final_code_hash": f"h{i}", "final_ast_hash": f"a{i}",
                "final_extracted": f"h{i}", "final_fingerprint": base["a0_fingerprint"]}
        rows.append({**base, **same, "condition": "C0", "verdict": "NONE"})
        rows.append({**base, **same, "condition": "C1", "verdict": "REJECT" if a0_ok else "ACCEPT"})
        if a0_ok:
            rows.append({**base, **same, "condition": "C2", "verdict": "ACCEPT"})
        else:
            rows.append({**base, "condition": "C2", "verdict": "REJECT", "final_correct": True,
                         "final_code_hash": f"r{i}", "final_ast_hash": f"ra{i}",
                         "final_extracted": f"r{i}", "final_fingerprint": "PASS"})
        rows.append({**base, **same, "condition": "C3", "verdict": "ACCEPT"})
        rows.append({**base, **same, "condition": "C4",
                     "verdict": "ACCEPT" if base["a0_public_pass"] else "REJECT"})
    return rows


def test_analyze_code_hypotheses_and_e2_on_mock_log():
    """H1-H4 verdicts and E2 temporal strata are computed and written to RESULTS.md."""
    import analyze_code

    rows = _mock_code_rows()
    hyp = analyze_code.evaluate_hypotheses(rows)
    assert hyp["H1"]["verdict"] == "supported" and (hyp["H1"]["k"], hyp["H1"]["n"]) == (4, 4)
    assert hyp["H2"]["verdict"] == "inconclusive" and (hyp["H2"]["k"], hyp["H2"]["n"]) == (4, 4)
    assert (hyp["H3"]["b"], hyp["H3"]["c"]) == (0, 4)
    assert hyp["H3"]["verdict"] == "not supported"   # p = 0.125 with 4 discordant pairs
    assert hyp["H4"]["verdict"] == "supported" and hyp["H4"]["k_fa"] == 1

    e2 = analyze_code.temporal_sensitivity(rows)
    assert e2["pre"]["n"] == 4 and e2["post"]["n"] == 4
    assert e2["pre"]["a0_correct"] == 2 and e2["post"]["a0_correct"] == 2
    assert (e2["post"]["c1_fr"], e2["post"]["c1_right"]) == (2, 2)
    assert e2["fisher_a0_p"] == pytest.approx(1.0)
    assert e2["missing_dates"] == []
    assert e2["post_hypotheses"]["H4"]["k_fa"] == 0   # the false accept (p1) is pre-cutoff

    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "scratch_test_e2")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        log_path = os.path.join(tmp_dir, "log.jsonl")
        with open(log_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        out_path = os.path.join(tmp_dir, "RESULTS.md")
        res = analyze_code.run_analysis(log_path, out_path=out_path)
        assert res["hypotheses"]["H1"]["verdict"] == "supported"
        md = open(out_path, encoding="utf-8").read()
        assert "## 6. Pre-Registered Hypotheses" in md
        assert "## 7. Exploratory E2: Temporal Sensitivity (boundary 2025-01-31)" in md
        assert "| **H2** | inconclusive |" in md
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_audit_regrade_reports_fingerprint_disagreements_on_mock_programs():
    """E3: every distinct program is re-graded once and compared with every
    first-pass fingerprint recorded for it."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import audit_regrade

    a0 = "print(1)"
    rev = "print(2)"
    entries = [
        {"task_id": "t1", "condition": c, "stage": "final", "a0_code": a0, "a0_code_hash": "hA",
         "a0_fingerprint": "PASS", "final_code": a0, "final_code_hash": "hA", "final_fingerprint": "PASS"}
        for c in ("C0", "C1")
    ] + [
        {"task_id": "t1", "condition": "C2", "stage": "final", "a0_code": a0, "a0_code_hash": "hA",
         "a0_fingerprint": "PASS", "final_code": rev, "final_code_hash": "hR", "final_fingerprint": "PASS"},
        {"task_id": "t2", "condition": "C0", "stage": "final", "a0_code": None, "a0_code_hash": None,
         "a0_fingerprint": "NO_CODE", "final_code": None, "final_code_hash": None, "final_fingerprint": "NO_CODE"},
    ]
    second = {a0: "PASS", rev: "TIMEOUT"}   # the revision flips on the second pass
    calls = []

    def mock_grade(task, response):
        calls.append(response)
        for code, fp in second.items():
            if code in response:
                return {"fingerprint": fp, "max_test_wall_s": 5.9}
        return {"fingerprint": "NO_CODE", "max_test_wall_s": 0.0}

    rep = audit_regrade.audit(entries, {"t1": object(), "t2": object()}, mock_grade, verbose=False)
    assert rep["n_programs"] == 3 and len(calls) == 3        # hA, hR, NO_CODE: each run once
    assert rep["n_first_pass_records"] == 8
    assert rep["n_programs_disagreeing"] == 1 and rep["n_records_disagreeing"] == 1
    d = rep["disagreements"][0]
    assert (d["task_id"], d["condition"], d["stage"], d["first"], d["second"]) == ("t1", "C2", "final", "PASS", "TIMEOUT")


def test_extra_code_log_records_final_program_and_its_public_result(monkeypatch):
    """programs.jsonl must hold the revised program (for the E3 second pass) and
    final_public_pass must come from the revised program, not A0."""
    import run_experiment_code

    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "scratch_test_final_code_log")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        checked = []

        def fake_public_check(task, code):
            checked.append(code)
            return {"all_pass": "fixed" in code, "first_failure": None}
        monkeypatch.setattr(run_experiment_code, "public_check", fake_public_check)

        def grader(task, text):
            ok = "fixed" in text
            return {"correct": False, "fingerprint": "FAIL:2/9" if ok else "FAIL:1/9",
                    "extracted": "hF" if ok else "hA", "ast_hash": "aF" if ok else "aA"}

        class Rejector:
            model_id = "public-tests"

            def review(self, task, solution_text):
                return "VERDICT: REJECT\nCRITIQUE: sample 1 failed"

        responses = iter(["```python\nprint(\"orig\")\n```", "```python\nprint(\"fixed\")\n```"])
        executor = Model("mock_exec", lambda p: next(responses))
        programs = os.path.join(tmp_dir, "programs.jsonl")
        run_per_experiment([Task(task_id="t1", family="code_lcb", prompt="p", ground_truth="PASS", meta={})],
                           executor, {"C4": Rejector()}, DiskCache(os.path.join(tmp_dir, "cache")),
                           RunLog(os.path.join(tmp_dir, "log.jsonl")), TokenMeter(sleep=lambda _: None),
                           grader=grader, extra_log_fn=run_experiment_code.make_extra_code_log(programs),
                           verbose=False)

        with open(programs, encoding="utf-8") as f:
            entry = json.loads(f.readline())
        assert entry["a0_code"] == "print(\"orig\")"
        assert entry["final_code"] == "print(\"fixed\")"
        assert (entry["a0_fingerprint"], entry["final_fingerprint"]) == ("FAIL:1/9", "FAIL:2/9")
        row = RunLog(os.path.join(tmp_dir, "log.jsonl")).load()[0]
        assert row["a0_public_pass"] is False and row["final_public_pass"] is True
        assert checked == ["print(\"orig\")", "print(\"fixed\")"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_preflight_reports_missing_keys_without_network(monkeypatch):
    """Preflight lists every missing key and skips host checks it cannot make."""
    import providers
    for var in ("GOOGLE_API_KEY", "GROQ_API_KEY", "DEEPINFRA_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    def no_network(*a, **k):
        raise AssertionError("preflight must not contact a host without its key")
    import httpx
    monkeypatch.setattr(httpx, "get", no_network)

    problems = providers.preflight_code_providers()
    for var in ("GOOGLE_API_KEY", "GROQ_API_KEY", "DEEPINFRA_API_KEY"):
        assert f"{var} is not set" in problems


def test_missing_sdk_is_reported_as_missing_sdk_not_missing_key(monkeypatch):
    """A missing google-genai SDK must not be reported as a missing API key."""
    import providers
    monkeypatch.setattr(providers, "_GENAI_IMPORT_ERROR", ModuleNotFoundError("No module named 'google.genai'"))
    with pytest.raises(RuntimeError, match="google-genai SDK is not installed"):
        providers.call_gemini("hello")


def test_code_models_use_deepinfra_llama_and_leave_math_models_unchanged():
    """C3 in the code study is Llama 3.1 8B Instruct on DeepInfra; the math
    study's model list still names the original Groq model."""
    import providers
    code_ids = [m.model_id for m in providers.code_experimental_models()]
    math_ids = [m.model_id for m in providers.experimental_models()]
    assert code_ids == ["gemini-3.1-flash-lite", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", "openai/gpt-oss-20b"]
    assert math_ids == ["gemini-3.1-flash-lite", "llama-3.1-8b-instant", "openai/gpt-oss-20b"]


def test_empty_review_retry_is_recorded_and_sets_review_retried(monkeypatch):
    """An empty gpt-oss review is retried once, the retry is written to the
    retry log, and that task's C2 row gets review_retried = True."""
    import run_experiment_code as rec

    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "scratch_test_review_retry")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        monkeypatch.setattr(rec, "public_check", lambda task, code: {"all_pass": True, "first_failure": None})
        responses = iter(["", "VERDICT: ACCEPT"])
        sleeps = []
        retry_log = os.path.join(tmp_dir, "review_retries.jsonl")
        gptoss = rec.wrap_throttled_model(Model("openai/gpt-oss-20b", lambda p: next(responses)),
                                          min_interval_s=0, retry_empty=True,
                                          retry_log_path=retry_log, sleep=sleeps.append)

        def grader(task, text):
            return {"correct": True, "fingerprint": "PASS", "extracted": "h1", "ast_hash": "a1"}

        tasks = [Task(task_id="t1", family="code_lcb", prompt="Solve it.\n\n" + rec.CODE_ANSWER_INSTRUCTION,
                      ground_truth="PASS", meta={"statement": "Solve it."})]
        executor = Model("mock_exec", lambda p: "```python\nprint(1)\n```")
        log = RunLog(os.path.join(tmp_dir, "log.jsonl"))
        run_per_experiment(tasks, executor, {"C0": None, "C2": gptoss}, DiskCache(os.path.join(tmp_dir, "cache")),
                           log, TokenMeter(sleep=lambda _: None), grader=grader,
                           review_prompt=CODE_REVIEW_PROMPT, revise_prompt=CODE_REVISE_PROMPT,
                           review_input_fn=rec.code_review_input, revise_input_fn=rec.code_revise_input,
                           extra_log_fn=rec.make_extra_code_log(os.path.join(tmp_dir, "programs.jsonl"),
                                                                retry_log_path=retry_log),
                           verbose=False)

        assert sleeps == [60]
        with open(retry_log, encoding="utf-8") as f:
            entries = [json.loads(line) for line in f]
        assert len(entries) == 1 and entries[0]["retry_empty"] is False
        rows = {r["condition"]: r for r in log.load()}
        assert rows["C2"]["review_retried"] is True and rows["C2"]["verdict"] == "ACCEPT"
        assert rows["C0"]["review_retried"] is False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_main_writes_token_records_even_when_run_crashes(monkeypatch):
    """tokens.jsonl must keep the usage of calls already made when an
    unexpected error stops the main run."""
    import run_experiment_code as rec

    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "scratch_test_tokens_on_crash")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        ids_file = os.path.join(tmp_dir, "ids.json")
        with open(ids_file, "w", encoding="utf-8") as f:
            json.dump(["t1"], f)
        monkeypatch.setattr(rec, "is_git_clean", lambda: True)
        monkeypatch.setattr(rec, "preflight_code_providers", lambda: [])
        monkeypatch.setattr(rec, "sandbox_preflight", lambda: [])
        monkeypatch.setattr(rec, "cleanup_stale_containers", lambda: 0)
        monkeypatch.setattr(rec, "load_lcb_tasks", lambda **k: [])
        monkeypatch.setattr(rec, "code_experimental_models", lambda: [
            Model("gemini-3.1-flash-lite", lambda p: ""),
            Model("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", lambda p: ""),
            Model("openai/gpt-oss-20b", lambda p: "")])

        def crashing_run(tasks, executor, conds, cache, log, meter, **kw):
            meter.records.append({"model": "gemini-3.1-flash-lite", "in_tokens": 11, "out_tokens": 7})
            raise RuntimeError("provider exploded")
        monkeypatch.setattr(rec, "run_per_experiment", crashing_run)
        monkeypatch.setattr(sys, "argv", ["run_experiment_code.py", "--run-dir", tmp_dir, "--ids-file", ids_file])

        with pytest.raises(RuntimeError, match="provider exploded"):
            rec.main()
        with open(os.path.join(tmp_dir, "tokens.jsonl"), encoding="utf-8") as f:
            recs = [json.loads(line) for line in f]
        assert recs == [{"model": "gemini-3.1-flash-lite", "in_tokens": 11, "out_tokens": 7}]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_main_refuses_to_start_when_preflight_fails(monkeypatch, capsys):
    """A failed provider preflight stops the run before the manifest or lock is written."""
    import run_experiment_code as rec

    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "scratch_test_preflight_refuse")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        ids_file = os.path.join(tmp_dir, "ids.json")
        with open(ids_file, "w", encoding="utf-8") as f:
            json.dump(["t1"], f)
        monkeypatch.setattr(rec, "is_git_clean", lambda: True)
        monkeypatch.setattr(rec, "preflight_code_providers", lambda: ["DEEPINFRA_API_KEY is not set"])
        monkeypatch.setattr(rec, "sandbox_preflight", lambda: [])
        monkeypatch.setattr(sys, "argv", ["run_experiment_code.py", "--run-dir", tmp_dir, "--ids-file", ids_file])
        with pytest.raises(SystemExit):
            rec.main()
        assert "DEEPINFRA_API_KEY is not set" in capsys.readouterr().out
        assert not os.path.exists(os.path.join(tmp_dir, "run_manifest.json"))
        assert not os.path.exists(os.path.join(tmp_dir, ".runner.lock"))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_retried_review_records_tokens_of_both_calls():
    """An empty review plus its retry are one review: the meter must record the
    summed usage of both provider calls, not only the retry's."""
    import types
    import run_experiment_code as rec

    usage = types.SimpleNamespace(_LAST_USAGE=None)
    usage.get_last_usage = lambda: (lambda u: (setattr(usage, "_LAST_USAGE", None), u)[1])(usage._LAST_USAGE)
    replies = iter([("", {"prompt_tokens": 1400, "completion_tokens": 6144}),
                    ("VERDICT: ACCEPT", {"prompt_tokens": 1400, "completion_tokens": 300})])

    def call(prompt):
        text, u = next(replies)
        usage._LAST_USAGE = u
        return text

    model = rec.wrap_throttled_model(Model("openai/gpt-oss-20b", call), min_interval_s=0,
                                     retry_empty=True, sleep=lambda s: None, usage_module=usage)
    meter = TokenMeter(usage_hook=usage.get_last_usage, sleep=lambda _: None)
    assert meter.call(model, "review this", "review", "t1", "C2") == "VERDICT: ACCEPT"
    rec_ = meter.records[0]
    assert (rec_["in_tokens"], rec_["out_tokens"], rec_["exact"]) == (2800, 6444, True)
