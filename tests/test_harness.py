"""Harness tests using a mock model — no API keys involved.

Run:  python3 tests/test_harness.py

Verifies the properties the real runs depend on:
  1. A full run makes exactly the expected number of API calls and log rows.
  2. Re-running makes ZERO new API calls (cache resume) and no duplicate logs.
  3. Per-minute rate limits are retried with exponential backoff and succeed.
  4. Daily-limit exhaustion stops the run gracefully; a later re-run resumes
     from exactly where it stopped and completes the remainder.
  5. Log records carry every field the analysis needs.
"""

import os, shutil, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core import Task, NUMERIC_ANSWER_INSTRUCTION
from harness import (Model, DiskCache, RunLog, run_single_agent,
                     call_with_retry, RetryableRateLimit, DailyLimitExhausted)

TMP = "/tmp/harness_test"


def fresh_dirs():
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP)
    return DiskCache(os.path.join(TMP, "cache")), RunLog(os.path.join(TMP, "log.jsonl"))


def make_tasks(n=4):
    return [Task(f"t{i:02d}", "math", f"What is {i} + {i}?" + NUMERIC_ANSWER_INSTRUCTION,
                 float(2 * i), meta={"template": "toy", "difficulty": 1, "round_to": 2})
            for i in range(n)]


class MockModel:
    """Deterministic mock: answers correctly for even task ids, wrong for odd.
    Counts calls; can be told to rate-limit or exhaust a daily quota."""

    def __init__(self, model_id, rpm_failures=0, daily_cap=None):
        self.model_id = model_id
        self.calls = 0
        self.rpm_failures = rpm_failures     # raise RetryableRateLimit this many times first
        self.daily_cap = daily_cap           # raise DailyLimitExhausted after this many calls
        self._rpm_left = rpm_failures

    def __call__(self, prompt: str) -> str:
        if self._rpm_left > 0:
            self._rpm_left -= 1
            raise RetryableRateLimit()
        if self.daily_cap is not None and self.calls >= self.daily_cap:
            raise DailyLimitExhausted()
        self.calls += 1
        # parse "What is i + i?" and answer; wrong (+1) for odd i
        i = int(prompt.split()[2])
        ans = 2 * i if i % 2 == 0 else 2 * i + 1
        return f"Thinking...\nANSWER: {ans}"

    def as_model(self):
        return Model(self.model_id, self)


def no_sleep(_):  # keep tests instant; records that backoff was requested
    no_sleep.waits.append(_)
no_sleep.waits = []


def run():
    tasks = make_tasks(4)

    # ------------------------------------------------------------------
    # 1 + 2: full run, then resume with zero new calls
    # ------------------------------------------------------------------
    cache, log = fresh_dirs()
    mocks = [MockModel("mock-a"), MockModel("mock-b"), MockModel("mock-c")]
    models = [m.as_model() for m in mocks]

    s = run_single_agent(tasks, models, n_samples=3, cache=cache, log=log,
                         sleep=no_sleep, verbose=False)
    expected = 3 * 4 * 3  # models x tasks x samples
    assert s["made"] == expected and s["logged"] == expected, s
    assert sum(m.calls for m in mocks) == expected

    s2 = run_single_agent(tasks, models, n_samples=3, cache=cache, log=log,
                          sleep=no_sleep, verbose=False)
    assert s2["made"] == 0, "resume made API calls despite full cache"
    assert s2["cache_hits"] == expected
    assert s2["logged"] == 0, "resume wrote duplicate log rows"
    assert len(log.load()) == expected
    print("[ok] full run + cache resume (zero new calls, zero duplicate logs)")

    # ------------------------------------------------------------------
    # 3: per-minute rate limit -> exponential backoff -> success
    # ------------------------------------------------------------------
    no_sleep.waits = []
    flaky = MockModel("flaky", rpm_failures=3)
    out = call_with_retry(flaky.as_model(), "What is 2 + 2?", sleep=no_sleep)
    assert "ANSWER" in out
    assert no_sleep.waits == [2.0, 4.0, 8.0], f"backoff schedule wrong: {no_sleep.waits}"
    print("[ok] RPM limit retried with exponential backoff (2s, 4s, 8s)")

    # ------------------------------------------------------------------
    # 4: daily limit -> graceful stop -> next-day resume completes the rest
    # ------------------------------------------------------------------
    cache, log = fresh_dirs()
    capped = MockModel("mock-a", daily_cap=5)      # dies after 5 calls "today"
    others = [MockModel("mock-b"), MockModel("mock-c")]
    models = [capped.as_model()] + [m.as_model() for m in others]

    s = run_single_agent(tasks, models, n_samples=3, cache=cache, log=log,
                         sleep=no_sleep, verbose=False)
    assert s["stopped_daily_limit"], "daily limit did not stop the run"
    assert s["made"] == 5
    assert len(log.load()) == 5, "logged rows should match completed calls"

    capped.daily_cap = None                        # "next day": quota reset
    s2 = run_single_agent(tasks, models, n_samples=3, cache=cache, log=log,
                          sleep=no_sleep, verbose=False)
    total = 3 * 4 * 3
    assert s2["made"] == total - 5, f"resume should make the remaining {total-5} calls, made {s2['made']}"
    assert s2["cache_hits"] == 5
    assert len(log.load()) == total
    print("[ok] daily-limit graceful stop + next-day resume completes the run")

    # ------------------------------------------------------------------
    # 5: log schema carries everything the analysis needs
    # ------------------------------------------------------------------
    rows = log.load()
    required = {"condition", "model_id", "task_id", "family", "sample_idx",
                "raw_response", "extracted", "correct", "fingerprint",
                "ground_truth", "template", "difficulty", "ts"}
    assert all(required <= set(r) for r in rows), "log rows missing fields"
    # mock answers wrong on odd tasks -> correctness pattern is as constructed
    odd_rows = [r for r in rows if r["task_id"] in ("t01", "t03")]
    even_rows = [r for r in rows if r["task_id"] in ("t00", "t02")]
    assert all(not r["correct"] for r in odd_rows)
    assert all(r["correct"] for r in even_rows)
    # wrong answers get real fingerprints (D_across depends on this)
    assert all(r["fingerprint"] not in (None, "") for r in odd_rows)
    print("[ok] log schema complete; correctness + fingerprints as constructed")

    print("\nALL HARNESS TESTS PASSED")


if __name__ == "__main__":
    run()
