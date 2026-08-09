"""Tests for the PER pipeline using mock models — no API calls.

Mock setup:
  - Executor: solves task correctly iff task index is even (so we KNOW which
    A0s are right/wrong).
  - Reviewers:
      perfect  : REJECTs exactly the wrong A0s, with a critique
      stamp    : ACCEPTs everything (rubber stamp)
      hostile  : REJECTs everything (false-rejection machine)
      broken   : returns malformed text (no VERDICT line)
  - Revision behavior: the executor's revision FIXES the answer if the critique
    contains the word 'sign' (our perfect reviewer's hint), otherwise the
    revision keeps/produces a wrong answer. Hostile rejections of CORRECT
    answers cause the revision to flip to a WRONG answer half the time
    (deterministic by task index) so damage_rate is testable.

Verifies:
  1. Executor solves each task ONCE (paired design) — call counting.
  2. C0 accuracy equals raw A0 accuracy.
  3. Perfect reviewer: recall=1, false_rejection=0, repair=1, final accuracy=1.
  4. Rubber stamp: verdicts all ACCEPT, accuracy == C0.
  5. Hostile: false_rejection=1, damage_rate matches construction, accuracy drops.
  6. Broken: malformed_review_rate=1, treated as ACCEPT, accuracy == C0.
  7. Caching: second run makes zero new model calls, no duplicate log rows.
  8. Token meter records every call with stage + condition.
"""

import sys, os, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core import Task, NUMERIC_ANSWER_INSTRUCTION
from harness import Model, DiskCache, RunLog
from pipeline import (run_per_experiment, analyze, parse_review, TokenMeter,
                      REVIEW_PROMPT, REVISE_PROMPT)

TMP = "/tmp/per_test"
N_TASKS = 8


def make_tasks():
    return [Task(f"t{i:02d}", "omnimath", f"Problem number {i}: compute the value."
                 + NUMERIC_ANSWER_INSTRUCTION, float(100 + i),
                 meta={"difficulty": 5.0, "round_to": 6})
            for i in range(N_TASKS)]


class MockExecutor:
    """Correct iff task index is even. Revision fixes iff critique mentions 'sign';
    if revising a CORRECT answer (hostile false rejection), flips to wrong when
    the task index is divisible by 4 (so damage is deterministic and testable)."""
    def __init__(self):
        self.model_id = "mock-executor"
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        i = int(__import__("re").search(r"Problem number (\d+)", prompt).group(1))
        truth = 100 + i
        if prompt.startswith("You previously solved"):
            # revision call
            if "sign" in prompt:                    # useful critique -> fix
                return f"Reconsidered.\nANSWER: {truth}"
            # useless critique: if A0 was correct, damage on i%4==0; else stay wrong
            if i % 2 == 0:                          # A0 was correct
                return (f"ANSWER: {truth + 7}" if i % 4 == 0
                        else f"ANSWER: {truth}")
            return f"ANSWER: {truth + 3}"           # A0 wrong, revision still wrong
        # initial solve
        ans = truth if i % 2 == 0 else truth + 1
        return f"Working...\nANSWER: {ans}"

    def as_model(self):
        return Model(self.model_id, self)


def make_reviewer(kind):
    calls = {"n": 0}
    def review(prompt: str) -> str:
        calls["n"] += 1
        i = int(__import__("re").search(r"Problem number (\d+)", prompt).group(1))
        a0_correct = (i % 2 == 0)
        if kind == "perfect":
            return ("VERDICT: ACCEPT" if a0_correct else
                    "VERDICT: REJECT\nCRITIQUE: there is a sign error near the end.")
        if kind == "stamp":
            return "VERDICT: ACCEPT"
        if kind == "hostile":
            return "VERDICT: REJECT\nCRITIQUE: this looks wrong to me."
        if kind == "broken":
            return "I am not sure what to say about this."
        raise ValueError(kind)
    m = Model(f"mock-{kind}", review)
    m._calls = calls
    return m


def fresh():
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP)
    return DiskCache(os.path.join(TMP, "cache")), RunLog(os.path.join(TMP, "log.jsonl"))


def run():
    # ---- input-truncation helpers (gpt-oss token-starvation fix) ----
    from pipeline import (strip_answer_instruction, truncate_solution,
                          SOLUTION_TAIL_CHARS)
    p = ("Solve the following problem.\n\nWhat is 2+2?\n\nWork through the problem "
         "step by step, then give your final answer on the last line in exactly "
         "this format:\nANSWER: <number>")
    stripped = strip_answer_instruction(p)
    assert "ANSWER: <number>" not in stripped, "answer instruction not stripped"
    assert "What is 2+2?" in stripped, "problem text lost during stripping"
    # no marker present -> unchanged (apart from trailing whitespace)
    assert strip_answer_instruction("Just a problem.") == "Just a problem."

    short = "ANSWER: 42"
    assert truncate_solution(short) == short, "short solution should be untouched"
    long_sol = ("A" * 5000) + "\nANSWER: 42"
    t = truncate_solution(long_sol)
    assert len(t) < len(long_sol), "long solution not truncated"
    assert "ANSWER: 42" in t, "truncation dropped the stated answer"
    assert "earlier steps omitted" in t, "no elision marker"
    assert len(t) <= SOLUTION_TAIL_CHARS + 60
    print("[ok] reviewer input truncation (strip instructions + keep tail)")

    # verdict parsing unit checks
    assert parse_review("VERDICT: ACCEPT")["verdict"] == "ACCEPT"
    r = parse_review("VERDICT: REJECT\nCRITIQUE: step 3 is wrong")
    assert r["verdict"] == "REJECT" and "step 3" in r["critique"]
    assert parse_review("")["malformed"] and parse_review("")["verdict"] == "ACCEPT"
    assert parse_review("gibberish")["malformed"]
    r = parse_review("verdict: reject")
    assert r["verdict"] == "REJECT" and r["critique"]  # stub critique
    print("[ok] verdict parsing")

    tasks = make_tasks()
    execu = MockExecutor()
    reviewers = {"C0": None,
                 "C1": make_reviewer("perfect"),
                 "C2": make_reviewer("stamp"),
                 "C3": make_reviewer("hostile"),
                 "C4": make_reviewer("broken")}
    cache, log = fresh()
    meter = TokenMeter(sleep=lambda s: None)

    s = run_per_experiment(tasks, execu.as_model(), reviewers, cache, log, meter,
                           verbose=False)
    assert not s["stopped_daily_limit"]
    assert s["completed_outcomes"] == N_TASKS * 5

    rows = log.load()
    m = analyze(rows, meter.records)

    # 1. paired design: executor solved each task once + revisions only
    #    revisions: C1 rejects the 4 wrong tasks; C3 rejects all 8. C2/C4 none.
    expected_exec_calls = N_TASKS + 4 + 8
    assert execu.calls == expected_exec_calls, (execu.calls, expected_exec_calls)
    print(f"[ok] paired design: executor called {execu.calls}x "
          f"({N_TASKS} solves + 12 revisions), not {N_TASKS*5}")

    # 2. C0 == raw A0 accuracy (evens correct = 0.5)
    assert abs(m["C0"]["accuracy"] - 0.5) < 1e-9

    # 3. perfect reviewer
    c1 = m["C1"]
    assert c1["recall"] == 1.0 and c1["false_rejection_rate"] == 0.0
    assert c1["repair_rate"] == 1.0 and c1["accuracy"] == 1.0
    assert c1["delta_vs_C0"] == 0.5
    print("[ok] perfect reviewer: recall=1, FRR=0, repair=1, acc=1.0")

    # 4. rubber stamp == baseline
    assert m["C2"]["accuracy"] == m["C0"]["accuracy"]
    assert m["C2"]["false_rejection_rate"] == 0.0 and m["C2"]["recall"] == 0.0
    print("[ok] rubber-stamp reviewer: identical to no-review baseline")

    # 5. hostile reviewer: FRR=1; damage on i%4==0 among the 4 correct A0s -> 2/4
    c3 = m["C3"]
    assert c3["false_rejection_rate"] == 1.0 and c3["recall"] == 1.0
    assert abs(c3["damage_rate"] - 0.5) < 1e-9, c3["damage_rate"]
    # accuracy: correct A0s -> 2 damaged, 2 kept; wrong A0s -> revisions all wrong
    assert abs(c3["accuracy"] - 0.25) < 1e-9
    assert c3["delta_vs_C0"] == -0.25
    print("[ok] hostile reviewer: FRR=1, damage=0.5, accuracy 0.25 (worse than none)")

    # 6. broken reviewer: all malformed -> treated as ACCEPT
    c4 = m["C4"]
    assert c4["malformed_review_rate"] == 1.0
    assert c4["accuracy"] == m["C0"]["accuracy"]
    print("[ok] malformed reviews -> ACCEPT + flagged, accuracy == baseline")

    # 7. caching: full rerun makes zero model calls, no duplicate rows
    before = execu.calls
    s2 = run_per_experiment(tasks, execu.as_model(), reviewers, cache, log,
                            TokenMeter(sleep=lambda s: None), verbose=False)
    assert execu.calls == before, "rerun made executor calls despite cache"
    assert s2["completed_outcomes"] == 0
    assert len(log.load()) == N_TASKS * 5
    print("[ok] resume: zero new calls, zero duplicate log rows")

    # 8. token meter recorded every call with stage+condition
    stages = {t["stage"] for t in meter.records}
    assert stages == {"execute", "review", "revise"}
    assert all(t["in_tokens"] > 0 and t["out_tokens"] > 0 for t in meter.records)
    assert any(t["condition"] == "shared" for t in meter.records)
    print("[ok] token meter: execute/review/revise all recorded")

    print("\nALL PIPELINE TESTS PASSED")


if __name__ == "__main__":
    run()