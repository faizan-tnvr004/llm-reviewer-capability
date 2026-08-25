"""Tests for the math generator + grader.

Run:  python3 tests/test_math.py

Checks:
  1. Determinism — same seed produces identical tasks (reproducibility).
  2. Grader accepts a correct 'ANSWER: x' response.
  3. Grader rejects a wrong answer and fingerprints it.
  4. Extractor falls back to the last number when no ANSWER line exists.
  5. Missing/garbled answers grade as incorrect with fingerprint NO_ANSWER.
  6. Two different wrong responses produce different fingerprints,
     and two identical wrong answers produce the same fingerprint
     (this is what D_across relies on).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core import grade_numeric
from tasks.math_tasks import generate_math_tasks


def main():
    failures = 0

    # 1. Determinism
    a = generate_math_tasks(n=10, difficulty=2, seed=42)
    b = generate_math_tasks(n=10, difficulty=2, seed=42)
    assert all(x.prompt == y.prompt and x.ground_truth == y.ground_truth
               for x, y in zip(a, b)), "seeded generation is not deterministic"
    c = generate_math_tasks(n=10, difficulty=2, seed=7)
    assert any(x.prompt != y.prompt for x, y in zip(a, c)), "different seeds gave identical tasks"
    print("[ok] determinism")

    task = a[0]
    truth = float(task.ground_truth)

    # 2. Correct answer accepted
    r = grade_numeric(task, f"Let me work through this...\nANSWER: {truth}")
    assert r["correct"], f"correct answer graded wrong: {r}"
    print("[ok] correct answer accepted")

    # 3. Wrong answer rejected + fingerprinted
    r = grade_numeric(task, f"Some reasoning.\nANSWER: {truth + 3.7}")
    assert not r["correct"], "wrong answer graded correct"
    assert r["fingerprint"] != "NO_ANSWER"
    print(f"[ok] wrong answer rejected, fingerprint={r['fingerprint']}")

    # 4. Fallback: last number in text
    r = grade_numeric(task, f"After computing everything the result is {truth} dollars.")
    assert r["correct"], f"fallback extraction failed: {r}"
    print("[ok] fallback extraction (last number)")

    # 5. No answer at all
    r = grade_numeric(task, "I am not sure how to solve this.")
    assert not r["correct"] and r["fingerprint"] == "NO_ANSWER"
    print("[ok] missing answer -> NO_ANSWER fingerprint")

    # 6. Fingerprint semantics for D_across
    w1 = grade_numeric(task, f"ANSWER: {truth + 1}")
    w2 = grade_numeric(task, f"ANSWER: {truth + 2}")
    w3 = grade_numeric(task, f"ANSWER: {truth + 1}")
    assert w1["fingerprint"] != w2["fingerprint"], "different wrong answers share a fingerprint"
    assert w1["fingerprint"] == w3["fingerprint"], "identical wrong answers differ in fingerprint"
    print("[ok] fingerprint distinguishes different wrong answers, matches identical ones")

    # Show a sample of generated tasks
    print("\n--- sample tasks (seed=42, difficulty=2) ---")
    for t in a[:5]:
        first_line = t.prompt.split("\n")[0]
        print(f"{t.task_id} [{t.meta['template']}] truth={t.ground_truth}\n  {first_line}\n")

    print("ALL TESTS PASSED" if failures == 0 else f"{failures} FAILURES")


if __name__ == "__main__":
    main()


def test_all():
    main()
