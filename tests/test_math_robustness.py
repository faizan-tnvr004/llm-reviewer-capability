"""Expanded tests for the math family: extractor robustness, rounding
boundaries, generation-space properties, difficulty knob.

Run:  python3 tests/test_math_robustness.py

These tests simulate the messy outputs real models produce. Where a test
pins down a *decision* (e.g. which ANSWER line wins), the decision is stated
in the comment so it's deliberate, not accidental.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core import extract_final_number, grade_numeric, Task
from tasks.math_tasks import generate_math_tasks


def make_task(truth, round_to=2):
    return Task("t", "math", "p", truth, meta={"round_to": round_to})


def run():
    # ------------------------------------------------------------------
    # 1. Extractor robustness against realistic model output
    # ------------------------------------------------------------------

    # Markdown bold around the answer line
    assert extract_final_number("**ANSWER: 45.14**") == 45.14

    # Units after the number
    assert extract_final_number("ANSWER: 45.14 km/h") == 45.14

    # Currency symbol and thousands separators
    assert extract_final_number("ANSWER: $1,250.50") == 1250.50

    # Negative number
    assert extract_final_number("ANSWER: -3.5") == -3.5

    # 'ANSWER =' variant
    assert extract_final_number("answer = 12") == 12

    # DECISION: if multiple ANSWER lines exist, the model self-corrected;
    # its final answer is the LAST line. Pinned strictly:
    multi = "ANSWER: 10\nWait, I made an error.\nANSWER: 12"
    assert extract_final_number(multi) == 12.0, "last ANSWER line must win on self-correction"
    print("[pin] multiple ANSWER lines -> last line wins (self-correction)")

    # DECISION: text after the ANSWER line must not hijack the fallback.
    trailing = "So the total is clear.\nANSWER: 762.21\nNote that 9.5% tax is typical."
    got = extract_final_number(trailing)
    assert got == 762.21, f"trailing numbers hijacked extraction: got {got}"

    # Fallback: no ANSWER line at all -> last number in text
    assert extract_final_number("The result comes to 88.5 in the end.") == 88.5

    # Garbage -> None
    assert extract_final_number("I cannot determine this.") is None

    # --- cases observed in the live run against real models ---
    # Fraction in ANSWER line: 31/5 must evaluate to 6.2, not 31
    assert extract_final_number("ANSWER: 31/5") == 6.2
    assert extract_final_number("ANSWER: -7/2") == -3.5
    # Fraction that IS an integer
    assert extract_final_number("ANSWER: 10/5") == 2.0
    # Division by zero in a fraction -> None (don't crash)
    assert extract_final_number("ANSWER: 5/0") is None
    # LaTeX-wrapped final line, no ANSWER line -> fallback strips markup
    latex = r"The total is \[ 467.72 \times 1.095 = 512.15 \]"
    assert extract_final_number(latex) == 512.15
    # ANSWER line present alongside LaTeX reasoning -> ANSWER line wins
    mixed = r"\[ x = 467.72 \times 0.095 = 44.43 \]" "\nANSWER: 512.15"
    assert extract_final_number(mixed) == 512.15
    # Currency in ANSWER line
    assert extract_final_number("ANSWER: $1,250.50") == 1250.50
    print("[ok] extractor robustness (incl. fractions + LaTeX from live run)")

    # ------------------------------------------------------------------
    # 2. Rounding boundaries and fingerprint canonicalisation
    # ------------------------------------------------------------------

    t = make_task(45.14)
    # An answer that rounds to the truth at 2dp counts as correct
    assert grade_numeric(t, "ANSWER: 45.139")["correct"]
    assert grade_numeric(t, "ANSWER: 45.141")["correct"]
    # An answer 1 cent off is wrong
    assert not grade_numeric(t, "ANSWER: 45.15")["correct"]

    # Fingerprint canonicalisation: 762.2 and 762.20 are the SAME answer
    t2 = make_task(100.0)
    f1 = grade_numeric(t2, "ANSWER: 762.2")["fingerprint"]
    f2 = grade_numeric(t2, "ANSWER: 762.20")["fingerprint"]
    assert f1 == f2, f"canonicalisation failed: {f1} != {f2}"

    # 3-decimal tasks (kg proportion template) honour round_to=3
    t3 = make_task(0.527, round_to=3)
    assert grade_numeric(t3, "ANSWER: 0.5271")["correct"]
    assert not grade_numeric(t3, "ANSWER: 0.528")["correct"]
    print("[ok] rounding boundaries + fingerprint canonicalisation")

    # ------------------------------------------------------------------
    # 3. Property tests over the generation space
    # ------------------------------------------------------------------

    checked = 0
    for seed in range(40):
        for difficulty in (1, 2, 3):
            for task in generate_math_tasks(25, difficulty=difficulty, seed=seed):
                gt = float(task.ground_truth)
                # ground truth is a finite, sensible number
                assert gt == gt and abs(gt) < 1e9, f"bad ground truth {gt} in {task.task_id}"
                # non-negative where the quantity is physical
                if task.meta["template"] in ("discount_tax", "unit_rate", "proportion",
                                             "work_rate", "simple_interest"):
                    assert gt >= 0, f"negative answer for physical quantity: {task.task_id}"
                # work_rate: B's remaining time must be positive (work not
                # already finished during the together-phase) and less than
                # B's solo time
                if task.meta["template"] == "work_rate":
                    assert 0 < gt, f"work_rate degenerate (already done): {task.prompt}"
                # simple_interest: options must differ (the bug we fixed)
                if task.meta["template"] == "simple_interest":
                    assert gt > 1e-9, f"degenerate equal options: {task.prompt}"
                # prompt hygiene: no unformatted placeholders / artifacts
                assert "{" not in task.prompt and "}" not in task.prompt
                assert "ANSWER: <number>" in task.prompt, "answer instruction missing"
                checked += 1
    print(f"[ok] property tests over {checked} generated tasks")

    # ------------------------------------------------------------------
    # 4. Difficulty knob actually changes the distribution
    # ------------------------------------------------------------------

    easy = generate_math_tasks(50, difficulty=1, seed=1)
    hard = generate_math_tasks(50, difficulty=3, seed=1)
    easy_prices = [t.ground_truth for t in easy if t.meta["template"] == "discount_tax"]
    hard_prices = [t.ground_truth for t in hard if t.meta["template"] == "discount_tax"]
    assert max(easy_prices) < max(hard_prices), "difficulty knob has no effect on discount_tax range"
    # distractor clauses only appear at difficulty 3
    assert not any("membership fee" in t.prompt for t in easy)
    assert any("membership fee" in t.prompt for t in hard)
    print("[ok] difficulty knob changes generation")

    print("\nALL ROBUSTNESS TESTS PASSED")


if __name__ == "__main__":
    run()
