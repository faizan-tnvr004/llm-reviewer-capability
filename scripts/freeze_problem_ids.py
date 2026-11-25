"""Freeze N = 125 problem IDs for the pre-registration (Task T14).

Steps:
  1. Load candidates from approved window: date_from="2024-10-01", difficulties=("medium", "hard").
  2. Permanently exclude the 20 Batch 1 calibration problem IDs.
  3. Filter n_hidden_tests >= 10 (excludes 10 candidates with <= 4 tests).
  4. N = 125 = all remaining eligible candidates (no sampling).
  5. Sort deterministically by (contest_date, question_id).
  6. Store in fixed processing order (shuffled with seed 42).
  7. Save to run_code_data/problem_ids.json using save_problem_ids.
  8. Verify round-trip loading and absence of special judge or functional problems.
"""

from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tasks.livecodebench_tasks import (
    load_lcb_tasks, save_problem_ids, load_problem_ids,
    is_special_judge_statement,
)

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "run_code_data", "raw")
CALIB_BATCH_1 = os.path.join(os.path.dirname(__file__), "..", "run_code_data", "calibration", "batch_1.json")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "run_code_data", "problem_ids.json")


def main():
    with open(CALIB_BATCH_1, "r", encoding="utf-8") as f:
        calib_data = json.load(f)
    calib_ids = set(calib_data["problem_ids"])
    print(f"Loaded {len(calib_ids)} calibration problem IDs to exclude.")

    # Load all candidates from the approved window
    candidates = load_lcb_tasks(
        n=None,
        difficulties=("medium", "hard"),
        date_from="2024-10-01",
        date_to=None,
        raw_dir=RAW_DIR,
    )
    print(f"Total candidates in window: {len(candidates)}")

    # Exclude calibration tasks
    non_calib = [t for t in candidates if t.task_id not in calib_ids]
    print(f"Candidates remaining after calibration exclusion: {len(non_calib)}")

    # Filter n_hidden_tests >= 10
    eligible = [t for t in non_calib if len(t.meta.get("hidden_tests", [])) >= 10]
    excluded_lt10 = [t for t in non_calib if len(t.meta.get("hidden_tests", [])) < 10]
    print(f"Excluded for < 10 hidden tests: {len(excluded_lt10)}")
    print(f"Eligible problems (all included, N = 125): {len(eligible)}")
    assert len(eligible) == 125, f"Expected 125 eligible problems, found {len(eligible)}"

    # Deterministic sorting prior to shuffling
    eligible.sort(key=lambda t: (t.meta.get("contest_date", ""), t.task_id))

    # Fixed processing order: shuffle with seed 42
    rng = random.Random(42)
    rng.shuffle(eligible)

    selected = eligible
    assert len(selected) == 125

    # Verification checks
    selected_ids = [t.task_id for t in selected]
    assert len(set(selected_ids)) == 125, "Duplicate IDs in selection!"
    assert len(set(selected_ids).intersection(calib_ids)) == 0, "Calibration ID leaked into main set!"

    by_diff = {}
    for t in selected:
        d = t.meta.get("difficulty")
        by_diff[d] = by_diff.get(d, 0) + 1
        assert not is_special_judge_statement(t.meta.get("statement", ""))
        for tc in t.meta.get("public_tests", []) + t.meta.get("hidden_tests", []):
            if isinstance(tc, dict) and "testtype" in tc:
                assert tc["testtype"] == "stdin"

    print("\nDifficulty distribution in frozen set (N = 125):")
    for d, count in sorted(by_diff.items()):
        print(f"  {d}: {count}")

    # Save to run_code_data/problem_ids.json
    save_problem_ids(OUT_PATH, selected)

    # Verify round-trip load
    loaded_ids = load_problem_ids(OUT_PATH)
    assert loaded_ids == selected_ids, "Loaded IDs do not match saved IDs!"
    print(f"\n[ok] Successfully frozen and verified 125 problem IDs in {OUT_PATH}")


if __name__ == "__main__":
    main()
