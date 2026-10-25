"""Audit script for hidden test counts and candidate problem selection.

Reproduces all problem selection numbers for the pre-registered analysis plan:
- Excluded candidate problems with n_hidden_tests < 10
- Composition and test statistics of the N = 125 eligible problems
- Pre- vs post-cutoff temporal breakdown (boundary: 2025-01-31)
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tasks.livecodebench_tasks import load_lcb_tasks, save_problem_ids, load_problem_ids

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "run_code_data", "raw")
CALIB_BATCH_1 = os.path.join(os.path.dirname(__file__), "..", "run_code_data", "calibration", "batch_1.json")
FROZEN_100_PATH = os.path.join(os.path.dirname(__file__), "..", "run_code_data", "problem_ids.json")


def hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    # 1. Record hash of current 100-problem draft
    if os.path.exists(FROZEN_100_PATH):
        old_hash = hash_file(FROZEN_100_PATH)
        old_ids = load_problem_ids(FROZEN_100_PATH)
        print(f"Superseded 100-problem draft hash : {old_hash}")
        print(f"Superseded draft problem count    : {len(old_ids)}")
    else:
        old_hash = "FILE_NOT_FOUND"
        print("No previous problem_ids.json found.")

    # 2. Calibration problem IDs to permanently exclude
    with open(CALIB_BATCH_1, "r", encoding="utf-8") as f:
        calib_data = json.load(f)
    calib_ids = set(calib_data["problem_ids"])
    print(f"Calibration excluded count        : {len(calib_ids)}")

    # 3. Load all candidates from approved window
    candidates = load_lcb_tasks(
        n=None,
        difficulties=("medium", "hard"),
        date_from="2024-10-01",
        date_to=None,
        raw_dir=RAW_DIR,
    )
    print(f"Total candidates in window        : {len(candidates)}")

    # Exclude calibration tasks
    pool_135 = [t for t in candidates if t.task_id not in calib_ids]
    print(f"Candidate pool after calib excls  : {len(pool_135)}")
    assert len(pool_135) == 135

    # 4. Filter n_hidden_tests >= 10
    excluded_lt10 = [t for t in pool_135 if len(t.meta.get("hidden_tests", [])) < 10]
    eligible_125 = [t for t in pool_135 if len(t.meta.get("hidden_tests", [])) >= 10]

    print(f"\n--- Excluded Candidates (< 10 hidden tests): {len(excluded_lt10)} ---")
    for t in sorted(excluded_lt10, key=lambda x: x.task_id):
        cnt = len(t.meta.get("hidden_tests", []))
        pub = len(t.meta.get("public_tests", []))
        print(f"  {t.task_id:10} | {cnt} hidden, {pub} public | diff={t.meta.get('difficulty'):6} | contest_date={t.meta.get('contest_date')}")

    print(f"\n--- Eligible Problems (>= 10 hidden tests): {len(eligible_125)} ---")
    assert len(eligible_125) == 125, f"Expected exactly 125, got {len(eligible_125)}"

    # 5. Difficulty distribution
    diff_counts = Counter(t.meta.get("difficulty") for t in eligible_125)
    print(f"Difficulty composition: {dict(diff_counts)}")

    # 6. Hidden test statistics for the 125
    test_counts = [len(t.meta.get("hidden_tests", [])) for t in eligible_125]
    sorted_tc = sorted(test_counts)
    t_min = min(test_counts)
    t_median = sorted_tc[len(sorted_tc) // 2]
    t_mean = sum(test_counts) / len(test_counts)
    t_max = max(test_counts)
    print(f"Hidden tests stats (N=125): min={t_min}, median={t_median}, mean={t_mean:.2f}, max={t_max}")

    # 7. Temporal breakdown across contest months and boundary 2025-01-31
    cutoff_boundary = "2025-01-31T23:59:59"
    pre_cutoff = [t for t in eligible_125 if (t.meta.get("contest_date") or "") <= cutoff_boundary]
    post_cutoff = [t for t in eligible_125 if (t.meta.get("contest_date") or "") > cutoff_boundary]
    print(f"\nTemporal split (boundary 2025-01-31):")
    print(f"  Pre-cutoff  (<= 2025-01-31): {len(pre_cutoff)}")
    print(f"  Post-cutoff (>  2025-01-31): {len(post_cutoff)}")

    months = Counter((t.meta.get("contest_date") or "")[:7] for t in eligible_125)
    print(f"Breakdown by contest month:")
    for m in sorted(months.keys()):
        print(f"  {m}: {months[m]} problems")

    # 8. Fixed processing order: deterministic sort then random.Random(42).shuffle
    # Sorting first ensures cross-platform consistency
    eligible_125.sort(key=lambda t: (t.meta.get("contest_date", ""), t.task_id))
    rng = random.Random(42)
    rng.shuffle(eligible_125)

    # 9. Save new 125-id file
    save_problem_ids(FROZEN_100_PATH, eligible_125)
    new_hash = hash_file(FROZEN_100_PATH)
    print(f"\nNew 125-problem pre-registered JSON hash: {new_hash}")


if __name__ == "__main__":
    main()
