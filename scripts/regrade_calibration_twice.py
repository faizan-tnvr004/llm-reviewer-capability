"""Run calibration re-grade twice in Docker sandbox and check for stability."""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tasks.livecodebench_tasks import load_lcb_tasks, grade_lcb
from harness import DiskCache

BATCH_1_PATH = os.path.join(os.path.dirname(__file__), "..", "run_code_data", "calibration", "batch_1.json")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "run_code_data", "calibration_cache")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "run_code_data", "raw")

def run_pass(tasks, cache, pass_num):
    print(f"\n=== STARTING CALIBRATION PASS {pass_num} ===")
    results = []
    t_start = time.perf_counter()
    for idx, task in enumerate(tasks, 1):
        resp = cache.get("per_execute", "gemini-3.1-flash-lite", task.task_id, 0)
        t0 = time.perf_counter()
        grade = grade_lcb(task, resp)
        wall = time.perf_counter() - t0
        results.append({
            "idx": idx,
            "task_id": task.task_id,
            "difficulty": task.meta.get("difficulty", ""),
            "fingerprint": grade["fingerprint"],
            "correct": grade["correct"],
            "tests_run": grade.get("tests_run", 0),
            "total_hidden": len(task.meta.get("hidden_tests", [])),
            "wall_time": wall,
            "max_test_wall_s": grade.get("max_test_wall_s", 0.0),
        })
    tot_wall = time.perf_counter() - t_start
    print(f"Pass {pass_num} completed in {tot_wall:.2f}s ({tot_wall/60:.2f} min).")
    return results

def main():
    with open(BATCH_1_PATH, "r", encoding="utf-8") as f:
        problem_ids = json.load(f)["problem_ids"]

    tasks = load_lcb_tasks(problem_ids=problem_ids, raw_dir=RAW_DIR)
    cache = DiskCache(CACHE_DIR)

    res1 = run_pass(tasks, cache, 1)
    res2 = run_pass(tasks, cache, 2)

    # Print Table for Pass 1 and Pass 2
    header = (
        f"| {'ID':10} | {'Diff':8} | {'Pass 1 FP':12} | {'P1 Wall':8} | {'P1 MaxTest':10} | "
        f"{'Pass 2 FP':12} | {'P2 Wall':8} | {'P2 MaxTest':10} | {'Match':6} |"
    )
    sep = (
        f"|:{'-'*10}-|:{'-'*8}-|:{'-'*12}-|:{'-'*8}-|:{'-'*10}-|:{'-'*12}-|:{'-'*8}-|:{'-'*10}-|:{'-'*6}:|"
    )
    print("\n" + header)
    print(sep)

    disagreements = []
    correct_above_4_8 = []

    for r1, r2 in zip(res1, res2):
        pid = r1["task_id"]
        match = (r1["fingerprint"] == r2["fingerprint"])
        if not match:
            disagreements.append((pid, r1["fingerprint"], r2["fingerprint"]))
        if r1["correct"]:
            if r1["max_test_wall_s"] > 4.8 or r2["max_test_wall_s"] > 4.8:
                correct_above_4_8.append((pid, r1["max_test_wall_s"], r2["max_test_wall_s"]))
        print(
            f"| {pid:10} | {r1['difficulty']:8} | {r1['fingerprint']:12} | {r1['wall_time']:7.2f}s | "
            f"{r1['max_test_wall_s']:9.2f}s | {r2['fingerprint']:12} | {r2['wall_time']:7.2f}s | "
            f"{r2['max_test_wall_s']:9.2f}s | {str(match):6} |"
        )

    print(f"\n--- COMPARISON REPORT ---")
    print(f"Fingerprint disagreements between Pass 1 and Pass 2: {len(disagreements)}")
    for d in disagreements:
        print(f"  DISAGREEMENT on {d[0]}: Pass 1 = {d[1]}, Pass 2 = {d[2]}")

    print(f"Correct programs with a single test wall time above 4.8s: {len(correct_above_4_8)} / 8")
    for c in correct_above_4_8:
        print(f"  {c[0]}: Pass 1 max = {c[1]:.2f}s, Pass 2 max = {c[2]:.2f}s")

if __name__ == "__main__":
    main()
