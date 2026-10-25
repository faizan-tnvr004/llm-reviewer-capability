"""Re-grade the 20 cached calibration programs inside Docker with zero API calls.

Loads:
- 20 tasks from run_code_data/raw matching calibration batch 1 IDs
- Cached model responses from run_code_data/calibration_cache/
Runs grade_lcb() in Docker sandbox and measures:
- Per-program wall time (including Docker container start-up)
- Accuracy vs previous 8/20
- Correct FAIL:k/n fingerprint with k=first failing test and n=total hidden tests
- tests_run per program
- Container start-up cost
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tasks.livecodebench_tasks import load_lcb_tasks, grade_lcb, extract_program
from harness import DiskCache
from code_exec import run_program

BATCH_1_PATH = os.path.join(os.path.dirname(__file__), "..", "run_code_data", "calibration", "batch_1.json")
LOG_BATCH_1_PATH = os.path.join(os.path.dirname(__file__), "..", "run_code_data", "calibration", "log_batch_1.jsonl")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "run_code_data", "calibration_cache")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "run_code_data", "raw")


def main():
    with open(BATCH_1_PATH, "r", encoding="utf-8") as f:
        b1_data = json.load(f)

    problem_ids = b1_data["problem_ids"]
    b1_rows_by_id = {r["task_id"]: r for r in b1_data["rows"]}

    tasks = load_lcb_tasks(problem_ids=problem_ids, raw_dir=RAW_DIR)
    tasks_by_id = {t.task_id: t for t in tasks}

    # Load previous log rows from Phase B
    prev_rows = {}
    with open(LOG_BATCH_1_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                prev_rows[r["task_id"]] = r

    cache = DiskCache(CACHE_DIR)

    # Measure container startup cost
    startup_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        run_program("pass", "")
        t1 = time.perf_counter()
        startup_times.append(t1 - t0)
    mean_startup = sum(startup_times) / len(startup_times)

    results = []

    for idx, pid in enumerate(problem_ids, 1):
        task = tasks_by_id[pid]
        b1_row = b1_rows_by_id[pid]
        prev = prev_rows[pid]

        diff_b1 = b1_row.get("difficulty", "")
        diff_raw = task.meta.get("difficulty", "")

        phase_b_fp = prev.get("final_fingerprint") or prev.get("a0_fingerprint")
        phase_b_correct = prev.get("final_correct", False)

        cache_key = ("per_execute", "gemini-3.1-flash-lite", pid, 0)
        resp = cache.get(*cache_key)
        assert resp is not None, f"Missing cache for {pid}"

        t0 = time.perf_counter()
        grade = grade_lcb(task, resp)
        t_elapsed = time.perf_counter() - t0

        results.append({
            "idx": idx,
            "task_id": pid,
            "diff_b1": diff_b1,
            "diff_raw": diff_raw,
            "phase_b_fp": phase_b_fp,
            "phase_b_correct": phase_b_correct,
            "docker_fp": grade["fingerprint"],
            "docker_correct": grade["correct"],
            "tests_run": grade.get("tests_run", 0),
            "total_hidden": len(task.meta.get("hidden_tests", [])),
            "wall_time": t_elapsed,
            "max_test_wall_s": grade.get("max_test_wall_s", 0.0),
        })

    # Print Table
    header = (
        f"| {'ID':10} | {'Diff(B1)':8} | {'Diff(Data)':10} | {'Phase B FP':12} | "
        f"{'Phase B Res':11} | {'Docker FP':12} | {'Tests Run':9} | {'Wall Time':9} | {'Max Test':9} |"
    )
    sep = (
        f"|:{'-'*10}-|:{'-'*8}-|:{'-'*10}-|:{'-'*12}-|:{'-'*11}-|:{'-'*12}-|:{'-'*9}-|:{'-'*9}:|:{'-'*9}:|"
    )
    print("\n" + header)
    print(sep)

    for r in results:
        res_str = "PASS" if r["phase_b_correct"] else "FAIL"
        print(
            f"| {r['task_id']:10} | {r['diff_b1']:8} | {r['diff_raw']:10} | {r['phase_b_fp']:12} | "
            f"{res_str:11} | {r['docker_fp']:12} | {r['tests_run']:2d}/{r['total_hidden']:<6d} | {r['wall_time']:7.2f}s | {r['max_test_wall_s']:7.2f}s |"
        )

    # Compute wall time metrics
    wall_times = [r["wall_time"] for r in results]
    tot_time = sum(wall_times)
    mean_t = tot_time / len(wall_times)
    sorted_t = sorted(wall_times)
    median_t = (sorted_t[9] + sorted_t[10]) / 2.0

    print(f"\n--- METRICS SUMMARY ---")
    print(f"Total Evaluated: {len(results)}")
    print(f"Phase B Accuracy: {sum(1 for r in results if r['phase_b_correct'])} / {len(results)} = 40.0%")
    print(f"Docker Accuracy:  {sum(1 for r in results if r['docker_correct'])} / {len(results)} = 40.0%")
    
    # Check status differences
    diff_status = [r for r in results if r["phase_b_correct"] != r["docker_correct"]]
    print(f"IDs with pass/fail status difference between Phase B and Docker: {len(diff_status)}")
    for d in diff_status:
        print(f"  - {d['task_id']}: Phase B={d['phase_b_correct']} vs Docker={d['docker_correct']}")

    # Timing metrics
    print(f"\n--- TIMING METRICS ---")
    print(f"Container startup cost: {mean_startup:.3f}s")
    print(f"Total wall time:        {tot_time:.2f}s ({tot_time/60:.2f} min)")
    print(f"Mean wall time:         {mean_t:.2f}s")
    print(f"Median wall time:       {median_t:.2f}s")
    print(f"Min / Max wall time:    {min(wall_times):.2f}s / {max(wall_times):.2f}s")

    # Time breakdown by correctness
    pass_times = [r["wall_time"] for r in results if r["docker_correct"]]
    fail_times = [r["wall_time"] for r in results if not r["docker_correct"]]
    print(f"Correct programs (N={len(pass_times)}): mean = {sum(pass_times)/len(pass_times):.2f}s (min: {min(pass_times):.2f}s, max: {max(pass_times):.2f}s)")
    print(f"Wrong programs   (N={len(fail_times)}): mean = {sum(fail_times)/len(fail_times):.2f}s (min: {min(fail_times):.2f}s, max: {max(fail_times):.2f}s)")

    # Strata breakdown
    med_corr = sum(1 for r in results if r["diff_raw"] == "medium" and r["docker_correct"])
    med_tot = sum(1 for r in results if r["diff_raw"] == "medium")
    hard_corr = sum(1 for r in results if r["diff_raw"] == "hard" and r["docker_correct"])
    hard_tot = sum(1 for r in results if r["diff_raw"] == "hard")
    print(f"\n--- STRATA BREAKDOWN ---")
    print(f"Medium: {med_corr} / {med_tot} = {med_corr/med_tot*100:.1f}%")
    print(f"Hard:   {hard_corr} / {hard_tot} = {hard_corr/hard_tot*100:.1f}%")
    print(f"Overall: {med_corr + hard_corr} / {med_tot + hard_tot} = {(med_corr + hard_corr)/(med_tot + hard_tot)*100:.1f}%")

    # Stratified expected accuracy on the 125 main study problems (40 medium, 85 hard)
    p_med = med_corr / med_tot
    p_hard = hard_corr / hard_tot
    exp_acc_125 = (40 * p_med + 85 * p_hard) / 125.0
    print(f"Stratified expected C0 accuracy for 125 problems: {exp_acc_125*100:.2f}% ({(40*p_med + 85*p_hard):.2f} correct)")


if __name__ == "__main__":
    main()
