import json
import os
import sys
import time
import subprocess
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tasks.livecodebench_tasks import load_lcb_tasks, extract_program
from harness import DiskCache
from code_exec import run_program

target_pids = ["abc377_f", "abc392_g", "arc186_e", "abc378_g"]

tasks = load_lcb_tasks(problem_ids=target_pids, raw_dir="run_code_data/raw")
tasks_by_id = {t.task_id: t for t in tasks}

cache = DiskCache("run_code_data/calibration_cache")

# Phase B log rows
phase_b = {}
for line in open("run_code_data/calibration/log_batch_1.jsonl"):
    if line.strip():
        r = json.loads(line)
        phase_b[r["task_id"]] = r

def run_local(code: str, inp: str, timeout: float = 6.0):
    with tempfile.TemporaryDirectory() as td:
        sp = os.path.join(td, "sol.py")
        with open(sp, "w", encoding="utf-8") as f:
            f.write(code)
        t0 = time.perf_counter()
        try:
            p = subprocess.run([sys.executable, sp], input=inp, text=True, capture_output=True, timeout=timeout)
            dur = time.perf_counter() - t0
            return dur, p.returncode, p.stdout[:50], False
        except subprocess.TimeoutExpired:
            dur = time.perf_counter() - t0
            return dur, -1, "TIMEOUT", True

print("=== COMPARISON OF THE FOUR TIMEOUT / EDGE PROBLEMS ===")
for pid in target_pids:
    t = tasks_by_id[pid]
    resp = cache.get("per_execute", "gemini-3.1-flash-lite", pid, 0)
    code = extract_program(resp)
    test1 = t.meta["hidden_tests"][0]
    inp1 = test1["input"]
    exp1 = test1["expected_output"]

    # Local run
    loc_dur, loc_rc, loc_out, loc_to = run_local(code, inp1, timeout=6.0)

    # Docker run
    t0 = time.perf_counter()
    doc_res = run_program(code, inp1, timeout=6.0)
    doc_dur = time.perf_counter() - t0

    pb = phase_b[pid]
    pb_fp = pb.get("final_fingerprint") or pb.get("a0_fingerprint")

    print(f"\nProblem: {pid}")
    print(f"  Phase B outcome:      {pb_fp} (correct: {pb['final_correct']})")
    print(f"  Test 1 Local time:    {loc_dur:.3f}s (timed out: {loc_to})")
    print(f"  Test 1 Docker time:   {doc_dur:.3f}s (timed out: {doc_res['timed_out']})")
    print(f"  Test 1 Docker status: {'TIMEOUT' if doc_res['timed_out'] else ('PASS' if doc_res['stdout'].strip() == exp1.strip() else 'WRONG OUTPUT')}")
