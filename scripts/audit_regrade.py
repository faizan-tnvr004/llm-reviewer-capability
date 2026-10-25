"""Exploratory Analysis E3: execution stability audit (second Docker pass).

Re-grades every program recorded in run_code_data/programs.jsonl (the A0
program and the final program of every (task, condition) row) in fresh Docker
containers, calling grade_lcb directly so the pipeline's grade cache is never
consulted. Each distinct (task_id, code hash) is run once; its second-pass
fingerprint is compared with every first-pass fingerprint recorded for it.

First-pass outcomes remain the confirmatory results; this pass only reports
fingerprint disagreements (borderline timeouts, non-determinism).

Run from repo root:  python scripts/audit_regrade.py [--programs PATH] [--out PATH]
Zero API calls. Requires Docker (UNSAFE_LOCAL_EXEC is refused).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Callable, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from analyze_code import load_programs  # noqa: E402

DEFAULT_PROGRAMS = os.path.join(ROOT, "run_code_data", "programs.jsonl")
DEFAULT_OUT = os.path.join(ROOT, "run_code_data", "audit_regrade.json")
RAW_DIR = os.path.join(ROOT, "run_code_data", "raw")


def parse_args():
    p = argparse.ArgumentParser(description="E3: second Docker pass over programs.jsonl")
    p.add_argument("--programs", default=DEFAULT_PROGRAMS, help="Path to programs.jsonl")
    p.add_argument("--out", default=DEFAULT_OUT, help="Path to JSON report")
    p.add_argument("--raw-dir", default=RAW_DIR, help="LiveCodeBench raw JSONL directory")
    return p.parse_args()


def collect_programs(entries: List[dict]) -> Dict[tuple, dict]:
    """Group programs by (task_id, code_hash). Each group holds the code and the
    first-pass fingerprints recorded for it as (condition, stage, fingerprint)."""
    groups: Dict[tuple, dict] = {}
    for e in entries:
        for stage in ("a0", "final"):
            code = e.get(f"{stage}_code")
            if stage == "final" and code is None and "final_code" not in e:
                continue  # entry written before final_code was logged
            chash = e.get(f"{stage}_code_hash")
            g = groups.setdefault((e["task_id"], chash), {"code": code, "first_pass": []})
            g["first_pass"].append((e.get("condition"), stage, e.get(f"{stage}_fingerprint")))
    return groups


def audit(entries: List[dict], tasks_by_id: dict, grade_fn: Callable,
          verbose: bool = True) -> dict:
    """Second-pass every distinct program and list fingerprint disagreements."""
    groups = collect_programs(entries)
    results, disagreements = [], []
    for idx, ((task_id, chash), g) in enumerate(sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))), 1):
        code = g["code"]
        response = "" if code is None else f"```python\n{code}\n```"
        t0 = time.perf_counter()
        grade = grade_fn(tasks_by_id[task_id], response)
        wall = time.perf_counter() - t0
        fp2 = grade["fingerprint"]
        row = {"task_id": task_id, "code_hash": chash, "second_fingerprint": fp2,
               "second_max_test_wall_s": grade.get("max_test_wall_s", 0.0),
               "second_wall_s": round(wall, 3), "first_pass": g["first_pass"]}
        results.append(row)
        for cond, stage, fp1 in g["first_pass"]:
            if fp1 != fp2:
                disagreements.append({"task_id": task_id, "code_hash": chash, "condition": cond,
                                      "stage": stage, "first": fp1, "second": fp2,
                                      "second_max_test_wall_s": row["second_max_test_wall_s"]})
        if verbose:
            mark = "" if all(fp == fp2 for _, _, fp in g["first_pass"]) else "  <-- DISAGREES"
            print(f"[{idx}/{len(groups)}] {task_id} {chash}: {fp2} ({wall:.1f}s){mark}", flush=True)

    n_records = sum(len(r["first_pass"]) for r in results)
    n_prog_dis = len({(d["task_id"], d["code_hash"]) for d in disagreements})
    return {"n_programs": len(results), "n_first_pass_records": n_records,
            "n_programs_disagreeing": n_prog_dis, "n_records_disagreeing": len(disagreements),
            "disagreements": disagreements, "results": results}


def print_summary(rep: dict) -> None:
    print("\n" + "=" * 72)
    print("E3 EXECUTION STABILITY AUDIT (exploratory; first pass stays primary)")
    print("=" * 72)
    print(f"Distinct programs re-graded:      {rep['n_programs']}")
    print(f"First-pass records compared:      {rep['n_first_pass_records']}")
    print(f"Programs with a disagreement:     {rep['n_programs_disagreeing']} / {rep['n_programs']}")
    print(f"Records with a disagreement:      {rep['n_records_disagreeing']} / {rep['n_first_pass_records']}")
    for d in rep["disagreements"]:
        print(f"  {d['task_id']} {d['condition']}/{d['stage']} {d['code_hash']}: "
              f"{d['first']} -> {d['second']} (max test {d['second_max_test_wall_s']}s)")


def main(argv: Optional[list] = None):
    args = parse_args() if argv is None else argparse.Namespace(**argv)
    from code_exec import is_docker_available
    if not is_docker_available():
        sys.exit("ERROR: Docker is not available; E3 must run in the Docker sandbox.")
    if os.environ.get("UNSAFE_LOCAL_EXEC") == "1":
        sys.exit("ERROR: unset UNSAFE_LOCAL_EXEC; E3 must run in the Docker sandbox.")

    from tasks.livecodebench_tasks import grade_lcb, load_lcb_tasks
    entries = load_programs(args.programs)
    if not entries:
        sys.exit(f"ERROR: no programs in {args.programs}")
    ids = sorted({e["task_id"] for e in entries})
    tasks_by_id = {t.task_id: t for t in load_lcb_tasks(problem_ids=ids, raw_dir=args.raw_dir)}
    print(f"Loaded {len(entries)} programs.jsonl entries over {len(ids)} problems")

    rep = audit(entries, tasks_by_id, grade_lcb)
    print_summary(rep)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(f"\nSaved report to {args.out}")


if __name__ == "__main__":
    main()
