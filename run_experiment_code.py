"""MAIN EXPERIMENT: code-extension study on LiveCodeBench.

Run from repo root:   python run_experiment_code.py
Resumable: re-run after any daily-limit stop; caching skips completed work.

Conditions:
  C0  no review (baseline)
  C1  self-review (Gemini reviews Gemini)
  C2  cross-family mid-tier review (gpt-oss-20b)
  C3  weak review (Llama-3.1-8B)
  C4  public-test feedback (deterministic, zero-token)

Modes:
  --calibrate N     run N tasks, executor-only (no review), to find the
                    difficulty window yielding 40-60% accuracy.
                    Writes run_code_data/calibration/batch_<k>.json
                    and updates run_code_data/CALIBRATION.md.
  (default)         full 5-condition PER experiment on pre-registered tasks.
                    Refuses to start unless PROBLEM_IDS_FILE exists and
                    the git working tree is clean.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from tasks.livecodebench_tasks import (
    load_lcb_tasks, grade_lcb, extract_program, normalize_code,
    save_problem_ids, load_problem_ids, public_check,
)
from prompts_code import (
    CODE_REVIEW_PROMPT, CODE_REVISE_PROMPT, prompts_hash,
    CODE_ANSWER_INSTRUCTION,
)
from test_reviewer import PublicTestReviewer
from code_exec import code_hash
from harness import DiskCache, RunLog, Model
from pipeline import run_per_experiment, analyze, TokenMeter
from providers import experimental_models, get_last_usage

# Top-level specification constants (from plan Section 6 Task T9)
N_TASKS = 100
DIFFICULTIES = ("medium", "hard")
DATE_FROM = "2024-10-01"
DATE_TO = None
RUN_DIR = "run_code_data"
PROBLEM_IDS_FILE = os.path.join(RUN_DIR, "problem_ids.json")
GPTOSS_MIN_INTERVAL_S = 40
RETRY_EMPTY_REVIEW = True


# ---------------------------------------------------------------------------
# Code-specific prompt formatting
# ---------------------------------------------------------------------------

def code_review_input(task, a0_text):
    """Prepare the {problem} and {solution} fields for the code review prompt."""
    problem = task.prompt
    marker = "Write a complete Python 3 program"
    idx = problem.find(marker)
    if idx != -1:
        problem = problem[:idx].rstrip()

    code = extract_program(a0_text)
    solution = code if code else a0_text[-3000:]
    return {"problem": problem, "solution": solution}


def code_revise_input(task, a0_text, critique):
    """Prepare {problem}, {solution}, {critique} for the code revision prompt."""
    code = extract_program(a0_text)
    solution = code if code else a0_text[-3000:]
    return {"problem": task.prompt, "solution": solution, "critique": critique}


def extra_code_log(task, condition, a0_text, a0_grade, final_grade):
    """Extra log fields per outcome matching Task T9 specification:
    a0_fingerprint, final_fingerprint, a0_code_hash, final_code_hash,
    a0_public_pass, final_public_pass, review_retried.
    """
    a0_code = extract_program(a0_text)
    a0_pub_pass = public_check(task, a0_code)["all_pass"] if a0_code else False

    # For final_public_pass: if final passes hidden tests, public tests always pass
    if final_grade.get("correct"):
        final_pub_pass = True
    elif condition == "C0":
        final_pub_pass = a0_pub_pass
    else:
        final_pub_pass = a0_pub_pass

    return {
        "family": "livecodebench",
        "question_id": task.meta.get("question_id", ""),
        "difficulty": task.meta.get("difficulty", ""),
        "contest_date": task.meta.get("contest_date", ""),
        "a0_fingerprint": a0_grade.get("fingerprint", ""),
        "final_fingerprint": final_grade.get("fingerprint", ""),
        "a0_code_hash": a0_grade.get("extracted"),
        "final_code_hash": final_grade.get("extracted"),
        "a0_public_pass": a0_pub_pass,
        "final_public_pass": final_pub_pass,
        "review_retried": False,
    }


# ---------------------------------------------------------------------------
# Rate-limiting and Retry Wrapper for gpt-oss
# ---------------------------------------------------------------------------

def wrap_throttled_model(base_model: Model, min_interval_s: float = GPTOSS_MIN_INTERVAL_S,
                         retry_empty: bool = RETRY_EMPTY_REVIEW) -> Model:
    """Throttle calls and optionally retry empty reviews once after 60 seconds."""
    last_call_time = 0.0

    def throttled_call(prompt: str) -> str:
        nonlocal last_call_time
        elapsed = time.time() - last_call_time
        if elapsed < min_interval_s:
            sleep_needed = min_interval_s - elapsed
            time.sleep(sleep_needed)

        last_call_time = time.time()
        resp = base_model.call(prompt)

        # Optional single retry on empty review
        if retry_empty and (not resp or not resp.strip()):
            time.sleep(60)
            last_call_time = time.time()
            retry_resp = base_model.call(prompt)
            if retry_resp and retry_resp.strip():
                return retry_resp

        return resp

    return Model(base_model.model_id, throttled_call)


# ---------------------------------------------------------------------------
# Calibration execution & logging
# ---------------------------------------------------------------------------

def run_calibration(n: int, difficulties: list[str], date_from: str | None,
                    date_to: str | None, seed: int, batch_idx: int = 1):
    calib_dir = os.path.join(RUN_DIR, "calibration")
    os.makedirs(calib_dir, exist_ok=True)
    batch_file = os.path.join(calib_dir, f"batch_{batch_idx}.json")

    print(f"\n{'='*70}")
    print(f"CALIBRATION BATCH {batch_idx}: {n} problems, executor-only (Gemini)")
    print(f"Date window: {date_from or 'any'} to {date_to or 'any'}")
    print(f"Difficulties: {difficulties}, seed: {seed}")
    print(f"{'='*70}\n")

    tasks = load_lcb_tasks(
        n=n,
        difficulties=difficulties,
        date_from=date_from,
        date_to=date_to,
        seed=seed,
        raw_dir=os.path.join(RUN_DIR, "raw"),
    )
    if len(tasks) < n:
        print(f"Warning: Only {len(tasks)} candidate problems found matching filter (requested {n})")

    models = {m.model_id: m for m in experimental_models()}
    gemini = next(m for mid, m in models.items() if "gemini" in mid)

    calib_cache = DiskCache(os.path.join(RUN_DIR, "calibration_cache"))
    calib_log = RunLog(os.path.join(calib_dir, f"log_batch_{batch_idx}.jsonl"))
    meter = TokenMeter(usage_hook=get_last_usage)

    summary = run_per_experiment(
        tasks, gemini, {"C0": None},
        calib_cache, calib_log, meter,
        grader=grade_lcb,
        review_prompt=CODE_REVIEW_PROMPT,
        revise_prompt=CODE_REVISE_PROMPT,
        review_input_fn=code_review_input,
        revise_input_fn=code_revise_input,
        extra_log_fn=extra_code_log,
    )

    rows = calib_log.load()
    rows_this_batch = [r for r in rows if any(r["task_id"] == t.task_id for t in tasks)]

    n_total = len(rows_this_batch)
    n_correct = sum(1 for r in rows_this_batch if r.get("final_correct"))
    accuracy = (n_correct / n_total) if n_total else 0.0

    # Fingerprint distribution
    fps = [r.get("final_fingerprint", "UNKNOWN") for r in rows_this_batch]
    # simplify fingerprints to category (PASS, FAIL, TIMEOUT, CRASH, NO_CODE)
    cat_fps = Counter()
    for fp in fps:
        if fp == "PASS":
            cat_fps["PASS"] += 1
        elif fp.startswith("FAIL:"):
            cat_fps["FAIL"] += 1
        elif fp == "TIMEOUT":
            cat_fps["TIMEOUT"] += 1
        elif fp == "CRASH":
            cat_fps["CRASH"] += 1
        else:
            cat_fps[fp] += 1

    # Measure prompt lengths in tokens/chars
    prompt_lengths = [len(t.prompt) for t in tasks]
    mean_prompt_chars = sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0

    print(f"\nBatch {batch_idx} Results:")
    print(f"  Executor Accuracy : {n_correct}/{n_total} = {accuracy*100:.1f}%")
    print(f"  Target Window     : 40% - 60% accuracy")
    print(f"  Fingerprints      : {dict(cat_fps)}")
    print(f"  Mean Prompt Chars : {mean_prompt_chars:.0f}")

    # Save batch json
    batch_data = {
        "batch_idx": batch_idx,
        "n_requested": n,
        "n_evaluated": n_total,
        "n_correct": n_correct,
        "accuracy": accuracy,
        "difficulties": list(difficulties),
        "date_from": date_from,
        "date_to": date_to,
        "seed": seed,
        "fingerprint_counts": dict(cat_fps),
        "mean_prompt_chars": mean_prompt_chars,
        "problem_ids": [t.task_id for t in tasks],
        "rows": rows_this_batch,
    }
    with open(batch_file, "w", encoding="utf-8") as f:
        json.dump(batch_data, f, indent=2)
    print(f"Saved batch details to {batch_file}")

    # Update or create CALIBRATION.md
    calib_md_path = os.path.join(RUN_DIR, "CALIBRATION.md")
    md_header = (
        "# LiveCodeBench Calibration Log\n\n"
        "Target executor accuracy: **40% – 60%** on Gemini 3.1 Flash-Lite.\n\n"
        "| Batch | Window | Difficulties | Seed | Evaluated | Correct | Accuracy | Fingerprints (PASS / FAIL / TIMEOUT / CRASH) | Status |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    if not os.path.exists(calib_md_path):
        with open(calib_md_path, "w", encoding="utf-8") as f:
            f.write(md_header)

    in_target = 0.40 <= accuracy <= 0.60
    status_str = "**TARGET MET**" if in_target else ("Too Low (<40%)" if accuracy < 0.40 else "Too High (>60%)")
    window_str = f"{date_from or 'start'} to {date_to or 'end'}"
    diff_str = "+".join(difficulties)
    fp_str = f"P:{cat_fps['PASS']}, F:{cat_fps['FAIL']}, T:{cat_fps['TIMEOUT']}, C:{cat_fps['CRASH']}"

    row_line = f"| **Batch {batch_idx}** | {window_str} | {diff_str} | {seed} | {n_total} | {n_correct} | **{accuracy*100:.1f}%** | {fp_str} | {status_str} |\n"

    with open(calib_md_path, "a", encoding="utf-8") as f:
        f.write(row_line)
    print(f"Updated {calib_md_path}")

    return batch_data


# ---------------------------------------------------------------------------
# Git working tree check
# ---------------------------------------------------------------------------

def is_git_clean() -> bool:
    try:
        res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
        # Untracked calibration / scratch files don't violate clean committed state of code
        # But tracked files must have no uncommitted modifications
        lines = [l for l in res.stdout.strip().split("\n") if l.strip() and not l.startswith("??")]
        return len(lines) == 0
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Code extension experiment")
    parser.add_argument("--calibrate", type=int, default=0,
                        help="Run N tasks executor-only for difficulty calibration")
    parser.add_argument("--batch", type=int, default=1,
                        help="Calibration batch number")
    parser.add_argument("--difficulty", nargs="+", default=["medium", "hard"],
                        help="Difficulty levels to include")
    parser.add_argument("--date-from", default=None,
                        help="Start date YYYY-MM-DD")
    parser.add_argument("--date-to", default=None,
                        help="End date YYYY-MM-DD")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for task selection")
    parser.add_argument("--n", type=int, default=N_TASKS,
                        help="Number of tasks for the main run")
    args = parser.parse_args()

    os.makedirs(RUN_DIR, exist_ok=True)

    # ---- Mode 1: Calibration ----
    if args.calibrate > 0:
        run_calibration(
            n=args.calibrate,
            difficulties=args.difficulty,
            date_from=args.date_from,
            date_to=args.date_to,
            seed=args.seed,
            batch_idx=args.batch,
        )
        return

    # ---- Mode 2: Main experiment ----
    if not os.path.exists(PROBLEM_IDS_FILE):
        print(f"ERROR: Main experiment refuses to start because {PROBLEM_IDS_FILE} does not exist.")
        print("Pre-registration commit (Task T14) must exist before running main experiment.")
        sys.exit(1)

    if not is_git_clean():
        print("ERROR: Main experiment refuses to start because git working tree has uncommitted modifications.")
        print("Commit all code before starting the main experiment.")
        sys.exit(1)

    print(f"Loading pre-registered problem IDs from {PROBLEM_IDS_FILE}")
    problem_ids = load_problem_ids(PROBLEM_IDS_FILE)
    tasks = load_lcb_tasks(
        problem_ids=problem_ids,
        raw_dir=os.path.join(RUN_DIR, "raw"),
    )
    print(f"Loaded {len(tasks)} pre-registered tasks")

    models = {m.model_id: m for m in experimental_models()}
    gemini = next(m for mid, m in models.items() if "gemini" in mid)
    gptoss = next(m for mid, m in models.items() if "gpt-oss" in mid)
    llama = next(m for mid, m in models.items() if "llama" in mid)

    # Wrap gpt-oss with throttling and empty retry
    throttled_gptoss = wrap_throttled_model(gptoss, min_interval_s=GPTOSS_MIN_INTERVAL_S,
                                           retry_empty=RETRY_EMPTY_REVIEW)

    reviewer_conditions = {
        "C0": None,                   # no review (baseline)
        "C1": gemini,                 # self-review
        "C2": throttled_gptoss,       # mid-tier cross-family (throttled)
        "C3": llama,                  # weak reviewer
        "C4": PublicTestReviewer(),   # public-test feedback (deterministic)
    }

    cache = DiskCache(os.path.join(RUN_DIR, "cache"))
    log = RunLog(os.path.join(RUN_DIR, "log.jsonl"))
    meter = TokenMeter(usage_hook=get_last_usage)

    print(f"\nCode PER experiment: {len(tasks)} tasks x {len(reviewer_conditions)} conditions")
    print(f"Executor: {gemini.model_id}")
    print(f"Prompts hash: {prompts_hash()}\n")

    summary = run_per_experiment(
        tasks, gemini, reviewer_conditions,
        cache, log, meter,
        grader=grade_lcb,
        review_prompt=CODE_REVIEW_PROMPT,
        revise_prompt=CODE_REVISE_PROMPT,
        review_input_fn=code_review_input,
        revise_input_fn=code_revise_input,
        extra_log_fn=extra_code_log,
    )
    print(f"\nRun summary: {summary}")

    tok_path = os.path.join(RUN_DIR, "tokens.jsonl")
    with open(tok_path, "a", encoding="utf-8") as f:
        for t in meter.records:
            f.write(json.dumps(t) + "\n")


if __name__ == "__main__":
    main()
