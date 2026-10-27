"""MAIN EXPERIMENT: code-extension study on LiveCodeBench.

Run from repo root:   python run_experiment_code.py
Resumable: re-run after any daily-limit stop; caching skips completed work.

Conditions:
  C0  no review (baseline)
  C1  self-review (Gemini reviews Gemini)
  C2  cross-family mid-tier review (gpt-oss-20b)
  C3  weak review (Llama-3.1-8B Instruct, served by DeepInfra)
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
import atexit
import hashlib
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
from code_exec import code_hash, cleanup_stale_containers, sandbox_preflight
from harness import DiskCache, RunLog, Model
from pipeline import run_per_experiment, analyze, TokenMeter
try:
    from providers import code_experimental_models, get_last_usage, preflight_code_providers
except (ImportError, ModuleNotFoundError):
    code_experimental_models = None
    get_last_usage = None
    preflight_code_providers = None

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


def _prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def load_retried_prompts(retry_log_path: str) -> set:
    """sha256 of every review prompt that was retried after an empty response."""
    if not retry_log_path or not os.path.exists(retry_log_path):
        return set()
    out = set()
    with open(retry_log_path, encoding="utf-8") as f:
        for line in f:
            try:
                out.add(json.loads(line)["prompt_sha256"])
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def make_extra_code_log(programs_log_path: str = os.path.join(RUN_DIR, "programs.jsonl"),
                        retry_log_path: str | None = None,
                        retried_conditions: tuple = ("C2",)):
    """Factory for extra_code_log with a configurable programs.jsonl path.
    review_retried is True for a retried condition's row when its review prompt
    appears in retry_log_path (written by wrap_throttled_model)."""
    def _extra_log_fn(task, condition, a0_text, a0_grade, final_grade, final_text=None):
        review_retried = False
        if retry_log_path and condition in retried_conditions:
            prompt = CODE_REVIEW_PROMPT.format(**code_review_input(task, a0_text))
            review_retried = _prompt_sha256(prompt) in load_retried_prompts(retry_log_path)
        a0_code = extract_program(a0_text)
        a0_pub_pass = public_check(task, a0_code)["all_pass"] if a0_code else False
        final_code = extract_program(final_text) if final_text is not None else a0_code

        # final_public_pass: a program passing every hidden test passes the samples;
        # an unchanged program inherits A0's result; otherwise run the samples.
        if final_grade.get("correct"):
            final_pub_pass = True
        elif final_grade.get("extracted") == a0_grade.get("extracted"):
            final_pub_pass = a0_pub_pass
        else:
            final_pub_pass = public_check(task, final_code)["all_pass"] if final_code else False

        # Log to programs.jsonl
        try:
            os.makedirs(os.path.dirname(programs_log_path) or ".", exist_ok=True)
            with open(programs_log_path, "a", encoding="utf-8") as pf:
                entry = {
                    "task_id": task.task_id,
                    "condition": condition,
                    "stage": "final",
                    "a0_code": a0_code,
                    "final_code": final_code,
                    "a0_fingerprint": a0_grade.get("fingerprint", ""),
                    "final_fingerprint": final_grade.get("fingerprint", ""),
                    "a0_code_hash": a0_grade.get("extracted"),
                    "a0_ast_hash": a0_grade.get("ast_hash"),
                    "final_code_hash": final_grade.get("extracted"),
                    "final_ast_hash": final_grade.get("ast_hash"),
                    "a0_max_test_wall_s": a0_grade.get("max_test_wall_s", 0.0),
                    "final_max_test_wall_s": final_grade.get("max_test_wall_s", 0.0),
                    "max_test_wall_s": final_grade.get("max_test_wall_s", 0.0),
                    "a0_infra_retries": a0_grade.get("infra_retries", 0),
                    "final_infra_retries": final_grade.get("infra_retries", 0),
                    "infra_retries": final_grade.get("infra_retries", 0),
                    "ts": time.time(),
                }
                pf.write(json.dumps(entry) + "\n")
        except Exception:
            pass

        return {
            "family": "livecodebench",
            "question_id": task.meta.get("question_id", ""),
            "difficulty": task.meta.get("difficulty", ""),
            "contest_date": task.meta.get("contest_date", ""),
            "a0_fingerprint": a0_grade.get("fingerprint", ""),
            "final_fingerprint": final_grade.get("fingerprint", ""),
            "a0_code_hash": a0_grade.get("extracted"),
            "final_code_hash": final_grade.get("extracted"),
            "a0_ast_hash": a0_grade.get("ast_hash"),
            "final_ast_hash": final_grade.get("ast_hash"),
            "a0_max_test_wall_s": a0_grade.get("max_test_wall_s", 0.0),
            "final_max_test_wall_s": final_grade.get("max_test_wall_s", 0.0),
            "max_test_wall_s": final_grade.get("max_test_wall_s", 0.0),
            "a0_infra_retries": a0_grade.get("infra_retries", 0),
            "final_infra_retries": final_grade.get("infra_retries", 0),
            "infra_retries": final_grade.get("infra_retries", 0),
            "a0_public_pass": a0_pub_pass,
            "final_public_pass": final_pub_pass,
            "review_retried": review_retried,
        }
    return _extra_log_fn

extra_code_log = make_extra_code_log()


# ---------------------------------------------------------------------------
# Rate-limiting and Retry Wrapper for gpt-oss
# ---------------------------------------------------------------------------

def wrap_throttled_model(base_model: Model, min_interval_s: float = GPTOSS_MIN_INTERVAL_S,
                         retry_empty: bool = RETRY_EMPTY_REVIEW,
                         retry_log_path: str | None = None,
                         sleep=time.sleep) -> Model:
    """Throttle calls and optionally retry empty reviews once after 60 seconds.
    Each retry is appended to retry_log_path (prompt sha256, whether the retry
    was also empty) before the retry is made."""
    last_call_time = 0.0

    def throttled_call(prompt: str) -> str:
        nonlocal last_call_time
        elapsed = time.time() - last_call_time
        if elapsed < min_interval_s:
            sleep_needed = min_interval_s - elapsed
            sleep(sleep_needed)

        last_call_time = time.time()
        resp = base_model.call(prompt)

        # Optional single retry on empty review
        if retry_empty and (not resp or not resp.strip()):
            sleep(60)
            last_call_time = time.time()
            retry_resp = base_model.call(prompt)
            if retry_log_path:
                os.makedirs(os.path.dirname(retry_log_path) or ".", exist_ok=True)
                with open(retry_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"model_id": base_model.model_id,
                                        "prompt_sha256": _prompt_sha256(prompt),
                                        "retry_empty": not (retry_resp and retry_resp.strip()),
                                        "ts": time.time()}) + "\n")
            if retry_resp and retry_resp.strip():
                return retry_resp

        return resp

    return Model(base_model.model_id, throttled_call)


# ---------------------------------------------------------------------------
# Calibration execution & logging
# ---------------------------------------------------------------------------

def run_calibration(n: int, difficulties: list[str], date_from: str | None,
                    date_to: str | None, seed: int, batch_idx: int = 1,
                    run_dir: str = RUN_DIR):
    calib_dir = os.path.join(run_dir, "calibration")
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
        raw_dir=os.path.join(run_dir, "raw"),
    )
    if len(tasks) < n:
        print(f"Warning: Only {len(tasks)} candidate problems found matching filter (requested {n})")

    models = {m.model_id: m for m in code_experimental_models()}
    gemini = next(m for mid, m in models.items() if "gemini" in mid)

    calib_cache = DiskCache(os.path.join(run_dir, "calibration_cache"))
    calib_log = RunLog(os.path.join(calib_dir, f"log_batch_{batch_idx}.jsonl"))
    meter = TokenMeter(usage_hook=get_last_usage)
    calib_extra_log = make_extra_code_log(os.path.join(run_dir, "calibration_programs.jsonl"))

    summary = run_per_experiment(
        tasks, gemini, {"C0": None},
        calib_cache, calib_log, meter,
        grader=grade_lcb,
        review_prompt=CODE_REVIEW_PROMPT,
        revise_prompt=CODE_REVISE_PROMPT,
        review_input_fn=code_review_input,
        revise_input_fn=code_revise_input,
        extra_log_fn=calib_extra_log,
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
    calib_md_path = os.path.join(run_dir, "CALIBRATION.md")
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
    """Fail closed: any exception or error running git status means refuse to start."""
    try:
        res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
        # Untracked calibration / scratch files don't violate clean committed state of code
        # But tracked files must have no uncommitted modifications
        lines = [l for l in res.stdout.strip().split("\n") if l.strip() and not l.startswith("??")]
        return len(lines) == 0
    except Exception as e:
        print(f"ERROR: git status check failed ({e}) — refusing to start (fail-closed).")
        return False


def append_run_manifest(manifest_path: str | None = None, ids_file: str | None = None,
                        run_dir: str = RUN_DIR) -> dict:
    """At every start, including resumes, append to run_code_data/run_manifest.json:
    UTC time, git rev-parse HEAD, git describe --tags --always, git status --porcelain,
    prompts_hash(), sha256 of problem_ids.json and the docker image digest in use.
    Append, never overwrite.
    """
    import datetime
    import hashlib
    from code_exec import DOCKER_IMAGE

    if manifest_path is None:
        manifest_path = os.path.join(run_dir, "run_manifest.json")
    if ids_file is None:
        ids_file = os.path.join(run_dir, "problem_ids.json")

    utc_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    describe = subprocess.run(["git", "describe", "--tags", "--always"], capture_output=True, text=True, check=False).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False).stdout.strip()
    p_hash = prompts_hash()

    prob_sha = ""
    if os.path.exists(ids_file):
        with open(ids_file, "rb") as f:
            prob_sha = hashlib.sha256(f.read()).hexdigest()

    record = {
        "utc_time": utc_time,
        "git_commit": rev,
        "git_rev_parse_head": rev,
        "git_describe": describe,
        "git_status_porcelain": status,
        "prompts_hash": p_hash,
        "problem_ids_sha256": prob_sha,
        "docker_image": DOCKER_IMAGE,
        "docker_image_digest": DOCKER_IMAGE,
    }

    entries = []
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    loaded = json.loads(content)
                    entries = loaded if isinstance(loaded, list) else [loaded]
        except Exception:
            entries = []

    entries.append(record)
    parent_dir = os.path.dirname(manifest_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

    return record


# ---------------------------------------------------------------------------
# Runner Lock & Process Liveness
# ---------------------------------------------------------------------------

def is_pid_alive(pid: int) -> bool:
    """Return True if a process with `pid` is currently running."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        SYNCHRONIZE = 0x00100000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, pid)
        if not handle:
            err = kernel32.GetLastError()
            if err == 5:  # ERROR_ACCESS_DENIED: process exists but cannot be inspected
                return True
            return False
        STILL_ACTIVE = 259
        exit_code = ctypes.c_ulong()
        success = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        kernel32.CloseHandle(handle)
        return bool(success and exit_code.value == STILL_ACTIVE)
    else:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True


class RunnerLock:
    def __init__(self, lock_path: str):
        self.lock_path = lock_path
        self.acquired = False

    def acquire(self) -> bool:
        parent = os.path.dirname(self.lock_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        if os.path.exists(self.lock_path):
            existing_pid = None
            try:
                with open(self.lock_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    existing_pid = int(content)
            except Exception:
                existing_pid = None

            if existing_pid is not None and is_pid_alive(existing_pid):
                return False  # Existing runner is active: refuse to start

            # Stale lock: dead PID or invalid content
            try:
                os.remove(self.lock_path)
            except OSError:
                pass

        my_pid = os.getpid()
        tmp_path = f"{self.lock_path}.{my_pid}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(f"{my_pid}\n")
            os.replace(tmp_path, self.lock_path)
            self.acquired = True
            return True
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            return False

    def release(self):
        if self.acquired:
            try:
                if os.path.exists(self.lock_path):
                    with open(self.lock_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    if content == str(os.getpid()):
                        os.remove(self.lock_path)
            except OSError:
                pass
            self.acquired = False

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError(f"Another runner is active (lock file: {self.lock_path})")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


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
    parser.add_argument("--run-dir", default=RUN_DIR,
                        help=f"Directory for run artifacts, logs, cache (default: {RUN_DIR})")
    parser.add_argument("--ids-file", default=None,
                        help="Path to problem IDs JSON file (default: <run-dir>/problem_ids.json)")
    args = parser.parse_args()

    run_dir = args.run_dir
    ids_file = args.ids_file or os.path.join(run_dir, "problem_ids.json")
    os.makedirs(run_dir, exist_ok=True)

    # ---- Mode 1: Calibration ----
    if args.calibrate > 0:
        run_calibration(
            n=args.calibrate,
            difficulties=args.difficulty,
            date_from=args.date_from,
            date_to=args.date_to,
            seed=args.seed,
            batch_idx=args.batch,
            run_dir=run_dir,
        )
        return

    # ---- Mode 2: Main experiment ----
    if not os.path.exists(ids_file):
        print(f"ERROR: Main experiment refuses to start because {ids_file} does not exist.")
        print("Pre-registration commit (Task T14) must exist before running main experiment.")
        sys.exit(1)

    if not is_git_clean():
        print("ERROR: Main experiment refuses to start because git working tree has uncommitted modifications.")
        print("Commit all code before starting the main experiment.")
        sys.exit(1)

    problems = sandbox_preflight() + preflight_code_providers()
    if problems:
        print("ERROR: Main experiment refuses to start; preflight failed:")
        for p_ in problems:
            print(f"  - {p_}")
        sys.exit(1)

    lock = RunnerLock(os.path.join(run_dir, ".runner.lock"))
    if not lock.acquire():
        print(f"ERROR: Another runner is active (lock file: {lock.lock_path}). Refusing to start.")
        sys.exit(1)

    atexit.register(lock.release)

    try:
        manifest_entry = append_run_manifest(
            manifest_path=os.path.join(run_dir, "run_manifest.json"),
            ids_file=ids_file,
            run_dir=run_dir,
        )
        print(f"Recorded run manifest entry (UTC {manifest_entry['utc_time']}, commit {manifest_entry['git_commit'][:8]})")

        print("Running startup sweep for stale code_exec_* Docker containers...")
        n_swept = cleanup_stale_containers()
        if n_swept == 0:
            print("Startup sweep clean: 0 stale containers found.")

        print(f"Loading pre-registered problem IDs from {ids_file}")
        problem_ids = load_problem_ids(ids_file)
        tasks = load_lcb_tasks(
            problem_ids=problem_ids,
            raw_dir=os.path.join(run_dir, "raw"),
        )
        print(f"Loaded {len(tasks)} pre-registered tasks")

        models = {m.model_id: m for m in code_experimental_models()}
        gemini = next(m for mid, m in models.items() if "gemini" in mid)
        gptoss = next(m for mid, m in models.items() if "gpt-oss" in mid)
        llama = next(m for mid, m in models.items() if "llama" in mid)

        # Wrap gpt-oss with throttling and empty retry
        retry_log_path = os.path.join(run_dir, "review_retries.jsonl")
        throttled_gptoss = wrap_throttled_model(gptoss, min_interval_s=GPTOSS_MIN_INTERVAL_S,
                                               retry_empty=RETRY_EMPTY_REVIEW,
                                               retry_log_path=retry_log_path)

        reviewer_conditions = {
            "C0": None,                   # no review (baseline)
            "C1": gemini,                 # self-review
            "C2": throttled_gptoss,       # mid-tier cross-family (throttled)
            "C3": llama,                  # weak reviewer
            "C4": PublicTestReviewer(),   # public-test feedback (deterministic)
        }

        cache = DiskCache(os.path.join(run_dir, "cache"))
        log = RunLog(os.path.join(run_dir, "log.jsonl"))
        meter = TokenMeter(usage_hook=get_last_usage)
        programs_path = os.path.join(run_dir, "programs.jsonl")
        extra_log = make_extra_code_log(programs_path, retry_log_path=retry_log_path)

        print(f"\nCode PER experiment: {len(tasks)} tasks x {len(reviewer_conditions)} conditions")
        print(f"Executor: {gemini.model_id}")
        print(f"Prompts hash: {prompts_hash()}\n")

        try:
            summary = run_per_experiment(
                tasks, gemini, reviewer_conditions,
                cache, log, meter,
                grader=grade_lcb,
                review_prompt=CODE_REVIEW_PROMPT,
                revise_prompt=CODE_REVISE_PROMPT,
                review_input_fn=code_review_input,
                revise_input_fn=code_revise_input,
                extra_log_fn=extra_log,
            )
            print(f"\nRun summary: {summary}")
        finally:
            # Written even when an unexpected error stops the run, so the
            # provider usage of every call already made is kept.
            tok_path = os.path.join(run_dir, "tokens.jsonl")
            with open(tok_path, "a", encoding="utf-8") as f:
                for t in meter.records:
                    f.write(json.dumps(t) + "\n")
    finally:
        lock.release()


if __name__ == "__main__":
    main()
