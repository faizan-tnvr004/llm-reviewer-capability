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
  (default)         full 5-condition PER experiment on pre-registered tasks.

The --calibrate flag exists so the agent can probe different difficulty
settings BEFORE the pre-registration commit.  Once the commit is made
and pushed, only the default mode is used.
"""

import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from tasks.livecodebench_tasks import (
    load_lcb_tasks, grade_lcb, extract_program,
    save_problem_ids, load_problem_ids,
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

RUN_DIR = "run_code_data"
PROBLEM_IDS_FILE = os.path.join(RUN_DIR, "problem_ids.json")


# ---------------------------------------------------------------------------
# Code-specific review input: strip the executor instruction from the problem
# shown to reviewers, and embed the extracted program (not the full response)
# ---------------------------------------------------------------------------

def code_review_input(task, a0_text):
    """Prepare the {problem} and {solution} fields for the code review prompt."""
    # Strip the executor-facing answer instruction from the problem
    problem = task.prompt
    marker = "Write a complete Python 3 program"
    idx = problem.find(marker)
    if idx != -1:
        problem = problem[:idx].rstrip()

    # Extract just the code for the reviewer (not the full reasoning trace)
    code = extract_program(a0_text)
    solution = code if code else a0_text[-3000:]  # fallback: tail of response

    return {"problem": problem, "solution": solution}


def code_revise_input(task, a0_text, critique):
    """Prepare {problem}, {solution}, {critique} for the code revision prompt."""
    code = extract_program(a0_text)
    solution = code if code else a0_text[-3000:]
    return {"problem": task.prompt, "solution": solution, "critique": critique}


def extra_code_log(task, condition, a0_text, a0_grade, final_grade):
    """Extra log fields for code tasks: fingerprint, family, question_id."""
    return {
        "family": "livecodebench",
        "a0_fingerprint": a0_grade.get("fingerprint", ""),
        "final_fingerprint": final_grade.get("fingerprint", ""),
        "question_id": task.meta.get("question_id", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="Code extension experiment")
    parser.add_argument("--calibrate", type=int, default=0,
                        help="Run N tasks executor-only for difficulty calibration")
    parser.add_argument("--difficulty", nargs="+", default=["medium", "hard"],
                        help="Difficulty levels to include")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for task selection")
    parser.add_argument("--n", type=int, default=100,
                        help="Number of tasks for the main run")
    args = parser.parse_args()

    os.makedirs(RUN_DIR, exist_ok=True)
    models = {m.model_id: m for m in experimental_models()}
    gemini = next(m for mid, m in models.items() if "gemini" in mid)
    gptoss = next(m for mid, m in models.items() if "gpt-oss" in mid)
    llama = next(m for mid, m in models.items() if "llama" in mid)

    # ---- Calibration mode ----
    if args.calibrate > 0:
        print(f"=== CALIBRATION: {args.calibrate} tasks, executor-only ===")
        print(f"Difficulty: {args.difficulty}, seed: {args.seed}")
        tasks = load_lcb_tasks(n=args.calibrate, difficulty=args.difficulty,
                               seed=args.seed)
        cache = DiskCache(os.path.join(RUN_DIR, "cache"))
        log = RunLog(os.path.join(RUN_DIR, "calibration_log.jsonl"))
        meter = TokenMeter(usage_hook=get_last_usage)

        # C0-only: just the executor, no reviewer
        summary = run_per_experiment(
            tasks, gemini, {"C0": None},
            cache, log, meter,
            grader=grade_lcb,
            review_prompt=CODE_REVIEW_PROMPT,
            revise_prompt=CODE_REVISE_PROMPT,
            review_input_fn=code_review_input,
            revise_input_fn=code_revise_input,
            extra_log_fn=extra_code_log,
        )
        print(f"\nCalibration summary: {summary}")

        rows = log.load()
        n_correct = sum(1 for r in rows if r["final_correct"])
        print(f"Executor accuracy: {n_correct}/{len(rows)} "
              f"= {n_correct/len(rows)*100:.0f}%")
        print(f"Target: 40-60% error rate (i.e. 40-60% accuracy)")
        return

    # ---- Main experiment ----
    # Load pre-registered problem IDs if available, otherwise load fresh
    if os.path.exists(PROBLEM_IDS_FILE):
        print(f"Loading pre-registered problem IDs from {PROBLEM_IDS_FILE}")
        problem_ids = load_problem_ids(PROBLEM_IDS_FILE)
        tasks = load_lcb_tasks(n=None, difficulty=args.difficulty,
                               seed=args.seed)
        # Filter to pre-registered IDs, preserving order
        id_set = set(problem_ids)
        tasks = [t for t in tasks if t.meta["question_id"] in id_set]
        print(f"Using {len(tasks)} pre-registered tasks")
    else:
        print(f"No pre-registration file found. Loading {args.n} tasks.")
        tasks = load_lcb_tasks(n=args.n, difficulty=args.difficulty,
                               seed=args.seed)

    executor = gemini
    reviewer_conditions = {
        "C0": None,              # no review (baseline)
        "C1": gemini,            # self-review
        "C2": gptoss,            # mid-tier cross-family
        "C3": llama,             # weak reviewer
        "C4": PublicTestReviewer(),  # public-test feedback (deterministic)
    }

    cache = DiskCache(os.path.join(RUN_DIR, "cache"))
    log = RunLog(os.path.join(RUN_DIR, "log.jsonl"))
    meter = TokenMeter(usage_hook=get_last_usage)

    print(f"\nCode PER experiment: {len(tasks)} tasks x "
          f"{len(reviewer_conditions)} conditions "
          f"(executor={executor.model_id})")
    print(f"Prompts hash: {prompts_hash()}\n")

    summary = run_per_experiment(
        tasks, executor, reviewer_conditions,
        cache, log, meter,
        grader=grade_lcb,
        review_prompt=CODE_REVIEW_PROMPT,
        revise_prompt=CODE_REVISE_PROMPT,
        review_input_fn=code_review_input,
        revise_input_fn=code_revise_input,
        extra_log_fn=extra_code_log,
    )
    print(f"\nRun summary: {summary}")

    # Persist token records
    tok_path = os.path.join(RUN_DIR, "tokens.jsonl")
    with open(tok_path, "a", encoding="utf-8") as f:
        for t in meter.records:
            f.write(json.dumps(t) + "\n")

    all_tokens = []
    if os.path.exists(tok_path):
        all_tokens = [json.loads(l) for l in open(tok_path, encoding="utf-8")
                      if l.strip()]

    m = analyze(log.load(), all_tokens)

    print("\n=== metrics by condition ===")
    for cond in sorted(m):
        x = m[cond]
        def fmt(v): return "n/a" if v is None else f"{v:.2f}"
        print(f"{cond}: n={x['n']} acc={fmt(x['accuracy'])} "
              f"Δ={fmt(x['delta_vs_C0'])} recall={fmt(x['recall'])} "
              f"FRR={fmt(x['false_rejection_rate'])} repair={fmt(x['repair_rate'])} "
              f"damage={fmt(x['damage_rate'])} malformed={fmt(x['malformed_review_rate'])} "
              f"tok/task={fmt(x.get('mean_tokens_per_task'))}")


if __name__ == "__main__":
    main()
