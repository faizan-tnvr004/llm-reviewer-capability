"""MAIN EXPERIMENT: reviewer-downgrade study on Omni-MATH.

Run from repo root:   python run_experiment.py
Resumable: re-run after any daily-limit stop; caching skips completed work.

Start with N_TASKS = 15 as a SMOKE RUN. Read the printed metrics and a few
transcripts. Only after the smoke run looks sane, raise N_TASKS to 100 for the
full run (the cache means the first 15 are never re-paid).
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from tasks.omnimath_tasks import load_omnimath_tasks
from harness import DiskCache, RunLog
from pipeline import run_per_experiment, analyze, TokenMeter
from providers import experimental_models, get_last_usage

N_TASKS = 100         # SMOKE RUN SIZE. Raise to 100 for the full run.
MIN_DIFF, MAX_DIFF = 5.0, 6.0
RUN_DIR = "run_experiment_data"


def main():
    tasks = load_omnimath_tasks(n=N_TASKS, min_difficulty=MIN_DIFF,
                                max_difficulty=MAX_DIFF, seed=0)
    models = {m.model_id: m for m in experimental_models()}
    gemini = next(m for mid, m in models.items() if "gemini" in mid)
    gptoss = next(m for mid, m in models.items() if "gpt-oss" in mid)
    llama = next(m for mid, m in models.items() if "llama" in mid)

    executor = gemini
    reviewer_conditions = {
        "C0": None,        # no review (baseline)
        "C1": gemini,      # strong reviewer
        "C2": gptoss,      # mid reviewer
        "C3": llama,       # weak reviewer (0% solver)
    }

    cache = DiskCache(os.path.join(RUN_DIR, "cache"))
    log = RunLog(os.path.join(RUN_DIR, "log.jsonl"))
    meter = TokenMeter(usage_hook=get_last_usage)

    print(f"Reviewer-downgrade experiment: {len(tasks)} tasks x "
          f"{len(reviewer_conditions)} conditions "
          f"(executor={executor.model_id})\n")

    summary = run_per_experiment(tasks, executor, reviewer_conditions,
                                 cache, log, meter)
    print("\nrun summary:", summary)

    # persist token records (append; analysis dedupes by nothing — records are
    # only produced for calls actually made this session, cache hits cost 0)
    tok_path = os.path.join(RUN_DIR, "tokens.jsonl")
    with open(tok_path, "a", encoding="utf-8") as f:
        for t in meter.records:
            f.write(json.dumps(t) + "\n")

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

    print("\n(smoke run: read a few transcripts in "
          f"{RUN_DIR}/log.jsonl before scaling to 100)")


if __name__ == "__main__":
    main()
