"""Re-estimate token usage for calls whose exact records were lost (Amendment 1, Section 2).

The kill at 09:37:10 UTC on 20 November 2026 discarded the provider-reported
usage of every call made before it. For each such call, this script rebuilds
the exact prompt the runner sent (from the cached responses, using the
runner's own prompt-building functions) and estimates prompt and response
tokens with the runner's estimator (`pipeline._estimate_tokens`, chars/4).

A call is re-estimated only if run_code_data/tokens.jsonl has no exact record
for the same (task_id, stage, condition). Every output record has
`exact: false` and `reconstructed: true`.

Limitation: chars/4 sees only visible text, so hidden reasoning tokens, which
gpt-oss-20b produces in quantity, are not counted. Retried empty reviews count
the prompt twice and the visible retry response once.

Run from repo root:  python scripts/reconstruct_lost_tokens.py
Writes run_code_data/tokens_reconstructed.jsonl. Zero API calls.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import run_experiment_code as rec  # noqa: E402
from harness import DiskCache  # noqa: E402
from pipeline import _estimate_tokens, parse_review  # noqa: E402

RUN_DIR = "run_code_data"
EXECUTOR = "gemini-3.1-flash-lite"
REVIEWERS = {"C1": EXECUTOR, "C2": "openai/gpt-oss-20b", "C3": "llama3.1:8b-instruct-q8_0"}
OUT = os.path.join(RUN_DIR, "tokens_reconstructed.jsonl")


def main():
    rows = [json.loads(l) for l in open(os.path.join(RUN_DIR, "log.jsonl"), encoding="utf-8") if l.strip()]
    exact = {(t["task_id"], t["stage"], t["condition"])
             for t in (json.loads(l) for l in open(os.path.join(RUN_DIR, "tokens.jsonl"), encoding="utf-8") if l.strip())
             if t.get("exact")}
    retried = {json.loads(l)["prompt_sha256"]
               for l in open(os.path.join(RUN_DIR, "review_retries.jsonl"), encoding="utf-8") if l.strip()}
    cache = DiskCache(os.path.join(RUN_DIR, "cache"))
    lost_tasks = sorted({r["task_id"] for r in rows if r["utc_ts"] < "2026-11-20T09:37:10"},
                        key=[r["task_id"] for r in rows].index)
    tasks = {t.task_id: t for t in rec.load_lcb_tasks(problem_ids=lost_tasks, raw_dir=os.path.join(RUN_DIR, "raw"))}
    verdicts = {(r["task_id"], r["condition"]): r["verdict"] for r in rows}

    out = []
    for tid in lost_tasks:
        task = tasks[tid]
        a0 = cache.get("per_execute", EXECUTOR, tid, 0)
        assert a0 is not None, f"missing cached A0 for {tid}"
        if (tid, "execute", "shared") not in exact:
            out.append({"in_tokens": _estimate_tokens(task.prompt), "out_tokens": _estimate_tokens(a0),
                        "model_id": EXECUTOR, "stage": "execute", "task_id": tid, "condition": "shared"})
        for cond in ("C1", "C2", "C3", "C4"):
            review_text = None
            if cond in REVIEWERS:
                review_text = cache.get(f"per_review_{cond}", REVIEWERS[cond], tid, 0)
                prompt = rec.CODE_REVIEW_PROMPT.format(**rec.code_review_input(task, a0))
                if review_text is not None and (tid, "review", cond) not in exact:
                    # C1-C3 share the same review prompt; only C2 is ever retried.
                    was_retried = cond == "C2" and hashlib.sha256(prompt.encode("utf-8")).hexdigest() in retried
                    n_prompts = 2 if was_retried else 1
                    out.append({"in_tokens": n_prompts * _estimate_tokens(prompt), "out_tokens": _estimate_tokens(review_text),
                                "model_id": REVIEWERS[cond], "stage": "review", "task_id": tid, "condition": cond,
                                "retried": n_prompts == 2})
            if verdicts.get((tid, cond)) != "REJECT" or (tid, "revise", cond) in exact:
                continue
            if review_text is None:   # C4: the tool reviewer's critique, cached like a model review
                review_text = cache.get(f"per_review_{cond}", "public-test-runner", tid, 0)
            revised = cache.get(f"per_revise_{cond}", EXECUTOR, tid, 0)
            if review_text is None or revised is None:
                print(f"skip {tid} {cond} revise: cache entry missing")
                continue
            prompt = rec.CODE_REVISE_PROMPT.format(**rec.code_revise_input(task, a0, parse_review(review_text)["critique"]))
            out.append({"in_tokens": _estimate_tokens(prompt), "out_tokens": _estimate_tokens(revised),
                        "model_id": EXECUTOR, "stage": "revise", "task_id": tid, "condition": cond})

    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            r.update({"exact": False, "reconstructed": True})
            f.write(json.dumps(r) + "\n")
    print(f"{len(out)} re-estimated records for {len(lost_tasks)} problems -> {OUT}")


if __name__ == "__main__":
    main()
