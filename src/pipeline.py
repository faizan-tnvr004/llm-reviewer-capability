"""PER pipeline: Executor -> Reviewer -> (Revise once).

Design (per the amended plan):
- Executor is FIXED (Gemini). It solves each task ONCE; the same solution A0 is
  then reviewed under every reviewer condition. This paired design isolates the
  reviewer's effect (no executor sampling noise between conditions) and cuts
  executor calls 4x.
- Reviewer conditions: None (C0 baseline), strong, mid, weak.
- Reviewer sees the problem + the executor's full solution, NEVER the ground
  truth. It must output 'VERDICT: ACCEPT' or 'VERDICT: REJECT' plus a critique.
- On REJECT, the executor receives its own solution + the critique and revises
  once. Final answer = revision; on ACCEPT, final = A0.
- Malformed/empty reviews are treated as ACCEPT and flagged (pre-registered
  decision), and their rate is reported.
- Token accounting: exact usage from the provider if available via a usage
  hook, else a chars/4 estimate (flagged as estimated).
"""

from __future__ import annotations

import re
import time
from typing import Callable, List, Optional

from core import Task, grade_numeric
from harness import Model, DiskCache, RunLog, call_with_retry, DailyLimitExhausted


# ---------------------------------------------------------------------------
# Prompts (FROZEN before the main run — do not edit after experiments begin)
# ---------------------------------------------------------------------------
#
# INPUT TRUNCATION (added before the N=100 analysis, applied to ALL reviewer
# conditions identically): olympiad solutions are long, and the mid-tier
# reviewer (gpt-oss-20b) hit its 8000 TPM ceiling on 29% of reviews, returning
# empty text that was conservatively counted as ACCEPT. Reducing reviewer INPUT
# is the only available lever (the TPM cap is hard and cannot be raised).
# We therefore (a) strip the answer-format instructions from the problem text —
# a reviewer does not need them — and (b) truncate the solution to its final
# SOLUTION_TAIL_CHARS characters, which retains the concluding reasoning and the
# stated answer, i.e. what is needed to judge final-answer correctness.
# Applied to every condition so all reviewers see identical input.

SOLUTION_TAIL_CHARS = 2000   # keep the reasoning tail + stated answer


def strip_answer_instruction(problem_prompt: str) -> str:
    """Remove the executor-facing answer-format instructions from the problem
    text; a reviewer does not need them and they cost input tokens."""
    marker = "Work through the problem step by step"
    idx = problem_prompt.find(marker)
    return problem_prompt[:idx].rstrip() if idx != -1 else problem_prompt.rstrip()


def truncate_solution(solution: str, limit: int = SOLUTION_TAIL_CHARS) -> str:
    """Keep the TAIL of the solution (conclusion + stated answer). Prepends an
    explicit elision marker so the reviewer knows earlier steps were omitted."""
    if len(solution) <= limit:
        return solution
    return ("[earlier steps omitted for length]\n...\n" + solution[-limit:])


REVIEW_PROMPT = """You are a mathematical reviewer. Below is a competition math problem and the final part of a proposed solution. Your job is to judge whether the FINAL ANSWER is correct.

PROBLEM:
{problem}

PROPOSED SOLUTION (final part):
{solution}

Check the reasoning and the final answer. Respond in exactly this format:
VERDICT: ACCEPT
or
VERDICT: REJECT
CRITIQUE: <one short paragraph naming the specific error>

Only reject if you believe the final answer is wrong. Do not reject for style. Be concise."""

REVISE_PROMPT = """You previously solved this competition math problem:

PROBLEM:
{problem}

YOUR SOLUTION:
{solution}

A reviewer examined your solution and rejected it with this critique:
CRITIQUE: {critique}

Reconsider the problem in light of the critique. If the critique identifies a real error, fix it. If you believe your original answer was correct, you may keep it. Work through the problem again, then give your final answer on the last line in exactly this format:
ANSWER: <number>
Give the number as a plain decimal (not a fraction, not in words, no units, no LaTeX)."""


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------

_VERDICT = re.compile(r"VERDICT\s*[:=]\s*(ACCEPT|REJECT)", re.IGNORECASE)
_CRITIQUE = re.compile(r"CRITIQUE\s*[:=]\s*(.+)", re.IGNORECASE | re.DOTALL)


def parse_review(review_text: str) -> dict:
    """Parse the reviewer's output.
    Returns {verdict: 'ACCEPT'|'REJECT', critique: str, malformed: bool}.
    Empty/unparseable reviews -> ACCEPT with malformed=True (pre-registered)."""
    if not review_text or not review_text.strip():
        return {"verdict": "ACCEPT", "critique": "", "malformed": True}
    m = _VERDICT.search(review_text)
    if not m:
        return {"verdict": "ACCEPT", "critique": "", "malformed": True}
    verdict = m.group(1).upper()
    critique = ""
    cm = _CRITIQUE.search(review_text)
    if cm:
        critique = cm.group(1).strip()
    if verdict == "REJECT" and not critique:
        # Reject with no reason given: still honor the reject, use a stub.
        critique = "(reviewer rejected without a stated critique)"
    return {"verdict": verdict, "critique": critique, "malformed": False}


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class TokenMeter:
    """Wraps model calls, recording exact usage when the provider exposes it
    (via an optional get_last_usage() hook in providers.py) else a chars/4
    estimate. Each record: (model_id, stage, in_tokens, out_tokens, exact)."""

    def __init__(self, usage_hook: Optional[Callable[[], Optional[dict]]] = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.usage_hook = usage_hook
        self.sleep = sleep
        self.records: List[dict] = []

    def call(self, model: Model, prompt: str, stage: str, task_id: str,
             condition: str) -> str:
        out = call_with_retry(model, prompt, sleep=self.sleep)
        usage = self.usage_hook() if self.usage_hook else None
        if usage and "prompt_tokens" in usage:
            rec = {"in_tokens": usage["prompt_tokens"],
                   "out_tokens": usage["completion_tokens"], "exact": True}
        else:
            rec = {"in_tokens": _estimate_tokens(prompt),
                   "out_tokens": _estimate_tokens(out), "exact": False}
        rec.update({"model_id": model.model_id, "stage": stage,
                    "task_id": task_id, "condition": condition})
        self.records.append(rec)
        return out


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------

def run_per_experiment(tasks: List[Task],
                       executor: Model,
                       reviewer_conditions: dict,   # {"C0": None, "C1": Model, ...}
                       cache: DiskCache,
                       log: RunLog,
                       meter: TokenMeter,
                       verbose: bool = True) -> dict:
    """Run the full PER experiment.

    Per task: executor solves ONCE (cached, shared across conditions). Then for
    each condition with a reviewer: review; on REJECT, executor revises once.
    Every (task, condition) outcome is logged with grading of A0 and final.

    Resumable: all three stages are cached; a daily-limit stop resumes cleanly.
    """
    already = {(r["task_id"], r["condition"]) for r in log.load()}
    done = 0

    for task in tasks:
        # ---- Stage 1: execute once, shared across conditions ----
        a0_key = ("per_execute", executor.model_id, task.task_id, 0)
        a0_text = cache.get(*a0_key)
        try:
            if a0_text is None:
                a0_text = meter.call(executor, task.prompt, "execute",
                                     task.task_id, "shared")
                cache.put(*a0_key, a0_text)

            a0_grade = grade_numeric(task, a0_text)

            for cond_name, reviewer in reviewer_conditions.items():
                if (task.task_id, cond_name) in already:
                    continue

                if reviewer is None:
                    # C0: no review, final = A0
                    _log_outcome(log, task, cond_name, a0_text, a0_grade,
                                 review=None, final_text=a0_text,
                                 final_grade=a0_grade)
                    already.add((task.task_id, cond_name)); done += 1
                    continue

                # ---- Stage 2: review ----
                r_key = (f"per_review_{cond_name}", reviewer.model_id,
                         task.task_id, 0)
                review_text = cache.get(*r_key)
                if review_text is None:
                    review_text = meter.call(
                        reviewer,
                        REVIEW_PROMPT.format(
                            problem=strip_answer_instruction(task.prompt),
                            solution=truncate_solution(a0_text)),
                        "review", task.task_id, cond_name)
                    cache.put(*r_key, review_text)
                review = parse_review(review_text)

                if review["verdict"] == "ACCEPT":
                    _log_outcome(log, task, cond_name, a0_text, a0_grade,
                                 review=review, final_text=a0_text,
                                 final_grade=a0_grade)
                    already.add((task.task_id, cond_name)); done += 1
                    continue

                # ---- Stage 3: revise once ----
                v_key = (f"per_revise_{cond_name}", executor.model_id,
                         task.task_id, 0)
                revised_text = cache.get(*v_key)
                if revised_text is None:
                    revised_text = meter.call(
                        executor,
                        REVISE_PROMPT.format(problem=task.prompt,
                                             solution=a0_text,
                                             critique=review["critique"]),
                        "revise", task.task_id, cond_name)
                    cache.put(*v_key, revised_text)
                final_grade = grade_numeric(task, revised_text)
                _log_outcome(log, task, cond_name, a0_text, a0_grade,
                             review=review, final_text=revised_text,
                             final_grade=final_grade)
                already.add((task.task_id, cond_name)); done += 1

        except DailyLimitExhausted:
            if verbose:
                print(f"[pipeline] daily limit hit — stopping gracefully after "
                      f"{done} new outcomes. Re-run after reset to resume.")
            return {"completed_outcomes": done, "stopped_daily_limit": True}

    return {"completed_outcomes": done, "stopped_daily_limit": False}


def _log_outcome(log, task, condition, a0_text, a0_grade, review,
                 final_text, final_grade):
    log.append({
        "task_id": task.task_id,
        "condition": condition,
        "difficulty": task.meta.get("difficulty"),
        "ground_truth": task.ground_truth,
        "a0_extracted": a0_grade["extracted"],
        "a0_correct": a0_grade["correct"],
        "verdict": (review or {}).get("verdict", "NONE"),
        "review_malformed": (review or {}).get("malformed", False),
        "critique": (review or {}).get("critique", "")[:500],
        "final_extracted": final_grade["extracted"],
        "final_correct": final_grade["correct"],
        "revised": bool(review and review.get("verdict") == "REJECT"),
        "ts": time.time(),
    })


# ---------------------------------------------------------------------------
# Analysis: the pre-registered metrics, computed from the log
# ---------------------------------------------------------------------------

def analyze(log_rows: List[dict], token_records: List[dict] | None = None) -> dict:
    """Compute per-condition metrics:
    accuracy, delta vs C0, reviewer recall, false-rejection rate, repair rate,
    damage rate, malformed-review rate, and (if token records given) mean
    tokens per task."""
    conditions = sorted({r["condition"] for r in log_rows})
    out = {}
    c0_acc = None
    for cond in conditions:
        rows = [r for r in log_rows if r["condition"] == cond]
        n = len(rows)
        acc = sum(r["final_correct"] for r in rows) / n if n else 0.0

        wrong_a0 = [r for r in rows if not r["a0_correct"]]
        right_a0 = [r for r in rows if r["a0_correct"]]
        rejected = [r for r in rows if r["verdict"] == "REJECT"]

        recall = (sum(1 for r in wrong_a0 if r["verdict"] == "REJECT") /
                  len(wrong_a0)) if wrong_a0 else None
        false_rej = (sum(1 for r in right_a0 if r["verdict"] == "REJECT") /
                     len(right_a0)) if right_a0 else None
        rej_wrong = [r for r in rejected if not r["a0_correct"]]
        repair = (sum(1 for r in rej_wrong if r["final_correct"]) /
                  len(rej_wrong)) if rej_wrong else None
        rej_right = [r for r in rejected if r["a0_correct"]]
        damage = (sum(1 for r in rej_right if not r["final_correct"]) /
                  len(rej_right)) if rej_right else None
        malformed = sum(r["review_malformed"] for r in rows) / n if n else 0.0

        m = {"n": n, "accuracy": acc, "recall": recall,
             "false_rejection_rate": false_rej, "repair_rate": repair,
             "damage_rate": damage, "malformed_review_rate": malformed}
        if cond == "C0":
            c0_acc = acc
        out[cond] = m

    for cond, m in out.items():
        m["delta_vs_C0"] = (m["accuracy"] - c0_acc) if c0_acc is not None else None

    if token_records:
        for cond in conditions:
            toks = [t["in_tokens"] + t["out_tokens"] for t in token_records
                    if t["condition"] in (cond, "shared")]
            n = out[cond]["n"] or 1
            out[cond]["mean_tokens_per_task"] = sum(toks) / n if toks else None
    return out