"""Analysis script for the code extension experiment.

Reads run_code_data/log.jsonl (or custom path) and produces the pre-registered metrics:
  - Per-condition accuracy with Wilson 95% CIs
  - Delta vs C0
  - Reviewer recall, false-rejection rate, repair rate, damage rate
  - Rejection-quality breakdown (repair / damage / no-change-wrong / no-change-right / other)
  - Pre-registered paired McNemar tests (C1-C4 vs C0, and C4 vs C2)
  - Exact Fisher tests on per-rejection proportions
  - Holm-Bonferroni correction for pre-specified comparisons
  - Malformed-review rate
  - C4 False-accept rate and Timeout rate
  - Token cost per additional correct answer
  - H1-H4 verdicts (evaluate_hypotheses) and exploratory E2 (temporal_sensitivity)
  - Writes run_code_data/RESULTS.md with metric tables and math-vs-code comparison

Run from repo root:  python analyze_code.py [--log PATH] [--tokens PATH] [--out PATH]
No API calls. Pure re-analysis of the experiment log.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from math import comb, sqrt
import os
import sys

# Ensure UTF-8 output on Windows consoles if possible
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DEFAULT_LOG = os.path.join("run_code_data", "log.jsonl")
DEFAULT_TOKENS = os.path.join("run_code_data", "tokens.jsonl")
DEFAULT_TOKENS_RECONSTRUCTED = os.path.join("run_code_data", "tokens_reconstructed.jsonl")
DEFAULT_PROGRAMS = os.path.join("run_code_data", "programs.jsonl")
DEFAULT_OUT = os.path.join("run_code_data", "RESULTS.md")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Analyze code extension results")
    p.add_argument("--log", default=DEFAULT_LOG, help="Path to experiment log")
    p.add_argument("--tokens", default=DEFAULT_TOKENS, help="Path to token records")
    p.add_argument("--tokens-reconstructed", default=DEFAULT_TOKENS_RECONSTRUCTED,
                   help="Re-estimated token records for calls whose exact usage was lost (Amendment 1)")
    p.add_argument("--programs", default=DEFAULT_PROGRAMS, help="Path to programs.jsonl")
    p.add_argument("--out", default=DEFAULT_OUT, help="Path to output RESULTS.md")
    return p.parse_args()


def load_programs(programs_path: str) -> list[dict]:
    """Load programs.jsonl, de-duplicating by (task_id, condition, stage).
    If duplicates exist (e.g. from a resumed run after an interruption), the
    last occurrence is preserved.
    """
    if not os.path.exists(programs_path):
        return []
    records = []
    with open(programs_path, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    for idx, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if idx == len(lines) - 1:
                continue
            raise
    dedup = {}
    for r in records:
        key = (r.get("task_id"), r.get("condition"), r.get("stage", "final"))
        dedup[key] = r
    return list(dedup.values())


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = z * sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def exact_binom_two_sided(b: int, c: int) -> float:
    """Exact two-sided binomial p-value for McNemar's test."""
    n = b + c
    if n == 0:
        return 1.0
    obs = comb(n, b)
    total = 2 ** n
    p = sum(comb(n, k) for k in range(n + 1) if comb(n, k) <= obs) / total
    return min(1.0, p)


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for 2x2 table [[a,b],[c,d]]."""
    n = a + b + c + d
    if n == 0:
        return 1.0
    row1, col1 = a + b, a + c

    def prob(x):
        return (comb(row1, x) * comb(n - row1, col1 - x)) / comb(n, col1)

    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)
    p_obs = prob(a)
    tol = p_obs * (1 + 1e-9)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= tol))


def holm_bonferroni(pvalues: list[float]) -> list[tuple[float, float]]:
    """Holm-Bonferroni correction. Returns list of (original_p, adjusted_p)."""
    if not pvalues:
        return []
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    m = len(pvalues)
    adjusted = [None] * m
    cum_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = min(1.0, p * (m - rank))
        cum_max = max(cum_max, adj)
        adjusted[orig_idx] = cum_max
    return [(pvalues[i], adjusted[i]) for i in range(m)]


# ---------------------------------------------------------------------------
# Rejection classification (matches rej_quality.py logic)
# ---------------------------------------------------------------------------

def classify_rejection(r: dict, use_ast: bool = True) -> str:
    """Classify a rejection outcome.
    If use_ast=True and ast_hash is present, evaluates program identity via normalized AST hash.
    Otherwise falls back to text code hash (final_extracted == a0_extracted).
    """
    if use_ast and r.get("final_ast_hash") is not None and r.get("a0_ast_hash") is not None:
        same = (r.get("final_ast_hash") == r.get("a0_ast_hash"))
    else:
        same = (r.get("final_extracted") == r.get("a0_extracted"))
    if same:
        return "no_change_right" if r.get("a0_correct") else "no_change_wrong"
    if not r.get("a0_correct") and r.get("final_correct"):
        return "repair"
    if r.get("a0_correct") and not r.get("final_correct"):
        return "damage"
    return "other"  # changed answer, still wrong



def check_a0_integrity(rows: list[dict]) -> None:
    """Integrity check: all rows for a given problem must share identical
    a0_code_hash, a0_ast_hash, and a0_fingerprint.
    Any violation indicates corrupted run state or cross-condition leakage,
    is reported loudly, and blocks primary results.
    """
    by_task: dict[str, list[dict]] = {}
    for r in rows:
        by_task.setdefault(r["task_id"], []).append(r)

    violations = []
    for task_id, task_rows in by_task.items():
        if len(task_rows) <= 1:
            continue
        first = task_rows[0]
        first_ch = first.get("a0_code_hash") or first.get("a0_extracted")
        first_ah = first.get("a0_ast_hash")
        first_fp = first.get("a0_fingerprint")

        for r in task_rows[1:]:
            cur_ch = r.get("a0_code_hash") or r.get("a0_extracted")
            cur_ah = r.get("a0_ast_hash")
            cur_fp = r.get("a0_fingerprint")

            mismatches = []
            if cur_ch != first_ch:
                mismatches.append(f"a0_code_hash ({first_ch!r} vs {cur_ch!r})")
            if cur_ah != first_ah:
                mismatches.append(f"a0_ast_hash ({first_ah!r} vs {cur_ah!r})")
            if cur_fp != first_fp:
                mismatches.append(f"a0_fingerprint ({first_fp!r} vs {cur_fp!r})")

            if mismatches:
                violations.append({
                    "task_id": task_id,
                    "cond_first": first.get("condition"),
                    "cond_cur": r.get("condition"),
                    "mismatches": mismatches,
                })

    if violations:
        print("\n" + "!" * 78)
        print("CRITICAL DATA INTEGRITY VIOLATION IN EXPERIMENT LOG:")
        print("A0 program identity mismatch across conditions for the same task!")
        for v in violations:
            print(f"  Task '{v['task_id']}': mismatch between {v['cond_first']} and {v['cond_cur']}: "
                  + ", ".join(v["mismatches"]))
        print("Primary results are BLOCKED to prevent reporting invalid paired metrics.")
        print("!" * 78 + "\n")
        raise ValueError(
            f"Integrity check failed: {len(violations)} task(s) have conflicting A0 hashes or fingerprints across conditions."
        )


# ---------------------------------------------------------------------------
# Pre-registered hypotheses H1-H4 (ANALYSIS_PLAN_CODE.md section 5)
# ---------------------------------------------------------------------------

PRIMARY_PAIRS = [("C0", "C1"), ("C0", "C2"), ("C0", "C3"), ("C0", "C4"), ("C2", "C4")]
LOW_N = 20
H2_MIN_FALSE_REJECTIONS = 10


def _low_n(n: int) -> str:
    return " (low-n)" if n < LOW_N else ""


def paired_mcnemar(rows: list[dict], pairs=PRIMARY_PAIRS) -> dict:
    """Exact McNemar for each pair, Holm-adjusted across the pairs present.
    Returns {(a, b): {"n", "b", "c", "accA", "accB", "p_raw", "p_adj"}}."""
    by_task: dict[str, dict[str, bool]] = {}
    for r in rows:
        by_task.setdefault(r["task_id"], {})[r["condition"]] = bool(r.get("final_correct"))
    out = {}
    for a, b in pairs:
        paired = [t for t, d in by_task.items() if a in d and b in d]
        if not paired:
            continue
        b_val = sum(1 for t in paired if by_task[t][a] and not by_task[t][b])
        c_val = sum(1 for t in paired if not by_task[t][a] and by_task[t][b])
        out[(a, b)] = {
            "n": len(paired), "b": b_val, "c": c_val,
            "accA": sum(by_task[t][a] for t in paired) / len(paired),
            "accB": sum(by_task[t][b] for t in paired) / len(paired),
            "p_raw": exact_binom_two_sided(b_val, c_val),
        }
    for key, (_, p_adj) in zip(out, holm_bonferroni([v["p_raw"] for v in out.values()])):
        out[key]["p_adj"] = p_adj
    return out


def evaluate_hypotheses(rows: list[dict]) -> dict:
    """Verdicts for H1-H4 with the counts behind them. Each verdict is
    'supported', 'not supported', 'inconclusive' or 'not evaluable'."""
    def cond_rows(c):
        return [r for r in rows if r["condition"] == c]

    res = {}

    # H1: C1 false-rejection rate on correct A0 >= 20%
    right = [r for r in cond_rows("C1") if r.get("a0_correct")]
    k = sum(1 for r in right if r.get("verdict") == "REJECT")
    if right:
        rate = k / len(right)
        res["H1"] = {"verdict": "supported" if rate >= 0.20 else "not supported",
                     "k": k, "n": len(right), "rate": rate, "ci": wilson_ci(k, len(right)),
                     "text": f"C1 FRR {k}/{len(right)} = {rate*100:.1f}%{_low_n(len(right))}"}
    else:
        res["H1"] = {"verdict": "not evaluable", "text": "no correct A0 under C1"}

    # H2: > 50% of C1 false rejections leave the AST hash unchanged; < 10 -> inconclusive
    fr = [r for r in right if r.get("verdict") == "REJECT"]
    k_ast = sum(1 for r in fr if classify_rejection(r, use_ast=True) == "no_change_right")
    k_txt = sum(1 for r in fr if classify_rejection(r, use_ast=False) == "no_change_right")
    if len(fr) < H2_MIN_FALSE_REJECTIONS:
        verdict = "inconclusive"
    else:
        verdict = "supported" if k_ast / len(fr) > 0.50 else "not supported"
    text = (f"unchanged after C1 false rejection: AST {k_ast}/{len(fr)}, "
            f"text {k_txt}/{len(fr)}{_low_n(len(fr))}")
    if verdict == "inconclusive":
        text += f"; fewer than {H2_MIN_FALSE_REJECTIONS} false rejections"
    res["H2"] = {"verdict": verdict, "k": k_ast, "k_text": k_txt, "n": len(fr),
                 "ci": wilson_ci(k_ast, len(fr)), "text": text}

    # H3: C2 accuracy > C0, two-sided exact McNemar, Holm-adjusted p < 0.05
    m = paired_mcnemar(rows).get(("C0", "C2"))
    if m:
        ok = m["accB"] > m["accA"] and m["p_adj"] < 0.05
        res["H3"] = {"verdict": "supported" if ok else "not supported", **m,
                     "text": (f"C0 {m['accA']*100:.1f}% vs C2 {m['accB']*100:.1f}%, b={m['b']} c={m['c']}, "
                              f"p={m['p_raw']:.4f}, Holm p={m['p_adj']:.4f}{_low_n(m['n'])}")}
    else:
        res["H3"] = {"verdict": "not evaluable", "text": "C0 or C2 missing"}

    # H4: C4 FRR <= 5% and at least one wrong A0 passes all public tests
    c4 = cond_rows("C4")
    if c4:
        right4 = [r for r in c4 if r.get("a0_correct")]
        wrong4 = [r for r in c4 if not r.get("a0_correct")]
        k_fr = sum(1 for r in right4 if r.get("verdict") == "REJECT")
        frr = k_fr / len(right4) if right4 else 0.0
        k_fa = sum(1 for r in wrong4 if r.get("a0_public_pass"))
        ok = bool(right4) and frr <= 0.05 and k_fa > 0
        res["H4"] = {"verdict": "supported" if ok else "not supported",
                     "k_fr": k_fr, "n_right": len(right4), "frr": frr,
                     "k_fa": k_fa, "n_wrong": len(wrong4),
                     "text": (f"C4 FRR {k_fr}/{len(right4)} = {frr*100:.1f}%{_low_n(len(right4))}; "
                              f"false accepts {k_fa}/{len(wrong4)}{_low_n(len(wrong4))}")}
    else:
        res["H4"] = {"verdict": "not evaluable", "text": "C4 missing"}
    return res


# ---------------------------------------------------------------------------
# Exploratory E2: temporal sensitivity across the 2025-01-31 boundary
# ---------------------------------------------------------------------------

E2_BOUNDARY = "2025-01-31"


def temporal_sensitivity(rows: list[dict], boundary: str = E2_BOUNDARY) -> dict:
    """A0 accuracy and C1 false-rejection rate pre (<= boundary) vs post
    (> boundary), Wilson CIs, two-sided exact Fisher on A0 accuracy, and H1-H4
    re-evaluated on the post-cutoff problems only."""
    dates = {}
    for r in rows:
        d = (r.get("contest_date") or "")[:10]
        if d:
            dates[r["task_id"]] = d
    missing = sorted({r["task_id"] for r in rows} - set(dates))

    strata = {"pre": {t for t, d in dates.items() if d <= boundary},
              "post": {t for t, d in dates.items() if d > boundary}}
    a0_by_task = {r["task_id"]: bool(r.get("a0_correct")) for r in rows}

    out = {"boundary": boundary, "missing_dates": missing}
    for name, tids in strata.items():
        k = sum(1 for t in tids if a0_by_task[t])
        c1_right = [r for r in rows if r["condition"] == "C1" and r["task_id"] in tids and r.get("a0_correct")]
        k_fr = sum(1 for r in c1_right if r.get("verdict") == "REJECT")
        out[name] = {"n": len(tids), "a0_correct": k, "a0_acc_ci": wilson_ci(k, len(tids)),
                     "c1_fr": k_fr, "c1_right": len(c1_right), "c1_frr_ci": wilson_ci(k_fr, len(c1_right))}
    pre, post = out["pre"], out["post"]
    out["fisher_a0_p"] = fisher_exact_two_sided(pre["a0_correct"], pre["n"] - pre["a0_correct"],
                                                post["a0_correct"], post["n"] - post["a0_correct"])
    out["post_hypotheses"] = evaluate_hypotheses([r for r in rows if r["task_id"] in strata["post"]])
    return out


# ---------------------------------------------------------------------------
# Main analysis & report generation
# ---------------------------------------------------------------------------

def run_analysis(log_path: str, tokens_path: str | None = None, out_path: str | None = None,
                 reconstructed_tokens_path: str | None = None) -> dict:
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    rows = [json.loads(line) for line in open(log_path, encoding="utf-8") if line.strip()]
    check_a0_integrity(rows)
    conds = sorted({r["condition"] for r in rows})

    all_tokens = []
    if tokens_path and os.path.exists(tokens_path):
        all_tokens = [json.loads(line) for line in open(tokens_path, encoding="utf-8") if line.strip()]

    # ---- 1. Accuracy & Wilson CIs ----
    cond_stats = {}
    c0_acc = None
    for cond in conds:
        rs = [r for r in rows if r["condition"] == cond]
        n = len(rs)
        k = sum(1 for r in rs if r.get("final_correct"))
        acc = k / n if n else 0.0
        lo, hi = wilson_ci(k, n)
        if cond == "C0":
            c0_acc = acc
        delta = (acc - c0_acc) if (c0_acc is not None and cond != "C0") else 0.0
        cond_stats[cond] = {
            "n": n, "correct": k, "accuracy": acc,
            "ci_lo": lo, "ci_hi": hi, "delta": delta,
        }

    # ---- 2. McNemar paired accuracy comparisons ----
    # Pre-specified primary comparisons: each vs C0, plus C4 vs C2
    target_pairs = [("C0", "C1"), ("C0", "C2"), ("C0", "C3"), ("C0", "C4"), ("C2", "C4")]
    active_pairs = [(a, b) for a, b in target_pairs if a in conds and b in conds]
    # If standard pairs not found (e.g. arbitrary conditions), fall back to pairwise combinations
    if not active_pairs:
        active_pairs = list(combinations(conds, 2))

    by_task: dict[str, dict[str, bool]] = {}
    for r in rows:
        by_task.setdefault(r["task_id"], {})[r["condition"]] = bool(r.get("final_correct"))

    mcnemar_results = []
    raw_pvals = []
    for a, b in active_pairs:
        paired_tasks = [t for t, d in by_task.items() if a in d and b in d]
        n_paired = len(paired_tasks)
        if n_paired == 0:
            continue
        accA = sum(1 for t in paired_tasks if by_task[t][a]) / n_paired
        accB = sum(1 for t in paired_tasks if by_task[t][b]) / n_paired
        b_val = sum(1 for t in paired_tasks if by_task[t][a] and not by_task[t][b])
        c_val = sum(1 for t in paired_tasks if not by_task[t][a] and by_task[t][b])
        p = exact_binom_two_sided(b_val, c_val)
        raw_pvals.append(p)
        mcnemar_results.append({
            "pair": f"{a} vs {b}", "a": a, "b": b,
            "accA": accA, "accB": accB, "diff": accB - accA,
            "b_val": b_val, "c_val": c_val, "n_disc": b_val + c_val,
            "p_raw": p,
        })

    if raw_pvals:
        corrected = holm_bonferroni(raw_pvals)
        for res, (_, p_adj) in zip(mcnemar_results, corrected):
            res["p_adj"] = p_adj
            res["star"] = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "~" if p_adj < 0.10 else "ns"

    # ---- 3. Reviewer quality metrics ----
    reviewer_stats = {}
    for cond in conds:
        rs = [r for r in rows if r["condition"] == cond]
        n = len(rs)
        wrong_a0 = [r for r in rs if not r.get("a0_correct")]
        right_a0 = [r for r in rs if r.get("a0_correct")]
        rejected = [r for r in rs if r.get("verdict") == "REJECT"]

        k_recall = sum(1 for r in wrong_a0 if r.get("verdict") == "REJECT")
        recall = k_recall / len(wrong_a0) if wrong_a0 else 0.0
        recall_ci = wilson_ci(k_recall, len(wrong_a0))

        k_frr = sum(1 for r in right_a0 if r.get("verdict") == "REJECT")
        frr = k_frr / len(right_a0) if right_a0 else 0.0
        frr_ci = wilson_ci(k_frr, len(right_a0))

        rej_wrong = [r for r in rejected if not r.get("a0_correct")]
        k_repair = sum(1 for r in rej_wrong if r.get("final_correct"))
        repair = k_repair / len(rej_wrong) if rej_wrong else 0.0
        repair_ci = wilson_ci(k_repair, len(rej_wrong))

        rej_right = [r for r in rejected if r.get("a0_correct")]
        k_damage = sum(1 for r in rej_right if not r.get("final_correct"))
        damage = k_damage / len(rej_right) if rej_right else 0.0
        damage_ci = wilson_ci(k_damage, len(rej_right))

        k_malf = sum(1 for r in rs if r.get("review_malformed"))
        malf = k_malf / n if n else 0.0
        malf_ci = wilson_ci(k_malf, n)

        reviewer_stats[cond] = {
            "n_wrong_a0": len(wrong_a0), "recall": recall, "recall_ci": recall_ci,
            "n_right_a0": len(right_a0), "frr": frr, "frr_ci": frr_ci,
            "n_rej_wrong": len(rej_wrong), "repair": repair, "repair_ci": repair_ci,
            "n_rej_right": len(rej_right), "damage": damage, "damage_ci": damage_ci,
            "malformed": malf, "malformed_ci": malf_ci,
        }

    # ---- 4. Rejection outcome breakdown & Fisher tests ----
    rejection_breakdown = {}
    for cond in conds:
        rs = [r for r in rows if r["condition"] == cond]
        rejects = [r for r in rs if r.get("verdict") == "REJECT"]
        counts_ast = {k: 0 for k in ("repair", "damage", "no_change_wrong", "no_change_right", "other")}
        counts_text = {k: 0 for k in ("repair", "damage", "no_change_wrong", "no_change_right", "other")}
        for r in rejects:
            counts_ast[classify_rejection(r, use_ast=True)] += 1
            counts_text[classify_rejection(r, use_ast=False)] += 1
        n_rej = len(rejects)
        rejection_breakdown[cond] = {
            "n_rej": n_rej,
            "counts": counts_ast,
            "counts_ast": counts_ast,
            "counts_text": counts_text,
            "no_change_total": counts_ast["no_change_wrong"] + counts_ast["no_change_right"],
            "no_change_text_total": counts_text["no_change_wrong"] + counts_text["no_change_right"],
        }

    # Fisher tests on active conditions (>= 3 rejections)
    active_rej_conds = [c for c in conds if rejection_breakdown[c]["n_rej"] >= 3]
    fisher_results = []
    raw_fisher_p = []
    for a, b in combinations(active_rej_conds, 2):
        A = rejection_breakdown[a]
        B = rejection_breakdown[b]
        for metric, key in [("repair rate", "repair"), ("no-change rate", "no_change_total")]:
            xa = A["counts"][key] if key != "no_change_total" else A["no_change_total"]
            xb = B["counts"][key] if key != "no_change_total" else B["no_change_total"]
            na, nb = A["n_rej"], B["n_rej"]
            p = fisher_exact_two_sided(xa, na - xa, xb, nb - xb)
            raw_fisher_p.append(p)
            fisher_results.append({
                "pair": f"{a} vs {b}", "metric": metric,
                "xa": xa, "na": na, "xb": xb, "nb": nb,
                "p_raw": p,
            })
    if raw_fisher_p:
        corr_f = holm_bonferroni(raw_fisher_p)
        for res, (_, p_adj) in zip(fisher_results, corr_f):
            res["p_adj"] = p_adj
            res["star"] = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "~" if p_adj < 0.10 else "ns"

    # ---- 5. Operational metrics: Timeouts, C4 False-Accept, Tokens ----
    # Timeouts: count rows where a0_fingerprint or final_fingerprint is TIMEOUT
    timeout_stats = {}
    for cond in conds:
        rs = [r for r in rows if r["condition"] == cond]
        n_timeouts = sum(1 for r in rs if r.get("a0_fingerprint") == "TIMEOUT" or r.get("final_fingerprint") == "TIMEOUT")
        rate = n_timeouts / len(rs) if rs else 0.0
        timeout_stats[cond] = {"n_timeouts": n_timeouts, "total": len(rs), "rate": rate}

    # C4 False-Accept rate: Wrong A0 passing all public tests / wrong A0
    # From log extra fields 'a0_public_pass' or C4 outcome
    wrong_a0_rows = [r for r in rows if r.get("condition") == "C4" and not r.get("a0_correct")]
    if not wrong_a0_rows:
        wrong_a0_rows = [r for r in rows if not r.get("a0_correct") and "a0_public_pass" in r]
    c4_false_accept_count = sum(1 for r in wrong_a0_rows if r.get("a0_public_pass", False))
    c4_false_accept_total = len(wrong_a0_rows)
    c4_false_accept_rate = (c4_false_accept_count / c4_false_accept_total) if c4_false_accept_total else 0.0

    # Token usage per condition. Amendment 1, Section 2: the primary view uses only
    # problems whose executor call has an exact usage record; the secondary view
    # adds the re-estimated (exact: false) records and covers every problem.
    def token_view(records, task_ids):
        stats = {}
        for cond in conds:
            rs = [r for r in rows if r["condition"] == cond and r["task_id"] in task_ids]
            c0 = [r for r in rows if r["condition"] == "C0" and r["task_id"] in task_ids]
            toks = sum(t.get("in_tokens", 0) + t.get("out_tokens", 0) for t in records
                       if t.get("condition") in (cond, "shared") and t.get("task_id") in task_ids)
            delta_correct = sum(bool(r.get("final_correct")) for r in rs) - sum(bool(r.get("final_correct")) for r in c0)
            stats[cond] = {"n_tasks": len(rs), "mean_tok": toks / len(rs) if rs else 0.0,
                           "delta_correct": delta_correct,
                           "tok_per_extra": (toks / delta_correct) if delta_correct > 0 else float("inf")}
        return stats

    all_task_ids = {r["task_id"] for r in rows}
    exact_task_ids = {t["task_id"] for t in all_tokens
                      if t.get("exact") and t.get("stage") == "execute" and t.get("condition") == "shared"}
    token_stats = token_view(all_tokens, exact_task_ids)
    token_stats_all = None
    if reconstructed_tokens_path and os.path.exists(reconstructed_tokens_path):
        extra = [json.loads(line) for line in open(reconstructed_tokens_path, encoding="utf-8") if line.strip()]
        token_stats_all = token_view(all_tokens + extra, all_task_ids)

    # ---- Print Console Summary (ASCII safe) ----
    print(f"\nLoaded {len(rows)} outcome rows from {log_path}")
    print("=" * 76)
    print("PER-CONDITION ACCURACY (with Wilson 95% CI)")
    print("=" * 76)
    print(f"{'Cond':6} {'N':>4} {'Correct':>8} {'Acc':>8} {'95% CI':>18} {'Delta vs C0':>12}")
    print("-" * 76)
    for c in conds:
        s = cond_stats[c]
        print(f"{c:6} {s['n']:>4} {s['correct']:>8} {s['accuracy']*100:>7.1f}% [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}] {s['delta']*100:>+11.1f}%")

    if mcnemar_results:
        print("\n" + "=" * 76)
        print("PRE-REGISTERED McNEMAR TESTS (Paired Accuracy)")
        print("=" * 76)
        print(f"{'Comparison':16} {'Acc A':>7} {'Acc B':>7} {'Diff':>8} {'b':>4} {'c':>4} {'p (raw)':>10} {'p (adj)':>10}")
        print("-" * 76)
        for m in mcnemar_results:
            p_adj_str = f"{m.get('p_adj', 1.0):.4f} {m.get('star', 'ns')}"
            print(f"{m['pair']:16} {m['accA']*100:>6.1f}% {m['accB']*100:>6.1f}% {m['diff']*100:>+7.1f}% {m['b_val']:>4} {m['c_val']:>4} {m['p_raw']:>10.4f} {p_adj_str:>10}")

    # ---- Build Markdown Report ----
    md_lines = [
        "# Code Extension: Empirical Results",
        "",
        f"**Log source**: `{log_path}`  ",
        f"**Completed outcomes**: {len(rows)}  ",
        f"**Conditions**: {', '.join(conds)}  ",
        "",
        "## 1. Primary Paired Accuracy & McNemar Tests",
        "",
        "| Condition | Reviewer | N | Correct | Accuracy | Wilson 95% CI | Δ vs C0 |",
        "|---|---|---|---|---|---|---|",
    ]
    reviewer_names = {
        "C0": "None (baseline)",
        "C1": "Self-review (Gemini 3.1 Flash-Lite)",
        "C2": "Cross-family (openai/gpt-oss-20b)",
        "C3": "Weak (llama-3.1-8b-instant)",
        "C4": "Public sample tests (deterministic)",
    }
    for c in conds:
        s = cond_stats[c]
        rname = reviewer_names.get(c, c)
        md_lines.append(f"| **{c}** | {rname} | {s['n']} | {s['correct']} | **{s['accuracy']*100:.1f}%** | [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}] | {s['delta']*100:+.1f}% |")

    md_lines.extend([
        "",
        "### Pre-registered Paired Comparisons (Exact McNemar)",
        "",
        "| Comparison | Acc A | Acc B | Diff | Discordant (b / c) | Raw p | Holm-Adj p | Verdict |",
        "|---|---|---|---|---|---|---|---|",
    ])
    for m in mcnemar_results:
        md_lines.append(f"| **{m['pair']}** | {m['accA']*100:.1f}% | {m['accB']*100:.1f}% | {m['diff']*100:+.1f}% | {m['b_val']} / {m['c_val']} | {m['p_raw']:.4f} | {m.get('p_adj', 1.0):.4f} | {m.get('star', 'ns')} |")

    md_lines.extend([
        "",
        "## 2. Reviewer Quality & Error Detection",
        "",
        "| Condition | Recall on Wrong A0 | False Rejection Rate (FRR) | Repair Rate | Damage Rate | Malformed Review Rate |",
        "|---|---|---|---|---|---|",
    ])
    for c in conds:
        r = reviewer_stats[c]
        md_lines.append(
            f"| **{c}** | {r['recall']*100:.1f}% [{r['recall_ci'][0]:.2f}, {r['recall_ci'][1]:.2f}] | "
            f"{r['frr']*100:.1f}% [{r['frr_ci'][0]:.2f}, {r['frr_ci'][1]:.2f}] | "
            f"{r['repair']*100:.1f}% [{r['repair_ci'][0]:.2f}, {r['repair_ci'][1]:.2f}] | "
            f"{r['damage']*100:.1f}% [{r['damage_ci'][0]:.2f}, {r['damage_ci'][1]:.2f}] | "
            f"{r['malformed']*100:.1f}% |"
        )

    md_lines.extend([
        "",
        "## 3. Rejection Outcomes & Quality Breakdown",
        "",
        "Rejection classification uses normalized AST hash (`ast_hash`) as the primary definition of program identity (ignoring comments, docstrings, and formatting). Text-hash counts (`final_code_hash == a0_code_hash`) are reported alongside in brackets `[text]`.",
        "",
        "| Condition | Rejections | Repair (Useful) | Damage (Harmful) | No-Change Wrong (Ignored) [AST / Text] | No-Change Right (Harmless FR) [AST / Text] | Other |",
        "|---|---|---|---|---|---|---|",
    ])
    for c in conds:
        rb = rejection_breakdown[c]
        cnt = rb["counts_ast"]
        cnt_t = rb["counts_text"]
        n_rej = rb["n_rej"]
        def p(x): return f"{x} ({100*x/n_rej:.0f}%)" if n_rej else "-"
        nc_wrong_str = f"{p(cnt['no_change_wrong'])} [text: {cnt_t['no_change_wrong']}]" if n_rej else "-"
        nc_right_str = f"{p(cnt['no_change_right'])} [text: {cnt_t['no_change_right']}]" if n_rej else "-"
        md_lines.append(f"| **{c}** | {n_rej} | {p(cnt['repair'])} | {p(cnt['damage'])} | {nc_wrong_str} | {nc_right_str} | {p(cnt['other'])} |")


    if fisher_results:
        md_lines.extend([
            "",
            "### Exact Fisher Tests on Per-Rejection Rates",
            "",
            "| Comparison | Metric | Proportion A | Proportion B | Raw p | Holm-Adj p | Sig |",
            "|---|---|---|---|---|---|---|",
        ])
        for f in fisher_results:
            propA = f"{f['xa']}/{f['na']} ({100*f['xa']/f['na']:.0f}%)"
            propB = f"{f['xb']}/{f['nb']} ({100*f['xb']/f['nb']:.0f}%)"
            md_lines.append(f"| **{f['pair']}** | {f['metric']} | {propA} | {propB} | {f['p_raw']:.4f} | {f.get('p_adj', 1.0):.4f} | {f.get('star', 'ns')} |")

    md_lines.extend([
        "",
        "## 4. Code Execution & Resource Metrics",
        "",
        f"- **C4 False-Accept Count / Rate**: {c4_false_accept_count}/{c4_false_accept_total} ({c4_false_accept_rate*100:.1f}%) wrong A0 programs passed all public sample tests.",
        "",
        f"Tokens use exact provider usage on the {len(exact_task_ids)} problems whose records survived (Amendment 1, Section 2). "
        "The all-problems column adds re-estimated chars/4 records for the rest, which miss hidden reasoning tokens.",
        "",
        "| Condition | Timeouts | Timeout Rate | Mean Tokens / Task (exact problems) | Cost / Extra Correct Answer (exact problems) | Mean Tokens / Task (all problems, incl. estimates) |",
        "|---|---|---|---|---|---|",
    ])
    for c in conds:
        tm = timeout_stats[c]
        tk = token_stats.get(c, {"mean_tok": 0.0, "tok_per_extra": float("inf")})
        cpe = "inf" if tk["tok_per_extra"] == float("inf") else f"{tk['tok_per_extra']:.0f}"
        all_tok = f"{token_stats_all[c]['mean_tok']:.0f}" if token_stats_all else "n/a"
        md_lines.append(f"| **{c}** | {tm['n_timeouts']}/{tm['total']} | {tm['rate']*100:.1f}% | {tk['mean_tok']:.0f} | {cpe} | {all_tok} |")

    # Math vs Code comparison table
    md_lines.extend([
        "",
        "## 5. Olympiad Math vs Executable Code Comparison",
        "",
        "| Finding / Metric | Math Study (Olympiad) | Code Study (LiveCodeBench) | Replication Finding |",
        "|---|---|---|---|",
        f"| Baseline Executor Accuracy (C0) | 52% (52/100) | {cond_stats.get('C0', {}).get('accuracy', 0)*100:.1f}% | Baseline calibration |",
        f"| Self-Review Delta (C1 vs C0) | +6.0% (52% -> 58%, p=0.1460 ns) | {cond_stats.get('C1', {}).get('delta', 0)*100:+.1f}% | Replicates lack of significant gain |",
        f"| Cross-Family Delta (C2 vs C0) | +12.0% (52% -> 64%, p=0.0005 ***) | {cond_stats.get('C2', {}).get('delta', 0)*100:+.1f}% | Test cross-family superiority |",
        f"| Self-Review False Rejection Rate (C1 FRR) | 35% (18/52 correct answers) | {reviewer_stats.get('C1', {}).get('frr', 0)*100:.1f}% | Test high false rejection rate |",
        f"| Ignored False Rejections | 83% (15 of 18 unchanged) | {rejection_breakdown.get('C1', {}).get('counts', {}).get('no_change_right', 0)} / {reviewer_stats.get('C1', {}).get('n_rej_right', 0)} | Test inertia on false critiques |",
        f"| Damage on False Rejection | 100% of changes broke code (3/3) | {rejection_breakdown.get('C1', {}).get('counts', {}).get('damage', 0)} damaged | Test destructive revision effect |",
        f"| Public Sample Tests Reviewer (C4) | N/A (code-only condition) | {cond_stats.get('C4', {}).get('accuracy', 0)*100:.1f}% acc (FRR={reviewer_stats.get('C4', {}).get('frr', 0)*100:.1f}%) | Deterministic test-based baseline |",
        "",
    ])

    # ---- 6. Hypotheses H1-H4 and exploratory E2 ----
    hypotheses = evaluate_hypotheses(rows)
    e2 = temporal_sensitivity(rows)

    print("\n" + "=" * 76)
    print("PRE-REGISTERED HYPOTHESES")
    print("=" * 76)
    for h, v in hypotheses.items():
        print(f"{h}: {v['verdict'].upper():14} {v['text']}")
    print("\n" + "=" * 76)
    print(f"E2 TEMPORAL SENSITIVITY (boundary {e2['boundary']}, exploratory)")
    print("=" * 76)
    for name in ("pre", "post"):
        st = e2[name]
        print(f"{name:5} n={st['n']:>3}  A0 {st['a0_correct']}/{st['n']} [{st['a0_acc_ci'][0]:.3f}, {st['a0_acc_ci'][1]:.3f}]  "
              f"C1 FRR {st['c1_fr']}/{st['c1_right']} [{st['c1_frr_ci'][0]:.3f}, {st['c1_frr_ci'][1]:.3f}]")
    print(f"Fisher (A0 accuracy pre vs post): p = {e2['fisher_a0_p']:.4f}")
    if e2["missing_dates"]:
        print(f"WARNING: {len(e2['missing_dates'])} task(s) have no contest_date and are excluded from E2")
    for h, v in e2["post_hypotheses"].items():
        print(f"  post-cutoff {h}: {v['verdict']} ({v['text']})")

    md_lines.extend([
        "## 6. Pre-Registered Hypotheses",
        "",
        "| Hypothesis | Verdict | Evidence |",
        "|---|---|---|",
    ])
    for h, v in hypotheses.items():
        md_lines.append(f"| **{h}** | {v['verdict']} | {v['text']} |")
    md_lines.extend([
        "",
        f"## 7. Exploratory E2: Temporal Sensitivity (boundary {e2['boundary']})",
        "",
        "Exploratory and non-confirmatory. Post-cutoff hypothesis checks are low-n sensitivity checks.",
        "",
        "| Stratum | Problems | A0 Correct | A0 Accuracy [Wilson 95% CI] | C1 FRR [Wilson 95% CI] |",
        "|---|---|---|---|---|",
    ])
    for name, label in (("pre", f"Pre (<= {e2['boundary']})"), ("post", f"Post (> {e2['boundary']})")):
        st = e2[name]
        acc = st["a0_correct"] / st["n"] * 100 if st["n"] else 0.0
        frr = st["c1_fr"] / st["c1_right"] * 100 if st["c1_right"] else 0.0
        md_lines.append(f"| {label} | {st['n']} | {st['a0_correct']} | {acc:.1f}% [{st['a0_acc_ci'][0]:.3f}, {st['a0_acc_ci'][1]:.3f}] | "
                        f"{st['c1_fr']}/{st['c1_right']} = {frr:.1f}% [{st['c1_frr_ci'][0]:.3f}, {st['c1_frr_ci'][1]:.3f}] |")
    md_lines.extend([
        "",
        f"- Two-sided exact Fisher on A0 accuracy, pre vs post: p = {e2['fisher_a0_p']:.4f}",
    ])
    if e2["missing_dates"]:
        md_lines.append(f"- {len(e2['missing_dates'])} task(s) without contest_date excluded: {', '.join(e2['missing_dates'])}")
    md_lines.extend(["", "| Hypothesis (post-cutoff only) | Verdict | Evidence |", "|---|---|---|"])
    for h, v in e2["post_hypotheses"].items():
        md_lines.append(f"| {h} | {v['verdict']} | {v['text']} |")
    md_lines.append("")

    if out_path:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")
        print(f"\n[ok] Analysis report written to {out_path}")

    return {
        "cond_stats": cond_stats,
        "mcnemar": mcnemar_results,
        "reviewer_stats": reviewer_stats,
        "rejection_breakdown": rejection_breakdown,
        "fisher": fisher_results,
        "timeout_stats": timeout_stats,
        "token_stats": token_stats,
        "token_stats_all": token_stats_all,
        "hypotheses": hypotheses,
        "e2": e2,
    }


def main():
    args = parse_args()
    if os.path.exists(args.programs):
        progs = load_programs(args.programs)
        print(f"Loaded {len(progs)} unique programs from {args.programs} (de-duplicated by task_id, condition, stage)")
    run_analysis(args.log, args.tokens, args.out, args.tokens_reconstructed)


if __name__ == "__main__":
    main()
