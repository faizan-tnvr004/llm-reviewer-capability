"""Rejection-quality analysis: the paper's reframed spine.

Two things:

(1) SPLIT the no-change column by whether the underlying answer A0 was correct.
    "No-change" currently conflates:
      - rejected a WRONG answer, revision returned the same wrong answer
        (detect-without-correct: the critique fired and was ignored)
      - rejected a CORRECT answer, revision returned the same correct answer
        (a false rejection that got away with it -- no damage, but the
         reviewer was still wrong to reject)
    The second belongs in the false-rejection count, not in a neutral bucket.
    Reported damage (right->wrong) alone UNDERCOUNTS false rejections.

(2) EXACT Fisher tests between conditions on two per-rejection proportions:
      - repair rate      = repairs / rejections
      - no-change rate   = no-change / rejections
    Denominators here are the REJECTIONS (59, 28), not the 100 tasks, which
    is why these have power the accuracy comparison lacks.

Run from repo root:  python rejection_quality.py
No API calls. Pure re-analysis of the existing log.
"""
import json, os
from math import comb
from itertools import combinations

LOG = os.path.join("run_experiment_data", "log.jsonl")
rows = [json.loads(l) for l in open(LOG, encoding="utf-8") if l.strip()]
conds = sorted({r["condition"] for r in rows})


# ---------------------------------------------------------------- Fisher exact
def fisher_exact_two_sided(a, b, c, d):
    """Two-sided Fisher exact p for table [[a,b],[c,d]] (hypergeometric)."""
    n = a + b + c + d
    row1, col1 = a + b, a + c
    def prob(x):
        # P(X = x) for hypergeometric with these margins
        return (comb(row1, x) * comb(n - row1, col1 - x)) / comb(n, col1)
    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)
    p_obs = prob(a)
    tol = p_obs * (1 + 1e-9)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= tol))


# ------------------------------------------------- classify every rejection
def classify(r):
    """Return one of: no_change_wrong, no_change_right, repair, damage, other."""
    same = (r["final_extracted"] == r["a0_extracted"])
    if same:
        return "no_change_right" if r["a0_correct"] else "no_change_wrong"
    if not r["a0_correct"] and r["final_correct"]:
        return "repair"
    if r["a0_correct"] and not r["final_correct"]:
        return "damage"
    return "other"          # changed answer, still wrong


stats = {}
for cond in conds:
    rs = [r for r in rows if r["condition"] == cond]
    rejects = [r for r in rs if r["verdict"] == "REJECT"]
    counts = {k: 0 for k in
              ("no_change_wrong", "no_change_right", "repair", "damage", "other")}
    for r in rejects:
        counts[classify(r)] += 1
    n_right_a0 = sum(1 for r in rs if r["a0_correct"])
    n_wrong_a0 = len(rs) - n_right_a0
    # FULL false rejections = rejected a CORRECT answer, however it turned out
    false_rej = counts["damage"] + counts["no_change_right"]
    stats[cond] = dict(counts, n=len(rs), rejects=len(rejects),
                       n_right_a0=n_right_a0, n_wrong_a0=n_wrong_a0,
                       false_rej=false_rej)

# ------------------------------------------------------------------ report
print("=== rejection outcomes, no-change SPLIT by whether A0 was correct ===\n")
hdr = (f"{'cond':5} {'rej':>4} {'repair':>11} {'damage':>10} "
       f"{'nc-wrong':>11} {'nc-right':>11} {'other':>10}")
print(hdr); print("-" * len(hdr))
for c in conds:
    s = stats[c]
    n = s["rejects"]
    f = lambda k: (f"{s[k]} ({100*s[k]/n:.0f}%)" if n else "-")
    print(f"{c:5} {n:>4} {f('repair'):>11} {f('damage'):>10} "
          f"{f('no_change_wrong'):>11} {f('no_change_right'):>11} {f('other'):>10}")

print("\n  repair    rejected wrong -> fixed")
print("  damage    rejected RIGHT -> broke it        (false rejection, harmful)")
print("  nc-wrong  rejected wrong -> same wrong      (critique ignored)")
print("  nc-right  rejected RIGHT -> same right      (false rejection, harmless)")
print("  other     rejected wrong -> different wrong (detect-miscorrect)")

print("\n=== corrected false-rejection rate ===")
print("(damage + nc-right) / tasks where A0 was correct -- the reported")
print("damage figure alone undercounts, since it misses harmless false rejections.\n")
for c in conds:
    s = stats[c]
    if s["n_right_a0"]:
        print(f"  {c}: {s['false_rej']}/{s['n_right_a0']} correct-A0 tasks rejected "
              f"= {100*s['false_rej']/s['n_right_a0']:.0f}%   "
              f"(harmful {s['damage']}, harmless {s['no_change_right']})")

print("\n=== exact Fisher tests between conditions (per-rejection rates) ===\n")
for a, b in combinations([c for c in conds if stats[c]["rejects"] > 0], 2):
    A, B = stats[a], stats[b]
    for label, key in (("repair rate", "repair"), ("no-change rate", None)):
        if key is None:
            xa = A["no_change_wrong"] + A["no_change_right"]
            xb = B["no_change_wrong"] + B["no_change_right"]
        else:
            xa, xb = A[key], B[key]
        na, nb = A["rejects"], B["rejects"]
        p = fisher_exact_two_sided(xa, na - xa, xb, nb - xb)
        star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else \
               "~" if p < .10 else "ns"
        print(f"  {a} vs {b}  {label:16} {xa}/{na} ({100*xa/na:.0f}%) vs "
              f"{xb}/{nb} ({100*xb/nb:.0f}%)   p = {p:.4f}  {star}")

print("\n=== selectivity ===")
for c in conds:
    s = stats[c]
    if s["repair"]:
        print(f"  {c}: {s['rejects']} rejections per {s['repair']} repairs "
              f"= {s['rejects']/s['repair']:.1f} rejections per useful fix")
    elif s["rejects"]:
        print(f"  {c}: {s['rejects']} rejections, 0 repairs")