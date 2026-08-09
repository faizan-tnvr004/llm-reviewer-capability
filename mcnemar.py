"""McNemar's test on the paired reviewer-condition comparisons.

The design is PAIRED: the same 100 problems are solved once and reviewed under
every condition. McNemar's test is the correct primary test for paired binary
outcomes -- it conditions on the DISCORDANT pairs (problems where the two
conditions disagree) and ignores the concordant ones, which is exactly what
gives paired designs more power than raw N suggests.

For each pair of conditions (A, B):
  b = tasks A correct, B wrong
  c = tasks A wrong,   B correct
  Under H0 (no difference), b and c are draws from Binomial(b+c, 0.5).
  Exact two-sided binomial test -- no chi-square approximation, which is
  unreliable when b+c is small (it is, here).

Also reports the paired accuracy difference and an exact 95% CI on the
proportion of discordant pairs favouring B.

Run from repo root:  python mcnemar.py
No API calls; reads the existing log.
"""
import json, os
from itertools import combinations
from math import comb

LOG = os.path.join("run_experiment_data", "log.jsonl")
rows = [json.loads(l) for l in open(LOG, encoding="utf-8") if l.strip()]

# task_id -> condition -> final_correct
by_task = {}
for r in rows:
    by_task.setdefault(r["task_id"], {})[r["condition"]] = bool(r["final_correct"])

conds = sorted({r["condition"] for r in rows})
# keep only tasks present in EVERY condition (true paired set)
complete = {t: d for t, d in by_task.items() if all(c in d for c in conds)}
print(f"{len(by_task)} tasks in log; {len(complete)} complete across all "
      f"{len(conds)} conditions (paired set)\n")


def exact_binom_two_sided(b, c):
    """Exact two-sided binomial p-value for b successes in n=b+c at p=0.5."""
    n = b + c
    if n == 0:
        return 1.0
    # two-sided: sum of probabilities of outcomes at least as extreme
    obs = comb(n, b)
    total = 2 ** n
    p = sum(comb(n, k) for k in range(n + 1) if comb(n, k) <= obs) / total
    return min(1.0, p)


print(f"{'comparison':16} {'acc A':>6} {'acc B':>6} {'diff':>7} "
      f"{'b':>4} {'c':>4} {'n_disc':>7} {'p (exact)':>10}  verdict")
print("-" * 86)

for a, b_ in combinations(conds, 2):
    accA = sum(d[a] for d in complete.values()) / len(complete)
    accB = sum(d[b_] for d in complete.values()) / len(complete)
    # b: A right, B wrong | c: A wrong, B right
    b = sum(1 for d in complete.values() if d[a] and not d[b_])
    c = sum(1 for d in complete.values() if not d[a] and d[b_])
    p = exact_binom_two_sided(b, c)
    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else \
           "~" if p < 0.10 else "ns"
    print(f"{a+' vs '+b_:16} {accA:>6.2f} {accB:>6.2f} {accB-accA:>+7.2f} "
          f"{b:>4} {c:>4} {b+c:>7} {p:>10.4f}  {star}")

print("\nb = A correct & B wrong   c = A wrong & B correct")
print("Only discordant pairs (b+c) carry information; concordant pairs cancel.")
print("*** p<.001  ** p<.01  * p<.05  ~ p<.10  ns = not significant")
print("\nPrimary pre-registered comparison: each review condition vs C0.")