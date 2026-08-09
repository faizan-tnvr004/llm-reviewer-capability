"""Compute the third C1 failure mode: 'rejection without change'.

Among tasks where the reviewer REJECTED, what fraction of revisions returned
the IDENTICAL answer? This is distinct from repair (wrong->right) and damage
(right->wrong): the critique fires and the pipeline absorbs it as a no-op.

Run from repo root:  python no_change.py
"""
import json, os

LOG = os.path.join("run_experiment_data", "log.jsonl")
rows = [json.loads(l) for l in open(LOG, encoding="utf-8") if l.strip()]

print(f"{len(rows)} rows\n")
print(f"{'cond':6} {'rejects':>8} {'no-change':>10} {'repair':>8} {'damage':>8} {'other':>7}")
print("-" * 52)

for cond in sorted({r["condition"] for r in rows}):
    rs = [r for r in rows if r["condition"] == cond]
    rejects = [r for r in rs if r["verdict"] == "REJECT"]
    if not rejects:
        print(f"{cond:6} {0:>8}")
        continue

    no_change = repair = damage = other = 0
    for r in rejects:
        same = (r["final_extracted"] == r["a0_extracted"])
        if same:
            no_change += 1
        elif not r["a0_correct"] and r["final_correct"]:
            repair += 1
        elif r["a0_correct"] and not r["final_correct"]:
            damage += 1
        else:
            other += 1          # changed the answer, still wrong (wrong->wrong)

    n = len(rejects)
    def pct(x): return f"{x}/{n} ({100*x/n:.0f}%)"
    print(f"{cond:6} {n:>8} {pct(no_change):>10} {pct(repair):>8} "
          f"{pct(damage):>8} {pct(other):>7}")

print("\nno-change = rejected, revision returned the IDENTICAL answer")
print("repair    = rejected a wrong answer, revision fixed it")
print("damage    = rejected a correct answer, revision broke it")
print("other     = changed the answer, still wrong (wrong -> different wrong)")