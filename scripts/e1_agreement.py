"""E1 agreement check (Amendment 1, Section 6): human vs agent specificity labels.

Reads the human labels in run_code_data/e1/human_sample_30.csv and the agent
labels in run_code_data/e1/agent_labels.json, and reports raw agreement and
Cohen's kappa on the items the human has labelled.

Run from repo root:  python scripts/e1_agreement.py
"""

import csv
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LABELS = ("specific", "vague")


def cohen_kappa(a: list, b: list) -> float:
    n = len(a)
    p_obs = sum(x == y for x, y in zip(a, b)) / n
    p_exp = sum((a.count(c) / n) * (b.count(c) / n) for c in LABELS)
    return 1.0 if p_exp == 1 else (p_obs - p_exp) / (1 - p_exp)


def main():
    human = {int(r["item_id"]): r["specificity_label"].strip().lower()
             for r in csv.DictReader(open(os.path.join(ROOT, "run_code_data", "e1", "human_sample_30.csv"), encoding="utf-8"))}
    agent = {x["item_id"]: x["label"]
             for x in json.load(open(os.path.join(ROOT, "run_code_data", "e1", "agent_labels.json"), encoding="utf-8"))}
    bad = {i: v for i, v in human.items() if v and v not in LABELS}
    if bad:
        sys.exit(f"labels must be 'specific' or 'vague'; fix items: {bad}")
    done = sorted(i for i, v in human.items() if v)
    if not done:
        sys.exit("no human labels yet: fill specificity_label in run_code_data/e1/human_sample_30.csv")
    h = [human[i] for i in done]
    g = [agent[i] for i in done]
    agree = sum(x == y for x, y in zip(h, g))
    print(f"items labelled by the human: {len(done)} of {len(human)}")
    print(f"raw agreement: {agree}/{len(done)} ({100 * agree / len(done):.1f}%)")
    print(f"Cohen's kappa: {cohen_kappa(h, g):.3f}")
    print("human:", {c: h.count(c) for c in LABELS}, "| agent:", {c: g.count(c) for c in LABELS})
    for i, x, y in zip(done, h, g):
        if x != y:
            print(f"  disagree item {i}: human={x} agent={y}")


if __name__ == "__main__":
    main()
