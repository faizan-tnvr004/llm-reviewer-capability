"""Exploratory Analysis E1: Critique specificity vs executor adoption.

Per pre-registered analysis plan:
- Restricts to LLM reviewer conditions C1–C3 (C4 deterministic test output is excluded).
- Exports double-blind hand-labeling template: rows are shuffled, condition column is hidden.
- Measures executor adoption rate (code changed: final_code_hash != a0_code_hash)
  stratified by critique specificity (specific vs vague).
- Purely descriptive / exploratory: reports counts and percentages, no significance tests.

Usage:
  Export blind template (shuffled, C1-C3 only, condition hidden):
    python analyze_critiques.py --log run_code_data/log.jsonl --export-blind-csv run_code_data/critiques_blind.csv

  Analyze hand-labeled file:
    python analyze_critiques.py --log run_code_data/log.jsonl --labeled-csv run_code_data/critiques_labeled.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path


DEFAULT_LOG = os.path.join("run_code_data", "log.jsonl")
DEFAULT_PROGRAMS = os.path.join("run_code_data", "programs.jsonl")


def parse_args():
    p = argparse.ArgumentParser(description="Analyze critique specificity vs adoption (E1)")
    p.add_argument("--log", default=DEFAULT_LOG,
                   help="Path to experiment log.jsonl")
    p.add_argument("--programs", default=DEFAULT_PROGRAMS,
                   help="Path to programs.jsonl")
    p.add_argument("--export-blind-csv", default=None,
                   help="Path to export CSV template for blind hand-labeling")
    p.add_argument("--key-file", default=None,
                   help="Path to save/load unblinding key JSON (default: <export_stem>_key.json)")
    p.add_argument("--labeled-csv", default=None,
                   help="Path to read hand-labeled CSV with 'specificity_label' column")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for shuffling blind rows (default: 42)")
    return p.parse_args()


def load_programs(programs_path: str) -> list[dict]:
    """Load programs.jsonl, de-duplicating by (task_id, condition, stage).
    If duplicates exist (e.g. from an interrupted run resumed later), the
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


def export_blind_template(rows: list[dict], out_path: str, key_path: str | None = None, seed: int = 42):
    """Export C1-C3 rejections for hand-labeling blind to condition and outcome."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    
    # Restrict strictly to C1, C2, C3 rejections
    llm_reject_rows = [
        r for r in rows
        if r.get("verdict") == "REJECT" and r.get("condition") in ("C1", "C2", "C3")
    ]

    # Deterministic shuffle to eliminate condition ordering
    rng = random.Random(seed)
    shuffled_rows = list(llm_reject_rows)
    rng.shuffle(shuffled_rows)

    if not key_path:
        p = Path(out_path)
        key_path = str(p.with_name(f"{p.stem}_key.json"))

    key_map = {}
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Condition column is explicitly HIDDEN to maintain blinding
        writer.writerow(["item_id", "task_id", "critique_text", "specificity_label", "notes"])
        for idx, r in enumerate(shuffled_rows, 1):
            critique = r.get("critique_text") or r.get("critique", "")
            writer.writerow([idx, r.get("task_id"), critique, "", ""])
            key_map[str(idx)] = {
                "condition": r.get("condition"),
                "task_id": r.get("task_id"),
                "a0_code_hash": r.get("a0_code_hash"),
                "final_code_hash": r.get("final_code_hash"),
                "a0_ast_hash": r.get("a0_ast_hash"),
                "final_ast_hash": r.get("final_ast_hash"),
                "a0_correct": r.get("a0_correct"),
                "final_correct": r.get("final_correct"),
            }

    with open(key_path, "w", encoding="utf-8") as f:
        json.dump(key_map, f, indent=2)

    print(f"[ok] Exported {len(shuffled_rows)} rejections (C1-C3 only, shuffled, condition hidden) to {out_path}")
    print(f"[ok] Saved unblinding key to {key_path}")
    print("Label each with specificity_label: 'specific' (identifies failing input/logic/trace) or 'vague' (generic).")


def analyze_labeled(rows: list[dict], labeled_csv_path: str, key_path: str | None = None):
    """Analyze adoption rate (code changed) by specificity label across C1-C3.
    Uses normalized AST hash (ast_hash) as the primary unchanged definition and reports
    text code hash alongside.
    """
    if not os.path.exists(labeled_csv_path):
        print(f"File not found: {labeled_csv_path}")
        return

    if not key_path:
        p = Path(labeled_csv_path)
        key_path = str(p.with_name(f"{p.stem}_key.json"))
        if not os.path.exists(key_path):
            key_path = str(p.with_name("critiques_blind_key.json"))

    if not os.path.exists(key_path):
        print(f"Unblinding key file not found: {key_path}. Cannot map items back to condition.")
        return

    with open(key_path, "r", encoding="utf-8") as f:
        key_map = json.load(f)

    with open(labeled_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        by_category = {}
        for row in reader:
            item_id = str(row.get("item_id", "")).strip()
            spec = row.get("specificity_label", "").strip().lower()
            if not spec:
                spec = "unlabeled"

            meta = key_map.get(item_id)
            if not meta:
                continue

            cond = meta.get("condition", "unknown")
            # Primary definition: AST hash
            if meta.get("final_ast_hash") is not None and meta.get("a0_ast_hash") is not None:
                changed_ast = (meta.get("final_ast_hash") != meta.get("a0_ast_hash"))
            else:
                changed_ast = (meta.get("final_code_hash") != meta.get("a0_code_hash"))

            changed_text = (meta.get("final_code_hash") != meta.get("a0_code_hash"))

            group = by_category.setdefault(spec, {
                "total": 0,
                "adopted_ast": 0,
                "adopted_text": 0,
                "by_cond": {},
            })
            group["total"] += 1
            if changed_ast:
                group["adopted_ast"] += 1
            if changed_text:
                group["adopted_text"] += 1

            cond_group = group["by_cond"].setdefault(cond, {
                "total": 0,
                "adopted_ast": 0,
                "adopted_text": 0,
            })
            cond_group["total"] += 1
            if changed_ast:
                cond_group["adopted_ast"] += 1
            if changed_text:
                cond_group["adopted_text"] += 1

    print("\n" + "=" * 80)
    print("EXPLORATORY ANALYSIS E1: CRITIQUE SPECIFICITY VS ADOPTION (C1-C3)")
    print("Primary identity: normalized ast_hash | Text-hash reported alongside in [brackets]")
    print("=" * 80)
    print(f"{'Specificity':16} {'Total Rejections':>18} {'Adopted (AST)':>16} {'Rate (AST)':>12} {'[Text Adopted]':>16}")
    print("-" * 80)
    for spec, stats in sorted(by_category.items()):
        tot = stats["total"]
        adp_ast = stats["adopted_ast"]
        adp_txt = stats["adopted_text"]
        rate_ast = (adp_ast / tot * 100) if tot else 0.0
        rate_txt = (adp_txt / tot * 100) if tot else 0.0
        txt_str = f"[{adp_txt}/{tot} ({rate_txt:.1f}%)]"
        print(f"{spec:16} {tot:>18} {adp_ast:>16} {rate_ast:>11.1f}% {txt_str:>16}")

    print("\nBreakdown by Condition (C1-C3):")
    for spec, stats in sorted(by_category.items()):
        print(f"\n--- Specificity: {spec} ---")
        for cond, cstats in sorted(stats["by_cond"].items()):
            c_tot = cstats["total"]
            c_ast = cstats["adopted_ast"]
            c_txt = cstats["adopted_text"]
            r_ast = (c_ast / c_tot * 100) if c_tot else 0.0
            r_txt = (c_txt / c_tot * 100) if c_tot else 0.0
            print(f"  {cond:6}: AST: {c_ast}/{c_tot} ({r_ast:.1f}%) | Text: [{c_txt}/{c_tot} ({r_txt:.1f}%)]")


def main():
    args = parse_args()
    if os.path.exists(args.programs):
        progs = load_programs(args.programs)
        print(f"Loaded {len(progs)} unique programs from {args.programs} (de-duplicated by task_id, condition, stage)")
    if not os.path.exists(args.log):
        print(f"Log file not found: {args.log}")
        return

    rows = [json.loads(l) for l in open(args.log, encoding="utf-8") if l.strip()]

    if args.export_blind_csv:
        export_blind_template(rows, args.export_blind_csv, key_path=args.key_file, seed=args.seed)
    elif args.labeled_csv:
        analyze_labeled(rows, args.labeled_csv, key_path=args.key_file)
    else:
        llm_rejects = [r for r in rows if r.get("verdict") == "REJECT" and r.get("condition") in ("C1", "C2", "C3")]
        print(f"Loaded {len(rows)} outcomes ({len(llm_rejects)} LLM rejections under C1-C3).")
        print("Use --export-blind-csv to generate labeling template or --labeled-csv to analyze.")


if __name__ == "__main__":
    main()
