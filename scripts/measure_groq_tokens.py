"""Measure Groq token counts for C2 and C3 review prompts.

1. Offline (0 API calls):
   Estimate prompt token counts for all 125 problems with a typical A0 program.
   Reports min, median, max.
2. 5 real gpt-oss-20b review calls on the 5 longest prompts:
   Reports prompt_tokens, completion_tokens (including reasoning), empty responses, wall time.
3. Computes updated tokens-per-task figures.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
import time
from typing import Dict, List, Tuple

# Load .env
if os.path.exists(".env"):
    with open(".env", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))

import urllib.request
import urllib.error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tasks.livecodebench_tasks import load_lcb_tasks, load_problem_ids, extract_program
from prompts_code import CODE_REVIEW_PROMPT
from pipeline import strip_answer_instruction

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "run_code_data", "raw")
PROBLEM_IDS_FILE = os.path.join(os.path.dirname(__file__), "..", "run_code_data", "problem_ids.json")
CALIB_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "run_code_data", "calibration_cache")


def get_typical_a0_program() -> str:
    """Find the median-length A0 program from the calibration cache."""
    cache_files = glob.glob(os.path.join(CALIB_CACHE_DIR, "*.json"))
    programs = []
    for cf in cache_files:
        with open(cf, "r", encoding="utf-8") as f:
            d = json.load(f)
            code = extract_program(d.get("response", ""))
            if code:
                programs.append(code)

    programs.sort(key=len)
    median_prog = programs[len(programs) // 2]
    return median_prog


def main():
    typical_prog = get_typical_a0_program()
    print(f"Typical A0 program length: {len(typical_prog)} chars (~{len(typical_prog)//4} est. tokens)\n")

    pids = load_problem_ids(PROBLEM_IDS_FILE)
    tasks = load_lcb_tasks(problem_ids=pids, raw_dir=RAW_DIR)

    prompts_data = []
    for t in tasks:
        prob = t.prompt
        marker = "Write a complete Python 3 program"
        idx = prob.find(marker)
        if idx != -1:
            prob = prob[:idx].rstrip()

        prompt_text = CODE_REVIEW_PROMPT.format(problem=prob, solution=typical_prog)
        est_tokens = len(prompt_text) // 4
        prompts_data.append({
            "task_id": t.task_id,
            "difficulty": t.meta.get("difficulty", ""),
            "prompt_chars": len(prompt_text),
            "est_tokens": est_tokens,
            "prompt_text": prompt_text,
        })

    # 1. Offline summary
    all_est = [d["est_tokens"] for d in prompts_data]
    all_est_sorted = sorted(all_est)
    min_tokens = all_est_sorted[0]
    median_tokens = all_est_sorted[len(all_est_sorted) // 2]
    mean_tokens = sum(all_est) / len(all_est)
    max_tokens = all_est_sorted[-1]

    print("=" * 70)
    print("OFFLINE PROMPT TOKEN ESTIMATES (N = 125, C2/C3 Review Prompts)")
    print("=" * 70)
    print(f"Sample size:     {len(prompts_data)} problems")
    print(f"Min Tokens:      {min_tokens}")
    print(f"Median Tokens:   {median_tokens}")
    print(f"Mean Tokens:     {mean_tokens:.1f}")
    print(f"Max Tokens:      {max_tokens}")
    print("=" * 70)

    # 2. Select 5 longest prompts
    prompts_data.sort(key=lambda d: d["prompt_chars"], reverse=True)
    top5 = prompts_data[:5]

    print("\nTop 5 Longest Review Prompts:")
    for idx, d in enumerate(top5, 1):
        print(f"  {idx}. {d['task_id']} ({d['difficulty']}): {d['prompt_chars']} chars (~{d['est_tokens']} est. tokens)")

    # 3. Real gpt-oss-20b calls via Groq API
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        print("\nERROR: GROQ_API_KEY not found in environment or .env. Cannot make real calls.")
        return

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": "application/json",
        "User-Agent": "MultiagentErrorDiversity/1.0",
    }
    results = []

    print("\n" + "=" * 80)
    print(f"{'Task ID':12} {'Prompt Tok':12} {'Compl Tok':12} {'Reason Tok':12} {'Empty?':8} {'Wall Time':10}")
    print("=" * 80)

    for idx, d in enumerate(top5, 1):
        prompt = d["prompt_text"]
        payload = {
            "model": "openai/gpt-oss-20b",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 6144,
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")

        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=60.0) as resp:
            raw_body = resp.read().decode("utf-8")
        elapsed = time.perf_counter() - t0

        data = json.loads(raw_body)
        choices = data.get("choices", [])
        choice = choices[0] if choices else {}
        message = choice.get("message", {})
        content = message.get("content", "") or ""
        is_empty = (not content.strip())

        usage = data.get("usage", {})
        prompt_tok = usage.get("prompt_tokens")
        compl_tok = usage.get("completion_tokens")

        compl_details = usage.get("completion_tokens_details", {}) or {}
        reason_tok = compl_details.get("reasoning_tokens", 0) or 0
        if not reason_tok and "reasoning" in message:
            reason_tok = len(message.get("reasoning", "")) // 4


        results.append({
            "task_id": d["task_id"],
            "difficulty": d["difficulty"],
            "prompt_tokens": prompt_tok,
            "completion_tokens": compl_tok,
            "reasoning_tokens": reason_tok,
            "empty": is_empty,
            "wall_time": elapsed,
            "content_preview": content[:120].replace("\n", " "),
        })

        print(f"{d['task_id']:12} {prompt_tok:<12} {compl_tok:<12} {reason_tok:<12} {'YES' if is_empty else 'NO':8} {elapsed:8.2f}s")
        if idx < len(top5):
            time.sleep(2.0)  # brief courtesy pause

    print("=" * 80)

    # 4. Summary of real calls
    empty_count = sum(1 for r in results if r["empty"])
    prompt_tokens_list = [r["prompt_tokens"] for r in results if r["prompt_tokens"] is not None]
    compl_tokens_list = [r["completion_tokens"] for r in results if r["completion_tokens"] is not None]
    reason_tokens_list = [r["reasoning_tokens"] for r in results]
    wall_times = [r["wall_time"] for r in results]

    print("\nREAL CALLS SUMMARY (5 Longest Prompts on openai/gpt-oss-20b):")
    print(f"Total calls:             5")
    print(f"Empty response count:    {empty_count} / 5 ({empty_count/5*100:.0f}%)")
    print(f"Prompt tokens:           min={min(prompt_tokens_list)}, median={sorted(prompt_tokens_list)[2]}, max={max(prompt_tokens_list)}, mean={sum(prompt_tokens_list)/len(prompt_tokens_list):.1f}")
    print(f"Completion tokens:       min={min(compl_tokens_list)}, median={sorted(compl_tokens_list)[2]}, max={max(compl_tokens_list)}, mean={sum(compl_tokens_list)/len(compl_tokens_list):.1f}")
    print(f"Reasoning tokens:        min={min(reason_tokens_list)}, median={sorted(reason_tokens_list)[2]}, max={max(reason_tokens_list)}, mean={sum(reason_tokens_list)/len(reason_tokens_list):.1f}")
    print(f"Wall times:              min={min(wall_times):.2f}s, median={sorted(wall_times)[2]:.2f}s, max={max(wall_times):.2f}s, mean={sum(wall_times)/len(wall_times):.2f}s")

    # Save to json for reference
    out_file = os.path.join(os.path.dirname(__file__), "..", "run_code_data", "groq_tokens_measured.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "offline_estimates": {
                "min": min_tokens,
                "median": median_tokens,
                "mean": round(mean_tokens, 1),
                "max": max_tokens,
            },
            "top5_real_calls": results,
        }, f, indent=2)
    print(f"\nSaved measurements to {out_file}")


if __name__ == "__main__":
    main()
