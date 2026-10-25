"""Offline token estimation script for the 125 pre-registered problems.
Estimates prompt tokens: (statement + longest cached A0 + review template)
converted with the characters-per-token ratio measured on the five real
gpt-oss calls in groq_tokens_measured.json, and adds configured max_tokens:
  - gpt-oss: 6,144
  - llama: 2,048

Zero API calls.
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from prompts_code import CODE_REVIEW_PROMPT
from tasks.livecodebench_tasks import extract_program, load_problem_ids


def median_a0_program(cache_dir: str) -> str:
    """Median-length extracted A0 program, as in measure_groq_tokens.py."""
    programs = []
    for fname in os.listdir(cache_dir):
        if fname.endswith(".json"):
            with open(os.path.join(cache_dir, fname), "r", encoding="utf-8") as f:
                code = extract_program(json.load(f).get("response", ""))
            if code:
                programs.append(code)
    programs.sort(key=len)
    return programs[len(programs) // 2]


def measured_chars_per_token(root: str, statements: dict, cache_dir: str):
    """Rebuild the five measured review prompts and divide their characters by
    the provider-reported prompt_tokens. Returns (per-call rows, pooled ratio)."""
    with open(os.path.join(root, "run_code_data", "groq_tokens_measured.json"), encoding="utf-8") as f:
        calls = json.load(f)["top5_real_calls"]
    prog = median_a0_program(cache_dir)
    rows = []
    for c in calls:
        prompt = CODE_REVIEW_PROMPT.format(problem=statements[c["task_id"]].rstrip(), solution=prog)
        rows.append((c["task_id"], len(prompt), c["prompt_tokens"], len(prompt) / c["prompt_tokens"]))
    pooled = sum(r[1] for r in rows) / sum(r[2] for r in rows)
    return rows, pooled


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ids_path = os.path.join(root, "run_code_data", "problem_ids.json")
    problem_ids = load_problem_ids(ids_path)
    print(f"Loaded {len(problem_ids)} problem IDs from {ids_path}")

    # 1. Longest cached A0
    cache_dir = os.path.join(root, "run_code_data", "calibration_cache")
    responses = []
    if os.path.exists(cache_dir):
        for fname in os.listdir(cache_dir):
            if fname.endswith(".json"):
                with open(os.path.join(cache_dir, fname), "r", encoding="utf-8") as f:
                    responses.append(json.load(f)["response"])

    prog_path = os.path.join(root, "run_code_data", "calibration_programs.jsonl")
    prog_codes = []
    if os.path.exists(prog_path):
        for line in open(prog_path, encoding="utf-8"):
            if line.strip():
                prog_codes.append(json.loads(line)["code"])

    longest_raw_resp = max(responses, key=len) if responses else ""
    longest_extracted_code = max(prog_codes, key=len) if prog_codes else ""

    print(f"Longest raw cached A0 response: {len(longest_raw_resp)} chars")
    print(f"Longest extracted code from A0: {len(longest_extracted_code)} chars")

    # 2. Extract statements for the 125 problems from raw/
    raw_dir = os.path.join(root, "run_code_data", "raw")
    target_ids = set(problem_ids)
    statements = {}

    import re
    qid_pattern = re.compile(r'"question_id":\s*"([^"]+)"')
    file_order = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]
    for fname in file_order:
        fpath = os.path.join(raw_dir, fname)
        if not os.path.exists(fpath):
            continue
        print(f"Scanning {fname} (remaining needed: {len(target_ids)})...", flush=True)
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                m = qid_pattern.search(line[:10000])
                if m:
                    qid = m.group(1)
                    if qid in target_ids:
                        row = json.loads(line)
                        statements[qid] = row.get("question_content", "")
                        target_ids.remove(qid)
                        if not target_ids:
                            break
        print(f"After {fname}: found {len(statements)} statements, remaining: {len(target_ids)}", flush=True)
        if not target_ids:
            break

    if target_ids:
        print(f"Missing target IDs: {target_ids}", flush=True)
    print(f"Loaded statements for {len(statements)} / {len(problem_ids)} problems", flush=True)

    rows, chars_per_token = measured_chars_per_token(root, statements, cache_dir)
    print("\nMeasured chars/token (5 real gpt-oss calls, median A0 program):")
    for tid, n_chars, n_tok, ratio in rows:
        print(f"  {tid}: {n_chars} chars / {n_tok} prompt_tokens = {ratio:.3f}")
    print(f"  per-call min={min(r[3] for r in rows):.3f}, mean={sum(r[3] for r in rows)/len(rows):.3f}, max={max(r[3] for r in rows):.3f}")
    print(f"  pooled (sum chars / sum tokens) = {chars_per_token:.3f}  <- used below instead of chars//4")

    # Evaluate using both candidate definitions of longest A0:
    candidates = [
        ("Longest Extracted Code (3,170 chars)", longest_extracted_code),
        ("Longest Raw Response (5,259 chars)", longest_raw_resp),
    ]

    for label, a0_text in candidates:
        print("\n" + "=" * 65)
        print(f"EVALUATION USING: {label}")
        print("=" * 65)

        stats = []
        for pid in problem_ids:
            stmt = statements.get(pid, "")
            # Apply code_review_input stripping of instruction marker if present
            marker = "Write a complete Python 3 program"
            idx = stmt.find(marker)
            if idx != -1:
                stmt = stmt[:idx].rstrip()

            review_prompt = CODE_REVIEW_PROMPT.format(problem=stmt, solution=a0_text)
            prompt_tokens = max(1, math.ceil(len(review_prompt) / chars_per_token))

            gptoss_total = prompt_tokens + 6144
            llama_total = prompt_tokens + 2048

            stats.append({
                "task_id": pid,
                "prompt_tokens": prompt_tokens,
                "gptoss_total": gptoss_total,
                "llama_total": llama_total,
            })

        max_prompt = max(s["prompt_tokens"] for s in stats)
        max_prompt_task = next(s["task_id"] for s in stats if s["prompt_tokens"] == max_prompt)

        max_gptoss = max(s["gptoss_total"] for s in stats)
        max_gptoss_task = next(s["task_id"] for s in stats if s["gptoss_total"] == max_gptoss)
        exceed_8k = [s for s in stats if s["gptoss_total"] > 8000]
        exceed_6k_llama = [s for s in stats if s["llama_total"] > 6000]

        max_llama = max(s["llama_total"] for s in stats)
        max_llama_task = next(s["task_id"] for s in stats if s["llama_total"] == max_llama)

        print(f"Prompt Tokens: Min={min(s['prompt_tokens'] for s in stats)}, Mean={sum(s['prompt_tokens'] for s in stats)/len(stats):.1f}, Max={max_prompt} (task: {max_prompt_task})")
        print(f"Llama-3.1-8b-instant (max_tokens=2,048): Max Total Tokens = {max_llama} (task: {max_llama_task})")
        print(f"OpenAI/gpt-oss-20b (max_tokens=6,144): Max Total Tokens = {max_gptoss} (task: {max_gptoss_task})")
        print(f"Number of problems exceeding 8,000 tokens for gpt-oss: {len(exceed_8k)} / {len(problem_ids)}")
        print(f"Number of problems exceeding 6,000 tokens for llama: {len(exceed_6k_llama)} / {len(problem_ids)}")
        if exceed_8k:
            print("Problems exceeding 8,000 for gpt-oss:")
            for s in exceed_8k:
                print(f"  - {s['task_id']}: prompt={s['prompt_tokens']} + 6,144 = {s['gptoss_total']}")


if __name__ == "__main__":
    main()
