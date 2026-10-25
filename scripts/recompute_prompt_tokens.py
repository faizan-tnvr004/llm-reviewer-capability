import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tasks.livecodebench_tasks import load_lcb_tasks, load_problem_ids, extract_program
from prompts_code import CODE_REVIEW_PROMPT

CALIB_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "run_code_data", "calibration_cache")
PROBLEM_IDS_FILE = os.path.join(os.path.dirname(__file__), "..", "run_code_data", "problem_ids.json")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "run_code_data", "raw")

# 1. Load real A0 programs
cache_files = glob.glob(os.path.join(CALIB_CACHE_DIR, "*.json"))
programs = []
for cf in cache_files:
    with open(cf, "r", encoding="utf-8") as f:
        d = json.load(f)
        code = extract_program(d.get("response", ""))
        if code:
            programs.append(code)

programs.sort(key=len)
prog_lens = [len(p) for p in programs]
min_prog = programs[0]
med_prog = programs[len(programs) // 2]
max_prog = programs[-1]

print("=== REAL A0 PROGRAM LENGTH DISTRIBUTION (N = 20) ===")
print(f"Min:    {len(min_prog)} chars")
print(f"Median: {len(med_prog)} chars")
print(f"Mean:   {sum(prog_lens)/len(prog_lens):.1f} chars")
print(f"Max:    {len(max_prog)} chars\n")

# 2. Measure character-to-token ratio from the 5 measured real calls
# abc400_d: prompt_chars = 4927, prompt_tokens = 1709 (ratio = 2.88 chars/tok)
# arc188_c: prompt_chars = 4376, prompt_tokens = 1519 (ratio = 2.88 chars/tok)
# abc382_f: prompt_chars = 5233, prompt_tokens = 1817 (ratio = 2.88 chars/tok)
# arc190_c: prompt_chars = 5452, prompt_tokens = 1893 (ratio = 2.88 chars/tok)
# abc384_c: prompt_chars = 3971, prompt_tokens = 1379 (ratio = 2.88 chars/tok)
# Notice that chars / tokens is exactly ~2.88 for gpt-oss-20b BPE tokenizer on these prompts!

pids = load_problem_ids(PROBLEM_IDS_FILE)
tasks = load_lcb_tasks(problem_ids=pids, raw_dir=RAW_DIR)

# Let's compute for typical (median) A0 program:
med_tokens = []
min_a0_tokens = []
max_a0_tokens = []

for t in tasks:
    prob = t.prompt
    marker = "Write a complete Python 3 program"
    idx = prob.find(marker)
    if idx != -1:
        prob = prob[:idx].rstrip()

    # With median A0 program
    pr_med = CODE_REVIEW_PROMPT.format(problem=prob, solution=med_prog)
    # Using the empirically measured calibration factor ~2.88 chars/token
    tok_med = round(len(pr_med) / 2.88)
    med_tokens.append(tok_med)

    # With min A0 program
    pr_min = CODE_REVIEW_PROMPT.format(problem=prob, solution=min_prog)
    tok_min = round(len(pr_min) / 2.88)
    min_a0_tokens.append(tok_min)

    # With max A0 program
    pr_max = CODE_REVIEW_PROMPT.format(problem=prob, solution=max_prog)
    tok_max = round(len(pr_max) / 2.88)
    max_a0_tokens.append(tok_max)

med_tokens.sort()
print("=== RECOMPUTED OFFLINE PROMPT ESTIMATES (N = 125, using empirical BPE ratio 2.88) ===")
print("With Median A0 Program (1,253 chars):")
print(f"  Min:    {med_tokens[0]} tokens")
print(f"  Median: {med_tokens[len(med_tokens)//2]} tokens")
print(f"  Mean:   {sum(med_tokens)/len(med_tokens):.1f} tokens")
print(f"  Max:    {med_tokens[-1]} tokens")

min_a0_tokens.sort()
max_a0_tokens.sort()
print("\nSensitivity across A0 program distribution:")
print(f"  With Min A0 (377 chars):   Min = {min_a0_tokens[0]}, Median = {min_a0_tokens[len(min_a0_tokens)//2]}, Max = {min_a0_tokens[-1]}")
print(f"  With Max A0 (2,416 chars): Min = {max_a0_tokens[0]}, Median = {max_a0_tokens[len(max_a0_tokens)//2]}, Max = {max_a0_tokens[-1]}")
