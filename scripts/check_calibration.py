import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tasks.livecodebench_tasks import extract_program, code_hash, normalize_code
from harness import DiskCache

b1 = json.load(open('run_code_data/calibration/batch_1.json', encoding='utf-8'))
pids = b1['problem_ids']

log_b1 = {}
for line in open('run_code_data/calibration/log_batch_1.jsonl', encoding='utf-8'):
    if line.strip():
        r = json.loads(line)
        log_b1[r['task_id']] = r

cache = DiskCache('run_code_data/calibration_cache')

for pid in pids:
    resp = cache.get('per_execute', 'gemini-3.1-flash-lite', pid, 0)
    code = extract_program(resp)
    chash = code_hash(normalize_code(code)) if code else None
    log_chash = log_b1[pid].get('a0_code_hash')
    match = (chash == log_chash)
    correct = log_b1[pid]['final_correct']
    fp = log_b1[pid]['final_fingerprint']
    print(f"{pid:12} cache_hash={chash} log_hash={log_chash} match={match} log_correct={correct} fp={fp}")
