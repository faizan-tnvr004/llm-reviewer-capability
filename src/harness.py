"""Run harness: reliable, resumable, logged execution of model calls.

Responsibilities:
  1. Caching  — every (condition, model, task, sample) call is cached to disk;
                interrupted runs resume without re-spending API quota.
  2. Backoff  — per-minute rate limits are retried with exponential backoff;
                per-DAY limit exhaustion exits gracefully (restart tomorrow,
                cache intact).
  3. Logging  — every completed call appends one JSONL record with everything
                the analysis needs: raw response, extracted answer, correct,
                fingerprint, model, task, sample, condition.

Provider-agnostic: models are registered as plain functions
    call(prompt: str) -> str
so the harness is fully testable with a mock. Real providers (Groq, Google
AI Studio) are wired in via providers.py on the machine that holds the keys.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from core import Task, grade_numeric


# ---------------------------------------------------------------------------
# Rate-limit signalling
# ---------------------------------------------------------------------------

class RetryableRateLimit(Exception):
    """Per-minute (RPM/TPM) limit: wait and retry."""


class DailyLimitExhausted(Exception):
    """Per-day (RPD/TPD) limit: retrying is pointless until the daily reset."""


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass
class Model:
    model_id: str                     # exact API id, recorded in every log row
    call: Callable[[str], str]        # prompt -> raw response text


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

class DiskCache:
    """One JSON file per completed call, keyed by a hash of
    (condition, model_id, task_id, sample_idx). Existence of the file means
    the call is done; its content is the raw response."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, condition: str, model_id: str, task_id: str, sample_idx: int) -> str:
        key = f"{condition}|{model_id}|{task_id}|{sample_idx}"
        h = hashlib.sha256(key.encode()).hexdigest()[:24]
        return os.path.join(self.root, f"{h}.json")

    def get(self, condition: str, model_id: str, task_id: str, sample_idx: int) -> Optional[str]:
        p = self._path(condition, model_id, task_id, sample_idx)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)["response"]
        return None

    def put(self, condition: str, model_id: str, task_id: str, sample_idx: int, response: str) -> None:
        p = self._path(condition, model_id, task_id, sample_idx)
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"key": f"{condition}|{model_id}|{task_id}|{sample_idx}",
                       "response": response}, f)
        os.replace(tmp, p)  # atomic: a crash never leaves a half-written entry


# ---------------------------------------------------------------------------
# Logging (JSONL: one record per completed call)
# ---------------------------------------------------------------------------

class RunLog:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def append(self, record: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def load(self) -> List[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

def call_with_retry(model: Model, prompt: str,
                    max_retries: int = 10,
                    base_wait: float = 2.0,
                    sleep: Callable[[float], None] = time.sleep) -> str:
    """Call the model; on RetryableRateLimit wait 2s, 4s, 8s, ... and retry.
    DailyLimitExhausted propagates immediately — the runner handles it."""
    attempt = 0
    while True:
        try:
            return model.call(prompt)
        except RetryableRateLimit:
            if attempt >= max_retries:
                raise
            wait = base_wait * (2 ** attempt)
            sleep(wait)
            attempt += 1


# ---------------------------------------------------------------------------
# The single-agent run loop
# ---------------------------------------------------------------------------

def run_single_agent(tasks: List[Task],
                     models: List[Model],
                     n_samples: int,
                     cache: DiskCache,
                     log: RunLog,
                     condition: str = "single_agent",
                     sleep: Callable[[float], None] = time.sleep,
                     verbose: bool = True) -> dict:
    """Run every (model, task, sample) combination once, resumably.

    Returns a summary dict: calls made, cache hits, whether a daily limit
    stopped the run early. Restarting after a daily stop (or crash) resumes
    exactly where it left off — completed calls are served from cache and,
    if not yet logged, logged from cache without an API call.
    """
    # Work out what's already logged so we never write duplicate records.
    already_logged = {(r["condition"], r["model_id"], r["task_id"], r["sample_idx"])
                      for r in log.load()}

    made, hits, logged = 0, 0, 0
    for model in models:
        for task in tasks:
            for s in range(n_samples):
                key = (condition, model.model_id, task.task_id, s)

                response = cache.get(*key)
                if response is None:
                    try:
                        response = call_with_retry(model, task.prompt, sleep=sleep)
                    except DailyLimitExhausted:
                        if verbose:
                            print(f"[harness] daily limit hit on {model.model_id} "
                                  f"after {made} new calls — stopping gracefully. "
                                  f"Re-run after the daily reset to resume.")
                        return {"made": made, "cache_hits": hits, "logged": logged,
                                "stopped_daily_limit": True}
                    cache.put(*key, response)
                    made += 1
                else:
                    hits += 1

                if key not in already_logged:
                    graded = grade_numeric(task, response)
                    log.append({
                        "condition": condition,
                        "model_id": model.model_id,
                        "task_id": task.task_id,
                        "family": task.family,
                        "sample_idx": s,
                        "raw_response": response,
                        "extracted": graded["extracted"],
                        "correct": graded["correct"],
                        "fingerprint": graded["fingerprint"],
                        "ground_truth": task.ground_truth,
                        "template": task.meta.get("template"),
                        "difficulty": task.meta.get("difficulty"),
                        "ts": time.time(),
                    })
                    already_logged.add(key)
                    logged += 1

    return {"made": made, "cache_hits": hits, "logged": logged,
            "stopped_daily_limit": False}
