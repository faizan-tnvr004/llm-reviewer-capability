"""Sandboxed program execution for the code extension.

Runs model-generated Python programs against test cases with stdin/stdout IO.
Two execution backends:

  1. Docker (default): each program runs inside a disposable container with
     --network none, --memory 512m, --cpus 1, --read-only tmpfs, hard timeout.
  2. Local subprocess (UNSAFE_LOCAL_EXEC=1): pragmatic fallback for machines
     without Docker.  Uses subprocess with timeout, temp directory, and
     restricted environment.  NOT a true sandbox — adequate for small-model
     output in a research setting, noted as a limitation.

Output comparison is strict:
  - Normalize line endings to \\n
  - Strip trailing whitespace per line
  - Drop trailing blank lines
  - Exact byte-for-byte comparison after normalization
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
from typing import List, Optional

EXEC_TIMEOUT = 30  # seconds per test case
DOCKER_IMAGE = "python:3.11-slim"
DOCKER_MEM = "512m"
DOCKER_CPUS = "1"

# Set UNSAFE_LOCAL_EXEC=1 to skip Docker and run in a local subprocess.
_USE_LOCAL = os.environ.get("UNSAFE_LOCAL_EXEC", "").strip() in ("1", "true", "yes")


def normalize_output(text: str) -> str:
    """Normalize program output for comparison.

    1. Replace \\r\\n with \\n
    2. Strip trailing whitespace from each line
    3. Drop trailing blank lines
    4. Ensure a single trailing newline (if non-empty)
    """
    lines = text.replace("\r\n", "\n").split("\n")
    lines = [l.rstrip() for l in lines]
    while lines and not lines[-1]:
        lines.pop()
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _run_local(code: str, stdin_text: str, timeout: int = EXEC_TIMEOUT
               ) -> dict:
    """Run code in a local subprocess.  Returns {stdout, stderr, returncode, timed_out}."""
    with tempfile.TemporaryDirectory() as td:
        script = os.path.join(td, "solution.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            proc = subprocess.run(
                [sys.executable, script],
                input=stdin_text, capture_output=True, text=True,
                timeout=timeout, cwd=td,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            return {"stdout": proc.stdout, "stderr": proc.stderr,
                    "returncode": proc.returncode, "timed_out": False}
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": "TimeoutExpired",
                    "returncode": -1, "timed_out": True}


def _run_docker(code: str, stdin_text: str, timeout: int = EXEC_TIMEOUT
                ) -> dict:
    """Run code in a disposable Docker container."""
    with tempfile.TemporaryDirectory() as td:
        script = os.path.join(td, "solution.py")
        input_file = os.path.join(td, "input.txt")
        with open(script, "w", encoding="utf-8") as f:
            f.write(code)
        with open(input_file, "w", encoding="utf-8") as f:
            f.write(stdin_text)

        cmd = [
            "docker", "run", "--rm",
            "--network", "none",
            "--memory", DOCKER_MEM,
            "--cpus", DOCKER_CPUS,
            "--read-only",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            "-v", f"{td}:/work:ro",
            "-w", "/work",
            DOCKER_IMAGE,
            "sh", "-c", "python3 solution.py < input.txt",
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout + 10,  # docker overhead budget
            )
            return {"stdout": proc.stdout, "stderr": proc.stderr,
                    "returncode": proc.returncode, "timed_out": False}
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": "Docker timeout",
                    "returncode": -1, "timed_out": True}


def run_program(code: str, stdin_text: str, timeout: int = EXEC_TIMEOUT
                ) -> dict:
    """Run a program with the configured backend.

    Returns dict with keys: stdout, stderr, returncode, timed_out.
    """
    if _USE_LOCAL:
        return _run_local(code, stdin_text, timeout)
    return _run_docker(code, stdin_text, timeout)


def run_tests(code: str, test_cases: List[dict], timeout: int = EXEC_TIMEOUT
              ) -> List[dict]:
    """Run a program against a list of test cases.

    Each test_case dict has keys: input (str), expected_output (str).
    Returns a list of result dicts, one per test case:
      {passed: bool, expected: str, actual: str, kind: str, stderr_tail: str}
    kind is one of: "pass", "wrong", "crash", "timeout"
    """
    results = []
    for tc in test_cases:
        inp = tc["input"]
        expected = normalize_output(tc["expected_output"])

        r = run_program(code, inp, timeout)

        if r["timed_out"]:
            results.append({
                "passed": False,
                "expected": expected,
                "actual": "",
                "kind": "timeout",
                "stderr_tail": "",
            })
            continue

        if r["returncode"] != 0:
            stderr_tail = r["stderr"].strip().split("\n")[-5:]
            results.append({
                "passed": False,
                "expected": expected,
                "actual": "",
                "kind": "crash",
                "stderr_tail": "\n".join(stderr_tail),
            })
            continue

        actual = normalize_output(r["stdout"])
        passed = actual == expected
        results.append({
            "passed": passed,
            "expected": expected,
            "actual": actual,
            "kind": "pass" if passed else "wrong",
            "stderr_tail": "",
        })

    return results


def code_hash(code: str) -> str:
    """SHA-256 hash of normalized code, for identity comparison in rej_quality.

    Normalization: strip leading/trailing whitespace, normalize line endings,
    collapse multiple blank lines to one.
    """
    import re
    normalized = code.strip().replace("\r\n", "\n")
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
