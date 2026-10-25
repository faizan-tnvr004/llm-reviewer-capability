"""Sandboxed program execution for the code extension.

Runs model-generated Python programs exclusively inside disposable Docker containers:
  - image: python:3.11-slim
  - isolation: --network none, --memory 512m, --cpus 1, --pids-limit 64,
               --read-only, --tmpfs /tmp:rw,size=64m
  - security: NO API keys or host environment secrets passed to the container.
  - policy: No local fallback allowed. If Docker is unavailable, execution
    immediately stops and reports an error.

Timeout policy:
  - Per-test timeout: 6.0 s
  - Per-problem budget: min(n_hidden_tests * 6.0 s, 180.0 s)
  - Exits early on the first failing or timed-out test.

Output comparison:
  - Normalize line endings to \n
  - Strip trailing whitespace per line
  - Drop trailing blank lines
  - Exact comparison after normalization
"""

from __future__ import annotations

import ast
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from typing import List, Optional

EXEC_TIMEOUT = 6.0  # seconds per test case
MAX_PROBLEM_BUDGET = 180.0  # max seconds total per problem
DOCKER_BIN = os.environ.get("DOCKER_BIN", "docker")
DOCKER_IMAGE = os.environ.get(
    "DOCKER_IMAGE",
    "python@sha256:b7a9e5f6a93d6d4a31f129bbd4bdee69efccb6ded0fb230d78aafaf37af98377",
)
DOCKER_MEM = "512m"
DOCKER_CPUS = "1"
DOCKER_PIDS_LIMIT = "64"


def is_docker_available() -> bool:
    """Check if the docker command-line tool is available on the system."""
    return shutil.which(DOCKER_BIN) is not None or os.path.exists(DOCKER_BIN)


def normalize_output(text: str) -> str:
    """Normalize program output for comparison.

    1. Replace \r\n with \n
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


try:
    from harness import SandboxUnavailable
except ImportError:
    class SandboxUnavailable(Exception):
        """Raised when the Docker sandbox infrastructure is unavailable or repeatedly fails."""
        pass


class _HostBackstopTimeout(Exception):
    """Internal sentinel for host-side subprocess.TimeoutExpired without exit 124."""
    pass


class _DockerInfraFailure(Exception):
    """Internal sentinel for docker daemon / CLI / container start failures."""
    pass


INFRA_ERROR_STRINGS = (
    "Cannot connect to the Docker daemon",
    "error during connect",
    "docker: Error",
)


def _is_infra_error(returncode: int, stderr: str) -> bool:
    if returncode in (125, 126, 127):
        return True
    stderr_lower = stderr.lower()
    for s in INFRA_ERROR_STRINGS:
        if s.lower() in stderr_lower or s in stderr:
            return True
    return False


def _run_docker_single_attempt(code: str, stdin_text: str, timeout: float = EXEC_TIMEOUT) -> dict:
    if not is_docker_available():
        raise _DockerInfraFailure(f"Docker binary '{DOCKER_BIN}' not found or unavailable.")

    container_name = f"code_exec_{os.getpid()}_{time.time_ns()}_{uuid.uuid4().hex[:8]}"

    with tempfile.TemporaryDirectory() as td:
        script = os.path.join(td, "solution.py")
        input_file = os.path.join(td, "input.txt")
        with open(script, "w", encoding="utf-8") as f:
            f.write(code)
        with open(input_file, "w", encoding="utf-8") as f:
            f.write(stdin_text)
        inner_cmd = f"timeout {timeout} python3 solution.py < input.txt"
        cmd = [
            DOCKER_BIN, "run", "--rm",
            "--name", container_name,
            "--network", "none",
            "--memory", DOCKER_MEM,
            "--cpus", DOCKER_CPUS,
            "--pids-limit", DOCKER_PIDS_LIMIT,
            "--read-only",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            "-v", f"{td}:/work:ro",
            "-w", "/work",
            DOCKER_IMAGE,
            "sh", "-c", inner_cmd,
        ]
        clean_env = {"PATH": os.environ.get("PATH", ""), "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout + 5.0,  # host-side backstop (docker startup budget)
                env=clean_env,
            )
            if _is_infra_error(proc.returncode, proc.stderr):
                raise _DockerInfraFailure(f"Docker CLI/daemon failure (exit {proc.returncode}): {proc.stderr[:200]}")

            if proc.returncode == 124:
                return {
                    "stdout": proc.stdout,
                    "stderr": proc.stderr or f"Inner timeout (exit 124 after {timeout}s)",
                    "returncode": 124,
                    "timed_out": True,
                }
            return {
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired:
            subprocess.run([DOCKER_BIN, "rm", "-f", container_name], capture_output=True)
            raise _HostBackstopTimeout("Host-side subprocess.TimeoutExpired before container exit")
        except (FileNotFoundError, PermissionError, OSError) as e:
            raise _DockerInfraFailure(f"Docker binary '{DOCKER_BIN}' invocation failed: {e}")
        finally:
            subprocess.run([DOCKER_BIN, "rm", "-f", container_name], capture_output=True)


def _run_docker(code: str, stdin_text: str, timeout: float = EXEC_TIMEOUT, backoff_s: float = 10.0) -> dict:
    """Run code in Docker sandbox with retry logic for infrastructure failures.

    (a) Host-side TimeoutExpired: retry test once in fresh container; score TIMEOUT only
        if retry also hits backstop or returns 124.
    (b)/(c) Docker CLI / start failures: retry up to 2 times with 10s backoff, then raise SandboxUnavailable.
    Never score infrastructure failure as CRASH, FAIL, or TIMEOUT.
    """
    infra_retries = 0
    max_infra_retries = 2

    for attempt in range(max_infra_retries + 1):
        try:
            try:
                res = _run_docker_single_attempt(code, stdin_text, timeout)
                res["infra_retries"] = infra_retries
                return res
            except _HostBackstopTimeout:
                # Path (a): retry test once in fresh container
                infra_retries += 1
                try:
                    res_retry = _run_docker_single_attempt(code, stdin_text, timeout)
                    res_retry["infra_retries"] = infra_retries
                    return res_retry
                except _HostBackstopTimeout:
                    return {
                        "stdout": "",
                        "stderr": "Docker host backstop timeout on retry",
                        "returncode": -1,
                        "timed_out": True,
                        "infra_retries": infra_retries,
                    }
        except _DockerInfraFailure as e:
            # Path (b)/(c): retry up to 2 times with backoff, then raise SandboxUnavailable
            if attempt < max_infra_retries:
                infra_retries += 1
                if backoff_s > 0:
                    time.sleep(backoff_s)
                continue
            raise SandboxUnavailable(
                f"Docker sandbox unavailable after {max_infra_retries} retries: {e}"
            ) from e

    raise SandboxUnavailable(f"Docker sandbox unavailable: exhausted {max_infra_retries} retries")


def cleanup_stale_containers() -> int:
    """Sweep and remove any stale Docker containers named 'code_exec_*'.
    Returns the number of removed containers.
    """
    if not is_docker_available():
        return 0
    try:
        proc = subprocess.run(
            [DOCKER_BIN, "ps", "-a", "--filter", "name=code_exec_", "--format", "{{.Names}}"],
            capture_output=True, text=True, check=False,
        )
        names = [n.strip() for n in proc.stdout.splitlines() if n.strip()]
        if not names:
            return 0
        subprocess.run([DOCKER_BIN, "rm", "-f"] + names, capture_output=True, check=False)
        print(f"Swept {len(names)} stale Docker container(s): {', '.join(names[:5])}{'...' if len(names) > 5 else ''}")
        return len(names)
    except Exception as e:
        print(f"Warning: Failed to sweep stale containers: {e}")
        return 0


def run_program(code: str, stdin_text: str, timeout: float = EXEC_TIMEOUT, backoff_s: float = 10.0) -> dict:
    """Run a program inside the Docker sandbox."""
    return _run_docker(code, stdin_text, timeout, backoff_s=backoff_s)


def run_tests(code: str, test_cases: List[dict],
              per_test_timeout_s: float = EXEC_TIMEOUT,
              max_budget_s: float = MAX_PROBLEM_BUDGET,
              fail_fast: bool = True,
              backoff_s: float = 10.0) -> List[dict]:
    """Run code against test cases with a per-problem budget and early exit.

    Budget: min(n_hidden_tests * per_test_timeout_s, max_budget_s).
    Exits early on the first failing or timed-out test when fail_fast=True.
    """
    total_budget = min(len(test_cases) * per_test_timeout_s, max_budget_s)
    start_time = time.time()
    results = []

    for tc in test_cases:
        inp = tc.get("input", "")
        expected = normalize_output(tc.get("expected_output", tc.get("output", "")))

        elapsed_total = time.time() - start_time
        remaining_budget = total_budget - elapsed_total

        if remaining_budget <= 0:
            results.append({
                "passed": False,
                "expected": expected,
                "actual": "",
                "kind": "timeout",
                "stderr_tail": f"Per-problem budget ({total_budget:.1f}s) exhausted",
                "wall_s": 0.0,
            })
            if fail_fast:
                break
            continue

        test_timeout = min(per_test_timeout_s, remaining_budget)
        t_test_start = time.perf_counter()
        r = run_program(code, inp, timeout=test_timeout, backoff_s=backoff_s)
        test_wall = time.perf_counter() - t_test_start

        retries = r.get("infra_retries", 0)

        if r["timed_out"]:
            results.append({
                "passed": False,
                "expected": expected,
                "actual": "",
                "kind": "timeout",
                "stderr_tail": r.get("stderr", "Test timeout"),
                "wall_s": round(test_wall, 3),
                "infra_retries": retries,
            })
            if fail_fast:
                break
            continue

        if r["returncode"] != 0:
            stderr_tail = r["stderr"].strip().split("\n")[-5:]
            results.append({
                "passed": False,
                "expected": expected,
                "actual": "",
                "kind": "crash",
                "stderr_tail": "\n".join(stderr_tail),
                "wall_s": round(test_wall, 3),
                "infra_retries": retries,
            })
            if fail_fast:
                break
            continue

        actual = normalize_output(r["stdout"])
        passed = (actual == expected)
        results.append({
            "passed": passed,
            "expected": expected,
            "actual": actual,
            "kind": "pass" if passed else "wrong",
            "stderr_tail": "",
            "wall_s": round(test_wall, 3),
            "infra_retries": retries,
        })

        if not passed and fail_fast:
            break

    return results


def code_hash(code: str) -> str:
    """SHA-256 hash of normalized code, for identity comparison in rej_quality."""
    import re
    normalized = code.strip().replace("\r\n", "\n")
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


class _DocstringRemover(ast.NodeTransformer):
    def _strip_doc(self, body):
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            return body[1:]
        return body

    def visit_Module(self, node):
        self.generic_visit(node)
        node.body = self._strip_doc(node.body)
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.body = self._strip_doc(node.body)
        return node

    def visit_AsyncFunctionDef(self, node):
        self.generic_visit(node)
        node.body = self._strip_doc(node.body)
        return node

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        node.body = self._strip_doc(node.body)
        return node


def ast_hash(code: str) -> str:
    """Compute 16-hex SHA-256 hash of normalized AST.

    1. Parse with ast.parse (comments are naturally ignored).
    2. Strip docstrings from modules, functions, classes.
    3. Serialize with ast.dump(include_attributes=False).
    4. Fall back to code_hash(code) on SyntaxError.
    """
    if not code or not code.strip():
        return ""
    try:
        tree = ast.parse(code)
        _DocstringRemover().visit(tree)
        ast.fix_missing_locations(tree)
        dumped = ast.dump(tree, include_attributes=False)
        return hashlib.sha256(dumped.encode("utf-8")).hexdigest()[:16]
    except SyntaxError:
        return code_hash(code)

