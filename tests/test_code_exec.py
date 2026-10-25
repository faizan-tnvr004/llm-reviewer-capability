"""Tests for code_exec.py — output normalization and sandboxed execution.

Run:  pytest tests/test_code_exec.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from code_exec import normalize_output, code_hash


def test_normalize_output_basic():
    """CRLF -> LF, trailing whitespace stripped, trailing blanks dropped."""
    assert normalize_output("hello\r\nworld\r\n") == "hello\nworld\n"
    assert normalize_output("hello  \nworld  \n") == "hello\nworld\n"
    assert normalize_output("hello\n\n\n") == "hello\n"
    assert normalize_output("") == ""
    assert normalize_output("\n\n") == ""


def test_normalize_output_preserves_internal_blanks():
    """Internal blank lines are preserved."""
    assert normalize_output("a\n\nb\n") == "a\n\nb\n"


def test_normalize_output_adds_trailing_newline():
    """Non-empty output always ends with exactly one newline."""
    assert normalize_output("hello") == "hello\n"
    assert normalize_output("hello\n") == "hello\n"


def test_code_hash_deterministic():
    """Same code -> same hash, different code -> different hash."""
    code1 = "print('hello')\n"
    code2 = "print('world')\n"
    assert code_hash(code1) == code_hash(code1)
    assert code_hash(code1) != code_hash(code2)


def test_code_hash_normalizes_whitespace():
    """Leading/trailing whitespace and multiple blank lines don't change hash."""
    code_a = "  print('hello')  \n"
    code_b = "print('hello')\n\n\n\n"
    assert code_hash(code_a) == code_hash(code_b)


def test_code_hash_length():
    """Hash is 16 hex characters."""
    h = code_hash("print('x')")
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


from code_exec import normalize_output, code_hash, ast_hash


def test_ast_hash_comment_only():
    """Comment additions/edits do not alter ast_hash (though text hash differs)."""
    code_orig = "def solve():\n    x = 1\n    return x\n"
    code_comment = "# Calculate the answer\ndef solve():\n    # Initialize x\n    x = 1  # inline comment\n    return x\n"
    assert code_hash(code_orig) != code_hash(code_comment)
    assert ast_hash(code_orig) == ast_hash(code_comment)


def test_ast_hash_docstring_only():
    """Docstrings inside module, class, or function are stripped."""
    code_orig = "def solve():\n    return 42\n"
    code_doc = 'def solve():\n    """Return the constant answer."""\n    return 42\n'
    assert ast_hash(code_orig) == ast_hash(code_doc)


def test_ast_hash_rename_changes():
    """Renaming an identifier alters ast_hash."""
    code_a = "def solve():\n    total_sum = 0\n    return total_sum\n"
    code_b = "def solve():\n    ans = 0\n    return ans\n"
    assert ast_hash(code_a) != ast_hash(code_b)


def test_ast_hash_real_logic_changes():
    """Altering constants, operations, or statements alters ast_hash."""
    code_a = "def solve(n):\n    return n * 2\n"
    code_b = "def solve(n):\n    return n * 3\n"
    assert ast_hash(code_a) != ast_hash(code_b)


def test_ast_hash_syntax_error_fallback():
    """Invalid syntax falls back to code_hash without raising SyntaxError."""
    bad_code = "def solve( incomplete syntax..."
    h = ast_hash(bad_code)
    assert h == code_hash(bad_code)
    assert len(h) == 16


# ---------------------------------------------------------------------------
# Infrastructure Failure & Sandbox Tests (Round 10 Requirements)
# ---------------------------------------------------------------------------

from unittest.mock import patch, MagicMock
import pytest
import json
import subprocess
import tempfile
from code_exec import (
    _run_docker,
    _is_infra_error,
    _HostBackstopTimeout,
    _DockerInfraFailure,
    SandboxUnavailable,
    run_program,
    run_tests,
)
from tasks.livecodebench_tasks import grade_lcb
from core import Task
import run_experiment_code
from run_experiment_code import is_git_clean, append_run_manifest


def test_infra_path_a_host_backstop_retry_succeeds():
    """(a) Host backstop TimeoutExpired: retry test once in fresh container; succeeds on retry."""
    with patch("code_exec._run_docker_single_attempt") as mock_attempt:
        mock_attempt.side_effect = [
            _HostBackstopTimeout("Host timeout attempt 1"),
            {"stdout": "res_ok\n", "stderr": "", "returncode": 0, "timed_out": False},
        ]
        res = _run_docker("print('ok')", "", timeout=6.0, backoff_s=0.0)
        assert res["timed_out"] is False
        assert res["stdout"] == "res_ok\n"
        assert res["infra_retries"] == 1
        assert mock_attempt.call_count == 2


def test_infra_path_a_host_backstop_retry_also_hits_backstop_scores_timeout():
    """(a) Host backstop TimeoutExpired: retry also hits backstop -> score TIMEOUT."""
    with patch("code_exec._run_docker_single_attempt") as mock_attempt:
        mock_attempt.side_effect = [
            _HostBackstopTimeout("Host timeout attempt 1"),
            _HostBackstopTimeout("Host timeout attempt 2"),
        ]
        res = _run_docker("print('ok')", "", timeout=6.0, backoff_s=0.0)
        assert res["timed_out"] is True
        assert res["infra_retries"] == 1
        assert mock_attempt.call_count == 2


def test_infra_path_a_host_backstop_retry_returns_124_scores_timeout():
    """(a) Host backstop TimeoutExpired: retry returns 124 -> score TIMEOUT."""
    with patch("code_exec._run_docker_single_attempt") as mock_attempt:
        mock_attempt.side_effect = [
            _HostBackstopTimeout("Host timeout attempt 1"),
            {"stdout": "", "stderr": "inner timeout", "returncode": 124, "timed_out": True},
        ]
        res = _run_docker("print('ok')", "", timeout=6.0, backoff_s=0.0)
        assert res["timed_out"] is True
        assert res["returncode"] == 124
        assert res["infra_retries"] == 1
        assert mock_attempt.call_count == 2


def test_infra_path_b_cli_failures_and_daemon_error_strings():
    """(b) Test _is_infra_error on return codes 125, 126, 127 and error strings."""
    assert _is_infra_error(125, "") is True
    assert _is_infra_error(126, "") is True
    assert _is_infra_error(127, "") is True
    assert _is_infra_error(1, "Cannot connect to the Docker daemon at unix:///var/run/docker.sock") is True
    assert _is_infra_error(1, "error during connect: Get http://docker_engine/containers") is True
    assert _is_infra_error(1, "docker: Error response from daemon: Cannot start container") is True
    assert _is_infra_error(1, "docker: error response from daemon") is True
    # Normal crash should NOT be treated as infra error
    assert _is_infra_error(1, "Traceback (most recent call last):\nZeroDivisionError: division by zero") is False
    assert _is_infra_error(0, "") is False


def test_infra_path_b_retry_twice_with_backoff_then_raise_sandbox_unavailable():
    """(b)/(c) Docker CLI failures retry up to 2 times with backoff, then raise SandboxUnavailable."""
    with patch("code_exec._run_docker_single_attempt") as mock_attempt, \
         patch("time.sleep") as mock_sleep:
        mock_attempt.side_effect = _DockerInfraFailure("CLI exit 125: daemon error")
        with pytest.raises(SandboxUnavailable) as exc_info:
            _run_docker("print('x')", "", timeout=6.0, backoff_s=10.0)
        assert "Docker sandbox unavailable after 2 retries" in str(exc_info.value)
        # Attempt 0, Retry 1, Retry 2 = 3 total attempts
        assert mock_attempt.call_count == 3
        # 2 backoffs of 10s
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(10.0)


def test_infra_path_b_retry_succeeds_on_second_attempt():
    """(b) Docker CLI failure on first attempt succeeds on retry 1."""
    with patch("code_exec._run_docker_single_attempt") as mock_attempt, \
         patch("time.sleep") as mock_sleep:
        mock_attempt.side_effect = [
            _DockerInfraFailure("Temporary daemon glitch"),
            {"stdout": "recovered\n", "stderr": "", "returncode": 0, "timed_out": False},
        ]
        res = _run_docker("print('x')", "", timeout=6.0, backoff_s=10.0)
        assert res["stdout"] == "recovered\n"
        assert res["infra_retries"] == 1
        assert mock_attempt.call_count == 2
        assert mock_sleep.call_count == 1


def test_infra_failure_never_scored_as_crash_fail_or_timeout():
    """Infra failure raises SandboxUnavailable and is NEVER scored as CRASH, FAIL, or TIMEOUT."""
    task = Task(
        task_id="infra_test_task",
        family="code_lcb",
        prompt="Solve problem",
        ground_truth="PASS",
        meta={"hidden_tests": [{"input": "1", "expected_output": "1"}]},
    )
    with patch("tasks.livecodebench_tasks.run_tests", side_effect=SandboxUnavailable("Daemon down")):
        with pytest.raises(SandboxUnavailable):
            grade_lcb(task, "```python\nprint(1)\n```")


def test_real_docker_bin_nonexistent_raises_sandbox_unavailable_not_crash():
    """REAL TEST: set DOCKER_BIN to a nonexistent path and confirm SandboxUnavailable, not CRASH."""
    import code_exec
    old_bin = code_exec.DOCKER_BIN
    code_exec.DOCKER_BIN = "nonexistent_docker_binary_path_round10_xyz_404"
    try:
        task = Task(
            task_id="nonexistent_docker_task",
            family="code_lcb",
            prompt="Solve problem",
            ground_truth="PASS",
            meta={"hidden_tests": [{"input": "1", "expected_output": "1"}]},
        )
        with pytest.raises(SandboxUnavailable):
            grade_lcb(task, "```python\nprint(1)\n```")
    finally:
        code_exec.DOCKER_BIN = old_bin


def test_is_git_clean_fail_closed():
    """is_git_clean: fail closed (any exception or git error returns False)."""
    # 1. Normal clean
    with patch("subprocess.run") as mock_sub:
        mock_sub.return_value = MagicMock(stdout="", returncode=0)
        assert is_git_clean() is True

    # 2. Untracked files only -> still clean
    with patch("subprocess.run") as mock_sub:
        mock_sub.return_value = MagicMock(stdout="?? scratch.py\n", returncode=0)
        assert is_git_clean() is True

    # 3. Modified tracked file -> dirty (False)
    with patch("subprocess.run") as mock_sub:
        mock_sub.return_value = MagicMock(stdout=" M src/code_exec.py\n", returncode=0)
        assert is_git_clean() is False

    # 4. Exception in subprocess -> fail-closed (False)
    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, ["git", "status"])):
        assert is_git_clean() is False

    with patch("subprocess.run", side_effect=FileNotFoundError("git binary not found")):
        assert is_git_clean() is False

    with patch("subprocess.run", side_effect=PermissionError("Permission denied")):
        assert is_git_clean() is False


def test_append_run_manifest_appends_and_never_overwrites():
    """append_run_manifest appends records to manifest and never overwrites existing entries."""
    with tempfile.TemporaryDirectory() as td:
        manifest_file = os.path.join(td, "run_manifest.json")
        rec1 = append_run_manifest(manifest_path=manifest_file)
        assert os.path.exists(manifest_file)

        with open(manifest_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 1
        assert "utc_time" in data[0]
        assert "git_commit" in data[0]
        assert "git_describe" in data[0]
        assert "git_status_porcelain" in data[0]
        assert "prompts_hash" in data[0]
        assert "problem_ids_sha256" in data[0]
        assert "docker_image_digest" in data[0]

        # Call again -> appends second entry, never overwrites
        rec2 = append_run_manifest(manifest_path=manifest_file)
        with open(manifest_file, "r", encoding="utf-8") as f:
            data2 = json.load(f)
        assert len(data2) == 2
        assert data2[0]["utc_time"] == rec1["utc_time"]
        assert data2[1]["utc_time"] == rec2["utc_time"]


def run():
    test_normalize_output_basic()
    test_normalize_output_preserves_internal_blanks()
    test_normalize_output_adds_trailing_newline()
    test_code_hash_deterministic()
    test_code_hash_normalizes_whitespace()
    test_code_hash_length()
    test_ast_hash_comment_only()
    test_ast_hash_docstring_only()
    test_ast_hash_rename_changes()
    test_ast_hash_real_logic_changes()
    test_ast_hash_syntax_error_fallback()
    test_infra_path_a_host_backstop_retry_succeeds()
    test_infra_path_a_host_backstop_retry_also_hits_backstop_scores_timeout()
    test_infra_path_a_host_backstop_retry_returns_124_scores_timeout()
    test_infra_path_b_cli_failures_and_daemon_error_strings()
    test_infra_path_b_retry_twice_with_backoff_then_raise_sandbox_unavailable()
    test_infra_path_b_retry_succeeds_on_second_attempt()
    test_infra_failure_never_scored_as_crash_fail_or_timeout()
    test_real_docker_bin_nonexistent_raises_sandbox_unavailable_not_crash()
    test_is_git_clean_fail_closed()
    test_append_run_manifest_appends_and_never_overwrites()
    print("[ok] all code_exec tests passed")


if __name__ == "__main__":
    run()

