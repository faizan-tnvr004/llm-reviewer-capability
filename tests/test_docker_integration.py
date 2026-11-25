"""Docker integration tests running against a real Docker daemon.

Per pre-registration requirements:
- Marked @pytest.mark.docker
- Fails (does NOT skip) if Docker daemon is absent
- Verifies:
  1. Correct program execution
  2. Wrong output detection
  3. Crash / unhandled exception handling
  4. Timeout on an infinite loop
  5. Outbound network attempt fails (--network none)
  6. No host env vars (GOOGLE_API_KEY, GROQ_API_KEY) visible inside container
  7. Writing outside /tmp fails (read-only filesystem)
  8. 1 GB memory allocation is killed by 512 MB memory limit
  9. Fork bomb bounded by --pids-limit
  10. Trailing-space output still passes via normalization
  11. After timeout, docker ps -a shows no leftover container
  12. Startup sweep removes stale containers
  13. 8s sleep times out in under ~7.5s wall time
  14. 3s program passes within 6s limit
  15. Problem budget exhaustion stops before test 5
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from code_exec import (
    is_docker_available,
    run_program,
    run_tests,
    DOCKER_BIN,
    EXEC_TIMEOUT,
)


@pytest.fixture(autouse=True)
def require_live_docker():
    """Ensure Docker daemon is available and fail (not skip) if absent."""
    if not is_docker_available():
        pytest.fail("Docker daemon is absent or not installed; integration tests require a real daemon.")
    try:
        proc = subprocess.run([DOCKER_BIN, "version"], capture_output=True, text=True, timeout=5.0)
        if proc.returncode != 0:
            pytest.fail(f"Docker daemon returned non-zero code: {proc.stderr}")
    except Exception as e:
        pytest.fail(f"Docker daemon check failed: {e}")


@pytest.mark.docker
def test_docker_correct_program():
    """1. Correct program passes against test cases."""
    code = "a, b = map(int, input().split())\nprint(a + b)\n"
    tc = [{"input": "5 7\n", "expected_output": "12\n"}]
    results = run_tests(code, tc)
    assert len(results) == 1
    assert results[0]["passed"] is True
    assert results[0]["kind"] == "pass"


@pytest.mark.docker
def test_docker_wrong_output():
    """2. Program with wrong output is flagged as failed."""
    code = "print(999)\n"
    tc = [{"input": "5 7\n", "expected_output": "12\n"}]
    results = run_tests(code, tc)
    assert len(results) == 1
    assert results[0]["passed"] is False
    assert results[0]["kind"] == "wrong"


@pytest.mark.docker
def test_docker_crash():
    """3. Crashing program (e.g. ZeroDivisionError) reports crash and stderr."""
    code = "x = 1 / 0\n"
    tc = [{"input": "", "expected_output": "0\n"}]
    results = run_tests(code, tc)
    assert len(results) == 1
    assert results[0]["passed"] is False
    assert results[0]["kind"] == "crash"
    assert "ZeroDivisionError" in results[0]["stderr_tail"]


@pytest.mark.docker
def test_docker_timeout_infinite_loop():
    """4. Infinite loop times out cleanly."""
    code = "while True:\n    pass\n"
    tc = [{"input": "", "expected_output": "done\n"}]
    results = run_tests(code, tc, per_test_timeout_s=1.0)
    assert len(results) == 1
    assert results[0]["passed"] is False
    assert results[0]["kind"] == "timeout"


@pytest.mark.docker
def test_docker_outbound_network_attempt_fails():
    """5. Outbound network attempts fail under --network none."""
    code = (
        "import urllib.request\n"
        "try:\n"
        "    urllib.request.urlopen('http://8.8.8.8', timeout=2)\n"
        "    print('NETWORK_SUCCESS')\n"
        "except Exception as e:\n"
        "    print('NETWORK_FAILED')\n"
    )
    res = run_program(code, "", timeout=5.0)
    assert res["returncode"] == 0
    assert "NETWORK_FAILED" in res["stdout"]
    assert "NETWORK_SUCCESS" not in res["stdout"]


@pytest.mark.docker
def test_docker_no_host_env_vars_visible():
    """6. Host sensitive environment variables are not visible inside container."""
    os.environ["GOOGLE_API_KEY"] = "super_secret_google_key_123"
    os.environ["GROQ_API_KEY"] = "super_secret_groq_key_456"

    code = (
        "import os\n"
        "g = os.environ.get('GOOGLE_API_KEY')\n"
        "q = os.environ.get('GROQ_API_KEY')\n"
        "if g is None and q is None:\n"
        "    print('CLEAN_ENV')\n"
        "else:\n"
        "    print(f'LEAK:{g}:{q}')\n"
    )
    res = run_program(code, "", timeout=5.0)
    assert res["returncode"] == 0
    assert "CLEAN_ENV" in res["stdout"]
    assert "LEAK" not in res["stdout"]


@pytest.mark.docker
def test_docker_readonly_fs_enforced():
    """7. Writing outside /tmp fails on read-only root filesystem."""
    code = (
        "# Writing to /tmp succeeds\n"
        "with open('/tmp/test.txt', 'w') as f:\n"
        "    f.write('ok')\n"
        "# Writing outside /tmp (e.g. current directory /work) must fail\n"
        "try:\n"
        "    with open('illegal.txt', 'w') as f:\n"
        "        f.write('fail')\n"
        "    print('WROTE_OUTSIDE')\n"
        "except OSError:\n"
        "    print('READ_ONLY_ENFORCED')\n"
    )
    res = run_program(code, "", timeout=5.0)
    assert res["returncode"] == 0
    assert "READ_ONLY_ENFORCED" in res["stdout"]
    assert "WROTE_OUTSIDE" not in res["stdout"]


@pytest.mark.docker
def test_docker_memory_limit_1gb_killed():
    """8. A 1 GB memory allocation is killed by the 512 MB memory limit."""
    code = (
        "b = bytearray(1024 * 1024 * 1024)\n"
        "print('ALLOCATED_1GB')\n"
    )
    res = run_program(code, "", timeout=5.0)
    # Container is either OOM killed (returncode 137 / non-zero) or raises MemoryError
    assert res["returncode"] != 0 or "MemoryError" in res["stderr"]
    assert "ALLOCATED_1GB" not in res["stdout"]


@pytest.mark.docker
def test_docker_fork_bomb_bounded():
    """9. Fork bomb is bounded by --pids-limit and cannot crash or hang the host."""
    code = (
        "import os\n"
        "try:\n"
        "    for _ in range(200):\n"
        "        os.fork()\n"
        "    print('FORK_UNBOUNDED')\n"
        "except OSError:\n"
        "    print('FORK_BOUNDED')\n"
    )
    res = run_program(code, "", timeout=5.0)
    # Either caught OSError (can't fork) or container terminated by pids-limit
    assert "FORK_UNBOUNDED" not in res["stdout"]


@pytest.mark.docker
def test_docker_trailing_space_passes():
    """10. Trailing spaces and excess blank lines pass through normalize_output."""
    code = "print('output text   ')\nprint('')\nprint('   ')\n"
    tc = [{"input": "", "expected_output": "output text\n"}]
    results = run_tests(code, tc)
    assert len(results) == 1
    assert results[0]["passed"] is True


@pytest.mark.docker
def test_docker_timeout_no_leftover_container():
    """11. After a timeout on an infinite loop, docker ps -a shows no leftover container."""
    code = "while True:\n    pass\n"
    tc = [{"input": "", "expected_output": "done\n"}]

    # Run the timed-out program
    results = run_tests(code, tc, per_test_timeout_s=1.0)
    assert len(results) == 1
    assert results[0]["kind"] == "timeout"

    # Inspect all containers created with our prefix 'code_exec_'
    proc = subprocess.run(
        [DOCKER_BIN, "ps", "-a", "--filter", "name=code_exec_", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    active_or_exited = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    assert active_or_exited == [], f"Found leftover container(s) after timeout: {active_or_exited}"


@pytest.mark.docker
def test_docker_cleanup_stale_containers_sweep():
    """12. Startup sweep removes any stale containers named code_exec_* and logs count."""
    from code_exec import cleanup_stale_containers, DOCKER_IMAGE

    dummy_name = f"code_exec_stale_sweep_dummy_{int(time.time())}"
    # Create an exited container
    subprocess.run([DOCKER_BIN, "create", "--name", dummy_name, DOCKER_IMAGE, "true"],
                   capture_output=True, check=True)

    try:
        # Check that it exists
        proc = subprocess.run([DOCKER_BIN, "ps", "-a", "--filter", f"name={dummy_name}", "--format", "{{.Names}}"],
                              capture_output=True, text=True, check=True)
        assert dummy_name in proc.stdout

        # Run sweep
        swept = cleanup_stale_containers()
        assert swept >= 1

        # Check that it is now gone
        proc2 = subprocess.run([DOCKER_BIN, "ps", "-a", "--filter", f"name={dummy_name}", "--format", "{{.Names}}"],
                               capture_output=True, text=True, check=True)
        assert dummy_name not in proc2.stdout
    finally:
        subprocess.run([DOCKER_BIN, "rm", "-f", dummy_name], capture_output=True)


@pytest.mark.docker
def test_docker_timeout_enforced_under_7_5s():
    """13. A program that sleeps 8 s must return TIMEOUT in under about 7.5 s wall time."""
    code = "import time\ntime.sleep(8)\nprint('done')\n"
    tc = [{"input": "", "expected_output": "done\n"}]
    t0 = time.time()
    results = run_tests(code, tc)  # default EXEC_TIMEOUT = 6.0 s
    wall = time.time() - t0
    assert len(results) == 1
    assert results[0]["passed"] is False
    assert results[0]["kind"] == "timeout"
    assert wall < 7.5, f"Expected timeout wall time under 7.5s, got {wall:.2f}s"
    assert wall >= 5.8, f"Expected wall time >= ~6.0s, got {wall:.2f}s"


@pytest.mark.docker
def test_docker_program_passes_within_timeout():
    """14. A program that takes 3 s must PASS under the 6 s limit."""
    code = "import time\ntime.sleep(3)\nprint('hello_ok')\n"
    tc = [{"input": "", "expected_output": "hello_ok\n"}]
    t0 = time.time()
    results = run_tests(code, tc)
    wall = time.time() - t0
    assert len(results) == 1
    assert results[0]["passed"] is True
    assert results[0]["kind"] == "pass"
    assert results[0]["actual"] == "hello_ok\n"
    assert wall >= 3.0, f"Expected wall time >= 3.0s, got {wall:.2f}s"


@pytest.mark.docker
def test_docker_problem_budget_exhaustion_stops_early():
    """15. Budget test: tiny budget (3 s) with 5 tests of 1.2 s each must stop with TIMEOUT before test 5."""
    code = "import time\ntime.sleep(1.2)\nprint('step_ok')\n"
    tc = [{"input": "", "expected_output": "step_ok\n"} for _ in range(5)]
    results = run_tests(code, tc, max_budget_s=3.0)
    assert len(results) < 5, f"Expected fewer than 5 tests run before budget exhaustion, got {len(results)}"
    assert results[-1]["passed"] is False
    assert results[-1]["kind"] == "timeout"

