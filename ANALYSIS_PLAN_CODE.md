# Pre-Registered Analysis Plan: Code Extension Study

**Title**: Reviewer Capability in Execute-Review-Revise Pipelines: Executable Code Extension  
**Repository**: `faizan-tnvr004/llm-reviewer-capability`  
**Branch**: `code-extension`  
**Date**: Drafted: 24 October 2026; committed: see git tag prereg-code-v1  
**Hard Stop**: 7 November 2026  
**Prompts SHA-256 Hash**: `d7b7203e7603dc5070bc42efdedc140de146caa88975f7ef1036d3271b56ca73`  
**Supersedes**: commit `9003fd0` ("Pre-registration: code extension", 24 October 2026). That commit froze a 100-problem draft and an earlier version of this plan. Both were revised before any main-run call was made or any main-run output existed (see Section 2.4). This document and the commit tagged `prereg-code-v1` are the binding pre-registration; `9003fd0` is kept in history for transparency.

---

## 1. Goal and Research Questions

This study replicates and extends the Olympiad Math pipeline-error study on executable Python programming tasks drawn from LiveCodeBench-Lite. We evaluate whether the central findings of self-review failure (high false rejections, executor inertia, destructive revisions) and cross-family reviewer benefit generalize to programming with ground-truth test execution, and introduce Condition C4 (deterministic public sample test feedback).

- **RQ1**: Does self-review falsely reject correct programs at a high rate? Does a cross-family reviewer raise accuracy without damage?
- **RQ2**: Does the executor leave its code unchanged after a false rejection, and do the changes it does make break correct code?
- **RQ3**: How does public-test feedback (C4) compare with LLM verdicts on detection, false rejection, and repair? How often do public sample tests accept a program that fails hidden tests (false accept)?

---

## 2. Benchmark & Problem Selection

### 2.1 Dataset Source & Version
- **Dataset**: `livecodebench/code_generation_lite` (HuggingFace)
- **HuggingFace Dataset Revision Hash**: `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`
- **Pinned Release Files & SHA-256 Digests**:
  - `test.jsonl` (400 problems, SHA-256: `2bd02b38beb48e8c46b5b9987095d999ff38cd8efc255ea5d58974317c48f63f`)
  - `test2.jsonl` (111 problems, SHA-256: `095df7c5daf15f882c51a9deb84085cff1e073495a5dbcf95015a564d485f3a3`)
  - `test3.jsonl` (101 problems, SHA-256: `28ed26cc83363ce3f1fe2d5fad9f8393077beb1907b167a31bd3b32f80801b79`)
  - `test4.jsonl` (101 problems, SHA-256: `d711138ddaebfcf5f8ec6a4283ee677298c0f5c5d374a235af92aaf0584510da`)
  - `test5.jsonl` (167 problems, SHA-256: `7f77571c2a6df0c2a72a3277650309f67e01e0008e18117e624633df53f81214`)
  - `test6.jsonl` (175 problems, SHA-256: `bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5`)
  - **Total**: 1,055 problems spanning contest dates `2023-05-07` to `2025-04-06`. No files or contest problems after April 2025 exist in the HuggingFace repository.

### 2.2 Filtering Criteria
1. **Stdin-only IO**: AtCoder and Codeforces problems where `testtype == "stdin"`. Functional / LeetCode problems (e.g. `class Solution:`) are dropped.
2. **Difficulty**: `medium` and `hard` only.
3. **Contest Date Window & Knowledge Cutoff**: Contests from `2024-10-01` to `2025-04-06` represent the most recent available contests in LiveCodeBench-Lite. Executor knowledge cutoff: January 2025, per Google's Gemini 3 developer guide (https://ai.google.dev/gemini-api/docs/gemini-3, model table row `gemini-3.1-flash-lite`, "Knowledge Cutoff: Jan 2025", retrieved 27 October 2026). Earlier drafts' March 2026 and March 2025 figures were never sourced and are withdrawn. A stated cutoff does not exclude contamination from later training updates, so contamination cannot be ruled out. The dataset is split at boundary `2025-01-31` for temporal sensitivity analysis (see Section 7, E2).
4. **No Special-Judge Problems**: Statements containing non-deterministic output phrases (`"any of them"`, `"print any"`, `"if there are multiple"`, `"absolute or relative error"`, `"relative error"`, `"absolute error"`) or `is_special_judge` flags are dropped.
5. **Minimum Hidden Tests ($n_{\text{hidden\_tests}} \ge 10$)**: Problems with fewer than 10 hidden tests are dropped to eliminate accidental false-acceptance noise and ensure robust grading. Exactly 10 candidate problems in the window were excluded under this rule (all have $\le 4$ hidden tests):
   - `abc382_g` (1 hidden test, hard)
   - `abc399_d` (1 hidden test, medium)
   - `abc400_e` (4 hidden tests, hard)
   - `abc400_g` (4 hidden tests, hard)
   - `arc185_b` (1 hidden test, hard)
   - `arc186_c` (1 hidden test, hard)
   - `arc188_b` (1 hidden test, hard)
   - `arc193_d` (1 hidden test, hard)
   - `arc195_d` (1 hidden test, hard)
   - `arc196_b` (1 hidden test, hard)

### 2.3 Calibration Batch (Phase B)
- **Batch 1**: Evaluated 20 candidate problems on `gemini-3.1-flash-lite` (executor only) in a local Windows subprocess:
  - **Overall Accuracy**: 8 solved out of 20 = **40.0%** (target 40% – 60% met).
  - **Difficulty Breakdown in Calibration**:
    - Medium: 7 evaluated, 5 correct = **71.4%**
    - Hard: 13 evaluated, 3 correct = **23.1%**
  - **Fingerprint Counts**: `PASS`: 8 (40%), `FAIL`: 10 (50%), `TIMEOUT`: 2 (10%), `CRASH`: 0 (0%).
  - **Reconciled Calibration Table (Phase B Subprocess vs Docker Sandbox)**:
    Under strict early exit (`fail_fast=True`) in Docker, execution halts on the first failing or timed-out hidden test ($k$ out of $n$ total hidden tests). In Phase B (old semantics), the fingerprint `FAIL:f/n` recorded the total count of failed tests $f$ out of total tests $n$ across a full test run; under Docker with early exit, `FAIL:k/n` records the 1-indexed position $k$ of the first failing test out of $n$ total hidden tests. Accuracy is identical at 8/20 (40.0%), with 100% agreement on all 8 PASS programs (0 status differences):

    | ID | Diff | Phase B FP (failed/total, old semantics) | Phase B Res | Docker FP (first-failing-test/total) | Tests Run | Wall Time (one run; observed to vary up to ~2x between runs, with occasional start-up stalls of several seconds) |
    |:---|:---|:---|:---|:---|:---|:---:|
    | `arc186_e` | hard | `TIMEOUT` | FAIL | `FAIL:1/40` | 1/40 | 0.38s |
    | `arc189_a` | medium | `FAIL:35/40` | FAIL | `FAIL:1/40` | 1/40 | 0.41s |
    | `abc390_c` | medium | `PASS` | PASS | `PASS` | 40/40 | 15.70s |
    | `arc196_d` | hard | `FAIL:32/40` | FAIL | `FAIL:1/40` | 1/40 | 0.97s |
    | `abc383_e` | hard | `FAIL:13/40` | FAIL | `FAIL:5/40` | 5/40 | 1.88s |
    | `arc195_a` | medium | `PASS` | PASS | `PASS` | 40/40 | 15.44s |
    | `abc382_d` | hard | `PASS` | PASS | `PASS` | 40/40 | 16.43s |
    | `abc376_b` | medium | `PASS` | PASS | `PASS` | 21/21 | 8.04s |
    | `abc377_f` | hard | `FAIL:38/40` | FAIL | `TIMEOUT` | 1/40 | 6.38s |
    | `abc376_d` | medium | `PASS` | PASS | `PASS` | 36/36 | 20.79s |
    | `arc194_b` | hard | `FAIL:25/40` | FAIL | `FAIL:4/40` | 4/40 | 1.53s |
    | `abc394_c` | medium | `FAIL:1/40` | FAIL | `FAIL:38/40` | 38/40 | 14.74s |
    | `abc394_g` | hard | `PASS` | PASS | `PASS` | 40/40 | 15.66s |
    | `arc189_b` | hard | `FAIL:38/40` | FAIL | `FAIL:1/40` | 1/40 | 0.57s |
    | `arc195_c` | hard | `FAIL:4/4` | FAIL | `FAIL:1/4` | 1/4 | 0.48s |
    | `abc385_c` | medium | `PASS` | PASS | `PASS` | 40/40 | 16.82s |
    | `abc381_e` | hard | `FAIL:12/40` | FAIL | `FAIL:1/40` | 1/40 | 0.43s |
    | `abc378_g` | hard | `TIMEOUT` | FAIL | `FAIL:1/40` | 1/40 | 0.40s |
    | `abc386_e` | hard | `PASS` | PASS | `PASS` | 40/40 | 15.75s |
    | `abc392_g` | hard | `FAIL:23/40` | FAIL | `TIMEOUT` | 1/40 | 6.41s |

  - **Disclosures on Timeout Fingerprints**:
    The Docker evaluation yields exactly 2 `TIMEOUT` outcomes (`abc377_f` and `abc392_g`), which coincides numerically with Phase B's count of 2 TIMEOUTs, but refers to completely different problems (`arc186_e` and `abc378_g` in Phase B). In Phase B, the TIMEOUT outcomes for `arc186_e` and `abc378_g` came from the 25.0 s outer per-problem limit in the earlier unconstrained Windows subprocess run, where tests continued executing until the total elapsed problem time exceeded 25 s. Under strict per-test early exit in Docker enforced by container-internal `timeout 6.0`, `abc377_f` and `abc392_g` score `TIMEOUT` because their Test 1 exceeds 6.0s (locally 6.05s and 6.02s; in Docker 6.38s and 6.41s under container-internal timeout, previously ~11.2s before inner enforcement), whereas in Phase B without per-test early exit they ran to completion as `FAIL:38/40` and `FAIL:23/40`. Conversely, `arc186_e` and `abc378_g` score `FAIL:1/40` in Docker because their Test 1 fails immediately on wrong output (about 0.05s locally: 0.066s and 0.049s; and 0.4–0.7s in Docker: 0.38s and 0.40s).
  - **Grading Time Projections**: Based on measured calibration wall times in Docker, total grading time for ~300 programs (~120 correct, ~180 wrong) is projected at about 55-65 minutes of grading (means 21.8-25.0 s per correct and 3.9-4.1 s per wrong program in two later passes). Earlier 68-minute projections based on 33s / 0.75s splits came from discredited synthetic data and are superseded.
  - **Calibration Exclusions**: The 20 calibration problem IDs (`arc186_e`, `arc189_a`, `abc390_c`, `arc196_d`, `abc383_e`, `arc195_a`, `abc382_d`, `abc376_b`, `abc377_f`, `abc376_d`, `arc194_b`, `abc394_c`, `abc394_g`, `arc189_b`, `arc195_c`, `abc385_c`, `abc381_e`, `abc378_g`, `abc386_e`, `abc392_g`) are permanently excluded from the main study.

### 2.4 Pre-Registered Problem Set ($N = 125$, Full Eligible Population)
- **Eligible Problem Set**: $N = 125$. All 125 candidate problems meeting the filtering criteria (after excluding the 20 calibration problems and the 10 problems with $< 10$ hidden tests) are included. There is no sub-sampling; $N = 125$ comprises the entire eligible population.
- **Fixed Processing Order**: To eliminate ordering bias and ensure reproducible execution pacing, the 125 problems are sorted deterministically by `(contest_date, question_id)` and then shuffled with seed `42` into a fixed execution order.
- **Superseded 100-Problem Draft Hash**: The earlier 100-problem draft (`run_code_data/problem_ids.json` as committed in `9003fd0`) was discarded prior to any main-run execution or outcome generation. Its SHA-256 is `9a54e545a27dfc8f191bed1300adf296773f06738c58dd6fff582a834c7c22b0` for the committed (LF) blob and `69a24e9a7cca092d4a48381d68866e0d6c8e214bf889f0fb6b02ac4489a94a7a` for the CRLF Windows working copy; the files differ only in line endings.
- **Frozen Problem IDs File**: `run_code_data/problem_ids.json` (SHA-256: `246a18b49edc49c6bccefa5eeec7de76db4d214e64b6c2a0a7af58dbf33d6a7f`). `.gitattributes` marks this file `-text`, so the committed bytes and the working-copy bytes are identical and this hash holds on every checkout.
- **Difficulty Composition**: 40 medium (32.0%), 85 hard (68.0%) (100% stdin testtype).
- **Hidden Test Counts for the 125 Problems**:
  - Min: 27 tests
  - Median: 40 tests
  - Mean: 39.30 tests
  - Max: 48 tests
- **Stratified Expected Accuracy**:
  $$\text{Expected } C_0 \text{ Accuracy} = \frac{40 \times (5/7) + 85 \times (3/13)}{125} = \frac{28.57 + 19.62}{125} = \frac{48.19}{125} = 38.55\% \approx 38.5\%$$
  - **Expected Correct Programs**: $\approx 48.2$ programs out of 125.
  - **Target Band Comparison & Calibration Strata**: The expected point estimate of 38.5% is slightly below the pre-registered 40%–60% target band. The 95% Wilson score confidence intervals for the calibration strata are:
    - **Medium Stratum** (5/7): 71.4% (95% Wilson CI: [35.9%, 91.8%])
    - **Hard Stratum** (3/13): 23.1% (95% Wilson CI: [8.2%, 50.3%])
    - **Overall Calibration** (8/20): 40.0% (95% Wilson CI: [21.9%, 61.3%])
    Treating strata separately across the 40 medium and 85 hard problems yields a plausible baseline range of **[17.0%, 63.5%]** (21.3 to 79.4 correct programs).
- **Expected C1 False Rejections & H2 Testability**:
  - Evaluated on the expected $\approx 48.2$ correct programs:
    - At a 20% false-rejection rate: expected false rejections = $48.19 \times 0.20 = \mathbf{9.6}$ (floor 9, round 10).
    - At a 35% false-rejection rate (observed in the Olympiad Math study): expected false rejections = $48.19 \times 0.35 = \mathbf{16.9}$ (round 17).
  - H2 requires $n_{\text{false\_rej}} \ge 10$ to avoid being declared inconclusive under the low-$n$ rule. A false-rejection rate of at least **20.8%** is required to yield $\ge 10$ false rejections. Therefore, H2 is on the testability borderline at the H1 threshold (20%), and comfortably testable ($16.9 \ge 10$) at the empirical rate observed in the math study (35%).

---

## 3. Experimental Conditions & Model Lineup

- **Executor**: `gemini-3.1-flash-lite` (Google AI Studio, Temperature = 0.7, max output tokens = 8,192, candidate count = 1). Solves each problem once to produce program $A_0$, which is shared across all conditions. On `REJECT`, the executor revises once.

| Condition | Role | Reviewer Model / Mechanism | Provider / Platform | Parameters |
|---|---|---|---|---|
| **C0** | Baseline (No Review) | None ($A_0$ is final) | N/A | N/A |
| **C1** | Self-Review | `gemini-3.1-flash-lite` | Google AI Studio | Temp = 0.7, max tokens = 8,192 |
| **C2** | Cross-Family Mid-Tier | `openai/gpt-oss-20b` | Groq | Temp = 0.7, max tokens = 6,144 |
| **C3** | Weak Reviewer | `llama3.1:8b-instruct-q8_0` (Llama 3.1 8B Instruct, Q8_0) | Local Ollama 0.40.16 on the run laptop | Temp = 0.7, max tokens = 2,048, context = 8,192 |
| **C4** | Public Sample Tests | Deterministic test runner | Docker sandbox | ToolReviewer (0 tokens) |

- **C3 Host Change (decided 25 October 2026, host finalised 18 November 2026, before any main-run call)**: The math study's weak reviewer was `llama-3.1-8b-instant` on Groq. Groq shut that model down on 16 August 2026 (free and developer tiers), and it returned HTTP 404 `model_not_found` in the 25 October smoke test. To keep the weak-reviewer arm comparable with the math study, C3 uses the same Llama 3.1 8B Instruct weights. A hosted copy on DeepInfra was tried first but needs a paid balance (HTTP 402 in the 28 October smoke test), so C3 runs locally in Ollama 0.40.16 (installer SHA-256 `84db4f756ddc9e091142a85cdc0f7ae5db68bc9b3e075f114388b326f8ed912d`) as model `llama3.1:8b-instruct-q8_0`, pinned by Ollama digest `b158ded76fa05be6bce8a682099ce5df8c5571340a04cf63a2923464679db576` (8-bit Q8_0 quantization, 8.0B parameters). It runs on an NVIDIA GeForce RTX 4050 Laptop GPU (6 GB) with partial CPU offload (about 57% CPU / 43% GPU), at about 3 output tokens per second. Requests set `num_ctx = 8192` because Ollama's default context is smaller than a review prompt plus 2,048 output tokens, and a request whose prompt fills the context is refused rather than reviewed. Groq did not publish its serving precision, so numeric differences from the math study's C3 can come from the serving setup (quantization, sampler implementation) as well as from the task. Prompt, temperature and `max_tokens` are unchanged. The math study's own model definitions (`experimental_models()` in `src/providers.py`) are left as they were; the code study uses `code_experimental_models()`.
- **Provider Preflight**: Before writing the manifest or taking the lock, the runner checks at zero cost that both SDKs import, that `GOOGLE_API_KEY` and `GROQ_API_KEY` are set, that Groq lists `openai/gpt-oss-20b`, and that the local Ollama server answers and holds `llama3.1:8b-instruct-q8_0` with exactly the pinned digest. If any check fails, it refuses to start.

---

## 4. Execution Sandbox, Timeouts, and Run Mechanics

### 4.1 Strict Docker Sandboxing (No UNSAFE_LOCAL_EXEC)
- All model-generated code runs exclusively inside disposable Docker containers:
  `docker run --rm --network none --memory 512m --cpus 1 --pids-limit 64 --read-only --tmpfs /tmp:rw,size=64m python@sha256:b7a9e5f6a93d6d4a31f129bbd4bdee69efccb6ded0fb230d78aafaf37af98377`
- **Docker Image RepoDigest**: Containers run strictly by immutable RepoDigest:
  `python@sha256:b7a9e5f6a93d6d4a31f129bbd4bdee69efccb6ded0fb230d78aafaf37af98377`
  (Image ID: `sha256:b7a9e5f6a93d6d4a31f129bbd4bdee69efccb6ded0fb230d78aafaf37af98377`, tagged `python:3.11-slim`).
- **Zero API Keys in Sandbox**: The container environment contains no host secrets (`GOOGLE_API_KEY` and `GROQ_API_KEY` are strictly stripped from the container environment).
- **Hard Stop Policy**: `UNSAFE_LOCAL_EXEC` is removed. If Docker is unavailable or cannot run, execution immediately stops and reports an error; it will **not** fall back to local subprocess execution.

### 4.2 Timeout Budget and Early Exit
- **Execution History Context**: Phase B calibration and the earlier TIMEOUT re-check ran in a local Windows subprocess before strict Docker enforcement. Calibration programs are re-graded inside Docker under identical resource limits.
- **Per-Test Timeout**: 6.0 seconds per individual test case, enforced inside the container via `timeout 6.0 python3 solution.py < input.txt` (exit code 124 treated as timeout). A host-side subprocess timeout of 11.0 s (`timeout + 5.0 s`) serves as a backstop against container engine hangs.
- **Per-Problem Budget**: $\min(n_{\text{hidden\_tests}} \times 6.0\text{ s}, 180.0\text{ s})$.
- **Early Exit**: Test execution terminates immediately on the **first failing or timed-out test**.
- **Fingerprint Definitions under Early Exit**:
  - `PASS`: All hidden tests passed.
  - `FAIL:k/n`: Test execution terminated on hidden test $k$ (1-indexed) out of $n$ tests due to incorrect output.
  - `TIMEOUT`: The first failure encountered was a timeout (test execution timed out under the 6.0 s limit or exhausted the per-problem budget).
  - `CRASH`: The first failure encountered was an uncaught runtime exception / nonzero exit.
- **Problem Budget Exhaustion Rule**: A program that passes every test it runs, but whose cumulative execution time exhausts the per-problem budget ($\min(n_{\text{hidden\_tests}} \times 6.0\text{ s}, 180.0\text{ s})$) before all hidden tests complete, terminates immediately and is scored as `TIMEOUT` with `correct=False`.
- **Calibration Re-Evaluation**: Re-evaluating the two calibration problems that previously timed out (`arc186_e` and `abc378_g`) under this early-exit rule confirms that neither times out: both fail Test 1 with `WRONG OUTPUT` in about 0.05 s locally and 0.4–0.7 s in Docker, and exit immediately as `FAIL:1/40`.
- **Slowest Single Test Logging (`max_test_wall_s`)**: For each graded program, the runner measures and logs `max_test_wall_s` (wall time in seconds of the slowest single test case, noting that `max_test_wall_s` includes container start-up) in both `log.jsonl` and `programs.jsonl`.

### 4.3 Run Mechanics
1. **Task-Major Execution Order**: The pipeline operates strictly problem-major: for each problem in the fixed order, it executes $A_0$ once, then evaluates conditions C0, C1, C2, C3, and C4 to completion before proceeding to the next problem.
2. **Fixed Processing Order**: Problems are executed in the fixed shuffled order stored in `run_code_data/problem_ids.json`.
3. **Daily-Quota Handling**: On daily quota exhaustion (e.g. Gemini 500 requests/day limit), the runner catches `DailyLimitExhausted`, writes all completed outcomes to disk, and exits cleanly. Execution resumes later from where it stopped with nothing changed. Quota interruptions are standard operational pauses and are not considered protocol deviations.
4. **Timestamps**: Every logged outcome records an ISO 8601 UTC timestamp (`utc_ts`) alongside the Unix epoch timestamp (`ts`).
5. **Empty / Malformed Reviews**: Count as `ACCEPT` with `review_malformed = True` and are reported.
6. **Groq Throttling & Single Retry**:
   - `openai/gpt-oss-20b` calls are throttled to at least 40 seconds apart to prevent hitting the 8,000 TPM limit.
   - If `openai/gpt-oss-20b` returns an empty review, the runner waits 60 seconds and retries once. If still empty, it is logged with `review_retried = True` and treated as `ACCEPT`.
   - Every retry is appended to `run_code_data/review_retries.jsonl` (prompt SHA-256 and whether the retry was also empty) before the retry is made. A C2 row has `review_retried = True` whenever its review prompt appears there, whether or not the retry produced a review. The token record for a retried review is the sum of both provider calls.
7. **Full Critique Logging**: The full, untruncated critique text is logged in `critique_text` for conditions C1–C4.
8. **Program Identity**: Judged primarily by `final_ast_hash == a0_ast_hash` (normalized AST dump with comments and docstrings stripped). The raw text code hash `final_code_hash == a0_code_hash` (first 16 hex characters of SHA-256 of normalized code) is reported alongside.
9. **No Code**: Programs where no Python code block can be extracted receive fingerprint `NO_CODE` and are scored as incorrect.
10. **Strict Pre-Registration Freeze**: Once the main experiment run commences, no code, prompts, models, parameters, timeouts, problem sets, or analysis scripts may be altered. Any anomalies, unexpected errors, or failed hypotheses will be reported as-is without post-hoc modifications.
11. **Program Artifact Logging (`programs.jsonl`)**: All model-generated code (initial programs $A_0$ and all post-review revisions), along with their task ID, condition, stage, code hash, AST hash, `infra_retries`, and `max_test_wall_s` (slowest single test wall time, noting that `max_test_wall_s` includes container start-up), are appended to `run_code_data/programs.jsonl` to ensure complete auditability and offline inspection. Each row (one per task and condition) holds `a0_code` and `final_code` (the extracted program the executor ended with; equal to `a0_code` when there was no revision), and the first-pass `a0_fingerprint` and `final_fingerprint`, so that E3 can re-grade every program and compare fingerprints without the log.
12. **Infrastructure Retries and Sandbox Availability**: If an individual test encounters a host-side backstop timeout (`subprocess.TimeoutExpired`) with no inner exit code 124, it is retried once in a fresh container; `TIMEOUT` is scored only if the retry also hits the backstop or returns exit 124. Docker CLI failures (return codes 125, 126, 127 or stderr containing "Cannot connect to the Docker daemon", "error during connect", "docker: Error", or "failed to connect to the docker API") and container start failures are retried up to 2 times with a 10 s backoff. Any other non-zero exit except 124 counts as a program crash only if the Docker daemon still answers (`docker version`); otherwise it is an infrastructure failure. This rule was added on 27 October 2026, after a smoke test with Docker Desktop stopped showed Docker CLI 29.x exiting 1 with "failed to connect to the docker API", which the earlier list missed and which was scored as `CRASH` for every program. Before the manifest is written, the runner also refuses to start unless the Docker daemon answers and the pinned image is present locally. If failures persist, the runner raises `SandboxUnavailable`, which stops the run cleanly without scoring anything (identical to `DailyLimitExhausted`), allowing seamless later resumption. Infrastructure retries are logged as `infra_retries` per program in both `log.jsonl` and `programs.jsonl`. An infrastructure failure is never scored as `CRASH`, `FAIL`, or `TIMEOUT`.
13. **Clean Stops on Operational and Sandbox Limits**: `RateLimitStop` (triggered after 6 attempts or 300 s of per-minute retries), `DailyLimitExhausted`, and `SandboxUnavailable` stop the run cleanly without scoring anything. Resuming later is standard operational procedure and is not considered a protocol deviation. The stop message prints the raw provider error text and the matched marker (or "no daily marker; per-minute retries exhausted").
14. **Process Concurrency and Resume Integrity**: Exactly one runner is permitted at a time, enforced via a process lock file (`.runner.lock`). On resume, any truncated final line in `log.jsonl` is ignored. `programs.jsonl` is de-duplicated by `(task_id, condition, stage)`, keeping the latest entry. To ensure atomic state transitions, the outcome log row is written last.
15. **Public-Test Fields**: `a0_public_pass` records whether $A_0$ passes every public sample test. `final_public_pass` is `True` when the final program passes every hidden test; equals `a0_public_pass` when the final code hash equals the $A_0$ code hash; and otherwise comes from running the final program on the public sample tests. Neither field enters a confirmatory metric except `a0_public_pass` in the false-accept count (Section 5.1).
16. **Grade Cache**: Grades of code programs are cached on disk by `(task_id, code_hash)` so identical programs are never re-run and resumed runs reuse the identical grade. The cache is bypassed for the math grader (`grade_numeric`), so the frozen math study's code path is unchanged, and it is bypassed by the E3 audit.

---

## 5. Pre-Registered Hypotheses & Success Criteria

All hypotheses are evaluated with explicit pre-registered thresholds:

- **H1 (Self-Review False Rejection)**: Under condition C1, the false-rejection rate on correct $A_0$ programs is at least 20%.  
  *Definition of Supported*: Point estimate $\ge 20\%$. Wilson 95% score confidence interval reported.
- **H2 (Executor Inertia on False Critiques)**: Following a C1 false rejection, the executor leaves its code unchanged (`final_ast_hash == a0_ast_hash`) in the majority of cases. Text-hash inertia (`final_code_hash == a0_code_hash`) is reported alongside.  
  *Definition of Supported*: More than 50% ($> 50\%$) of C1 false rejections leave the AST hash unchanged. Point estimate and Wilson 95% CI reported.
- **H3 (Cross-Family Superiority)**: Condition C2 achieves higher accuracy than baseline C0.  
  *Definition of Supported*: $C_2 \text{ accuracy} > C_0 \text{ accuracy}$ with two-sided exact McNemar test and Holm-adjusted $p < 0.05$.
- **H4 (Deterministic Public Test Baseline)**: Condition C4 exhibits a false-rejection rate near zero and some wrong programs pass all public tests.  
  *Definition of Supported*: $C_4 \text{ false-rejection rate} \le 5\%$ and false-accept count is strictly $> 0$ (at least one wrong $A_0$ program passes all public tests).

### 5.1 False Accept Definition
A **false accept** is defined as an $A_0$ program that fails one or more hidden tests, but passes all public sample tests from the problem statement ($A_0 \text{ incorrect on hidden} \land A_0 \text{ passes all public tests}$).

### 5.2 Low-$n$ Rule
- Any proportion with denominator $n < 20$ will be explicitly flagged as "low-$n$".
- If the total number of C1 false rejections is fewer than 10 ($n_{\text{false\_rej}} < 10$), **H2 will be formally declared "inconclusive"** due to insufficient statistical power.

---

## 6. Statistical Analysis Plan

All analyses are implemented in `analyze_code.py` with zero API calls:

1. **Paired Accuracy (Exact McNemar's Test)**:
   - Evaluated on discordant pairs $(b, c)$ across the $N=125$ paired problems:
     - $C_1 \text{ vs } C_0$
     - $C_2 \text{ vs } C_0$
     - $C_3 \text{ vs } C_0$
     - $C_4 \text{ vs } C_0$
     - $C_4 \text{ vs } C_2$
   - Exact two-sided binomial p-values conditioned on discordant pairs ($b+c$).
2. **Multiplicity Correction**: Holm-Bonferroni step-down correction applied across the 5 primary pre-registered McNemar comparisons.
3. **Per-Rejection Proportions (Exact Fisher Tests)**:
   - Two-sided Fisher's exact test on repair rate ($\text{repairs} / \text{rejections}$) and no-change rate between active reviewer conditions.
4. **All Primary Pipeline Metrics Computed (matching Section 7 of the main study)**:
   - Accuracy, $\Delta$ vs C0, Recall, False-Rejection Rate, Repair Rate, Damage Rate.
   - **No-Change Rate split by $A_0$ correctness**: Evaluated primarily via AST hash (`ast_hash`), with text-hash (`code_hash`) reported alongside. Split into `nc-wrong` (critique ignored on wrong code: executor fails to adopt revision) vs `nc-right` (harmless false rejection: executor correctly retains valid logic).
   - **False-Accept Rate**: $P(\text{pass public} \mid A_0 \text{ fails hidden})$: proportion of hidden-test failing $A_0$ programs that pass all public sample tests.
   - **Timeout Rate**: Proportion of programs where hidden test execution terminated with `TIMEOUT` (either per-test duration $\ge 6.0$s or total problem budget exhausted) among all evaluated tasks in that condition ($n_{\text{timeout}} / n_{\text{total}}$).
   - **Token Cost**: Mean tokens per task and tokens per additional correct answer relative to C0 baseline.
5. **Rule on Failed Hypotheses**: Any failed hypothesis will be reported plainly as not supported; no post-hoc filtering or window alterations are permitted.
6. **Hypothesis Verdicts**: `analyze_code.evaluate_hypotheses` applies the Section 5 thresholds exactly, including the low-$n$ flag and the H2 inconclusive rule, and returns one of `supported`, `not supported`, `inconclusive` or `not evaluable` for each of H1–H4. H3 uses the Holm-adjusted p-value from the same five McNemar comparisons (`analyze_code.paired_mcnemar`). Verdicts are written to `run_code_data/RESULTS.md`, Section 6.
7. **Implementation Map** (all in `analyze_code.py` unless stated):

   | Item | Function |
   |---|---|
   | H1–H4 verdicts | `evaluate_hypotheses` (H3 via `paired_mcnemar`) |
   | Accuracy, $\Delta$ vs C0, Wilson CIs | `run_analysis` §1, `wilson_ci` |
   | McNemar (5 pairs) and Holm correction | `run_analysis` §2, `exact_binom_two_sided`, `holm_bonferroni` |
   | Recall, false-rejection, repair, damage, malformed-review rates | `run_analysis` §3 |
   | No-change split (AST and text hash), Fisher tests | `run_analysis` §4, `classify_rejection`, `fisher_exact_two_sided` |
   | Timeout rate, C4 false accepts, token cost | `run_analysis` §5 |
   | E1 | `analyze_critiques.py`: `export_blind_template`, `analyze_labeled` |
   | E2 | `temporal_sensitivity` |
   | E3 | `scripts/audit_regrade.py`: `audit` |

   `rej_quality.py`, `mcnemar.py` and `no_change.py` are also run on the code log with `--log` and `--conds` for direct comparison with the math study.

---

## 7. Exploratory Analyses (Non-Confirmatory)

- **E1 (Critique Specificity vs Adoption)**:
  - Implemented in `analyze_critiques.py`.
  - Restricted strictly to LLM reviewer conditions C1–C3 (C4 deterministic test output is excluded).
  - Exported to a double-blind CSV template with shuffled rows; both the `condition` column and execution outcome columns are omitted from the CSV to preserve strict blinding.
  - The unblinding key mapping `item_id` to `condition` and outcomes is saved to a separate JSON sidecar file.
  - Hand-labeled for specificity (`specific`: names exact failing input, logic edge case, or trace vs `vague`: generic disapproval) blind to condition and execution outcome (`final_correct`).
  - Reports executor adoption rate ($\text{final\_ast\_hash} \neq \text{a0\_ast\_hash}$ as primary, and text hash $\text{final\_code\_hash} \neq \text{a0\_code\_hash}$ alongside) by specificity category overall and broken down by C1–C3. Purely descriptive counts and percentages; no significance tests.
- **E2 (Temporal Sensitivity Analysis across Cutoff Boundary 2025-01-31)**:
  - Pre-registered exploratory analysis. Evaluates $A_0$ accuracy and C1 false-rejection rate pre- vs post-cutoff (`2025-01-31` boundary):
    - **Pre-cutoff** ($\le 2025-01-31$): 78 eligible problems.
      - **Difficulty mix**: 25 medium (32.1%), 53 hard (67.9%).
      - **Series mix**: 63 ABC (80.8%), 15 ARC (19.2%).
    - **Post-cutoff** ($> 2025-01-31$): 47 eligible problems.
      - **Difficulty mix**: 15 medium (31.9%), 32 hard (68.1%).
      - **Series mix**: 33 ABC (70.2%), 14 ARC (29.8%).
    - **Balanced Difficulty**: The difficulty distribution is virtually identical across both sides of the boundary ($32.1\%$ vs $31.9\%$ medium; $67.9\%$ vs $68.1\%$ hard).
    - **Contest Month Breakdown**: `2024-10` (22), `2024-11` (22), `2024-12` (15), `2025-01` (19), `2025-02` (18), `2025-03` (25), `2025-04` (4).
  - Reports Wilson 95% score confidence intervals for pre- and post-cutoff strata and evaluates differences with a two-sided exact Fisher test for $A_0$ accuracy.
  - Re-evaluates hypotheses H1–H4 restricted to the 47 post-cutoff problems as a sensitivity check (explicitly flagged as low-$n$).
  - Implemented in `analyze_code.temporal_sensitivity`, using the first 10 characters (`YYYY-MM-DD`) of each row's logged `contest_date`. Problems with no `contest_date` are excluded from E2 and listed. Written to `run_code_data/RESULTS.md`, Section 7.
- **E3 (Execution Stability Audit / Second Docker Pass)**:
  - After the main experiment run completes, every model-generated program recorded in `run_code_data/programs.jsonl` (initial $A_0$ programs and all post-review revisions across conditions C0–C4) will be re-graded in a complete second pass inside disposable Docker containers under identical sandbox resource limits and early-exit rules.
  - Reports the proportion and listing of any fingerprint disagreements between the primary pass and the audit pass ($\text{fingerprint}_1 \neq \text{fingerprint}_2$), specifically identifying any borderline timeouts or non-deterministic execution behaviors.
  - Primary confirmatory results and all official pipeline metrics strictly use the first-pass outcomes; the second pass serves purely as an empirical reproducibility and sandbox stability audit.
  - Implemented in `scripts/audit_regrade.py`. It calls `grade_lcb` directly, bypassing the grade cache, and refuses to run without Docker or with `UNSAFE_LOCAL_EXEC` set. Each distinct `(task_id, code_hash)` program is run once, and its second-pass fingerprint is compared with every first-pass fingerprint recorded for it. Reports distinct programs and records disagreeing, and writes `run_code_data/audit_regrade.json`.

---

## 8. Known Operational Risk: gpt-oss-20b Token Budget

- **Measurement**: Five real `openai/gpt-oss-20b` review calls on the five longest review prompts (`run_code_data/groq_tokens_measured.json`, `max_tokens = 6,144`) returned non-empty reviews with a `VERDICT` line. They used 1,379–1,893 prompt tokens and 125–218 completion tokens. The largest prompt plus `max_tokens` among them was 1,893 + 6,144 = 8,037, and that call succeeded.
- **Offline Projection**: The measured pooled ratio is 3.174 characters per token (per-call 2.685–3.599). Applied to the review prompts of all 125 problems with the longest calibration program (3,170 characters), estimated prompt tokens are min 1,325, mean 1,709.5, max 2,393 (`scripts/estimate_offline_tokens.py`). With `max_tokens = 6,144`, 20 of 125 problems exceed 8,000 tokens (max 8,537). For the C3 Llama reviewer (`max_tokens = 2,048`) the maximum is 4,441 and none of the 125 exceed 6,000; this projection used gpt-oss's tokenizer ratio, and with `num_ctx = 8,192` the largest projected C3 request (prompt plus 2,048 output tokens) fits; a real 1,341-token review prompt was measured on 18 November 2026.
- **Live Probe (25 October 2026)**: A realistic 2,675-token review prompt to `openai/gpt-oss-20b` succeeded with `max_tokens = 6,144` (prompt plus `max_tokens` = 8,819 > 8,000; HTTP 200, 295 and 1,466 completion tokens on two calls) and with `max_tokens = 4,096` (HTTP 200, 1,644 completion tokens). Groq therefore does not reject a request because its prompt plus `max_tokens` exceeds the per-minute cap. Headers: `x-ratelimit-limit-tokens: 8000`, `x-ratelimit-limit-requests: 1000`.
- **Daily Quotas and Expected Stops**: Groq's rate-limit documentation (https://console.groq.com/docs/rate-limits, retrieved 27 October 2026) lists `openai/gpt-oss-20b` at 1,000 requests and 200,000 tokens per day, per organization; the account's console Limits page is authoritative. At roughly 2,000–3,000 total tokens per C2 review, 125 reviews need about 250,000–375,000 tokens, so C2 is expected to reach the daily token cap and finish on a second day. Gemini's daily request quota resets at midnight Pacific time (https://ai.google.dev/gemini-api/docs/rate-limits). Either stop is a `DailyLimitExhausted` clean stop under Section 4.3, item 3, and is not a deviation.
- **Decision**: `max_tokens` stays at 6,144 for gpt-oss-20b and 2,048 for llama, unchanged from the math study. Mitigation is the 40 s throttle and single empty-review retry (Section 4.3, item 6). Any empty or malformed review that remains is counted as `ACCEPT`, flagged `review_malformed`, and reported per condition (`run_code_data/RESULTS.md`, Section 2), so the C2 results can be read against the C2 malformed-review rate.

