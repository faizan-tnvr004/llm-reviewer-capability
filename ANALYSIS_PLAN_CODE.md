# Pre-Registered Analysis Plan: Code Extension Study

**Title**: Reviewer Capability in Execute-Review-Revise Pipelines: Executable Code Extension  
**Repository**: `faizan-tnvr004/llm-reviewer-capability`  
**Branch**: `code-extension`  
**Date**: 24 October 2026  
**Hard Stop**: 7 November 2026  
**Prompts SHA-256 Hash**: `d7b7203e7603dc5070bc42efdedc140de146caa88975f7ef1036d3271b56ca73`

---

## 1. Goal and Research Questions

This study replicates and extends the Olympiad Math pipeline-error study on executable Python programming tasks drawn from LiveCodeBench-Lite. We evaluate whether the central findings of self-review failure (high false rejections, executor inertia, destructive revisions) and cross-family reviewer benefit generalize to programming with ground-truth test execution, and introduce Condition C4 (deterministic public sample test feedback).

- **RQ1**: Does self-review falsely reject correct programs at a high rate? Does a cross-family reviewer raise accuracy without damage?
- **RQ2**: Does the executor leave its code unchanged after a false rejection, and do the changes it does make break correct code?
- **RQ3**: How does public-test feedback (C4) compare with LLM verdicts on detection, false rejection, and repair? How often do public sample tests accept a program that fails hidden tests (false accept)?

---

## 2. Benchmark & Problem Selection

### 2.1 Dataset Source
- **Dataset**: `livecodebench/code_generation_lite` (HuggingFace)
- **Pinned Release Files**:
  - `test.jsonl` (400 problems)
  - `test2.jsonl` (111 problems)
  - `test3.jsonl` (101 problems)
  - `test4.jsonl` (101 problems)
  - `test5.jsonl` (167 problems)
  - `test6.jsonl` (175 problems)
  - **Total**: 1,055 problems across dates `2023-05-07` to `2025-04-06`.

### 2.2 Filtering Criteria
1. **Stdin-only IO**: AtCoder and Codeforces problems where `testtype == "stdin"`. Functional / LeetCode problems (e.g. `class Solution:`) are excluded.
2. **Difficulty**: `medium` and `hard` only.
3. **Contest Date Window**: Recent contest window `2024-10-01` to `2025-04-06` (strictly after the executor's knowledge cutoff).
4. **No Special-Judge Problems**: Statements containing non-deterministic output phrases (`"any of them"`, `"print any"`, `"if there are multiple"`, `"absolute or relative error"`, `"relative error"`, `"absolute error"`) or `is_special_judge` flags are excluded.

### 2.3 Calibration Batch (Phase B)
- **Batch 1**: Evaluated 20 candidate problems on `gemini-3.1-flash-lite` (executor only):
  - Solved: 8 / 20 = **40.0%** (target 40% – 60% met).
  - Fingerprints: `PASS`: 8, `FAIL`: 10, `TIMEOUT`: 2, `CRASH`: 0.
  - The 20 calibration problem IDs (`arc186_e`, `arc189_a`, `abc390_c`, `arc196_d`, `abc383_e`, `arc195_a`, `abc382_d`, `abc376_b`, `abc377_f`, `abc376_d`, `arc194_b`, `abc394_c`, `abc394_g`, `arc189_b`, `arc195_c`, `abc385_c`, `abc381_e`, `abc378_g`, `abc386_e`, `abc392_g`) are permanently excluded from the main study.

### 2.4 Pre-Registered Problem Selection
- **Candidate Pool**: 135 remaining candidate problems in the approved window.
- **Sample Size**: $N = 100$.
- **Selection Seed**: `42`.
- **Frozen File**: `run_code_data/problem_ids.json` (committed prior to main experiment execution).
- **Composition**: 32 medium, 68 hard; 100% stdin testtype.

---

## 3. Experimental Conditions & Model Lineup

The executor solves each problem once to produce program $A_0$, which is shared across all conditions. On `REJECT`, the executor revises once.

| Condition | Role | Reviewer Model / Mechanism | Provider / Platform | Parameters |
|---|---|---|---|---|
| **C0** | Baseline (No Review) | None ($A_0$ is final) | N/A | N/A |
| **C1** | Self-Review | `gemini-3.1-flash-lite` | Google AI Studio | Temp = 0.7, max tokens = 8,192 |
| **C2** | Cross-Family Mid-Tier | `openai/gpt-oss-20b` | Groq | Temp = 0.7, max tokens = 6,144 |
| **C3** | Weak Reviewer | `llama-3.1-8b-instant` | Groq | Temp = 0.7, max tokens = 2,048 |
| **C4** | Public Sample Tests | Deterministic test runner | Local Sandbox | ToolReviewer (0 tokens) |

---

## 4. Operational and Handling Rules

1. **Deterministic Public Tests (C4)**: Evaluates extracted code against public sample test cases from the problem statement. If all pass, returns `VERDICT: ACCEPT`. Otherwise returns `VERDICT: REJECT` with the first failing input, expected output, and actual output/crash trace (capped at 1,000 chars per field).
2. **Review Formatting**: Reviewers must output `VERDICT: ACCEPT` or `VERDICT: REJECT` plus `CRITIQUE: <text>`. Empty or malformed reviews count as `ACCEPT` with `review_malformed = True`.
3. **Groq / gpt-oss Throttling & Retry**:
   - Calls to `openai/gpt-oss-20b` are throttled to $\ge 40$ seconds apart to respect the 8,000 TPM limit.
   - If `openai/gpt-oss-20b` returns an empty review, the runner waits 60 seconds and retries once. If still empty, it is logged with `review_retried = True` and counted as `ACCEPT`.
4. **Code Execution Sandbox**:
   - Program execution runs in a sandboxed subprocess with `UNSAFE_LOCAL_EXEC=1` (or disposable Docker container).
   - Timeouts: 6.0 seconds per test case, 25.0 seconds outer wall-clock limit per problem.
   - Output comparison: lines normalized with trailing whitespace stripped, unified line endings (`\n`), trailing blank lines dropped.
5. **Program Identity & Code Hash**:
   - Program identity is judged by `final_code_hash == a0_code_hash` (first 16 hex characters of SHA-256 of normalized code).
   - `NO_CODE` (failure to extract Python code fence) is scored as incorrect.
   - Timeouts are logged as fingerprint `TIMEOUT` and distinguished from wrong output (`FAIL:k/n`).

---

## 5. Pre-Registered Hypotheses

- **H1 (Self-Review False Rejection)**: Under condition C1, the false-rejection rate on correct $A_0$ programs is at least 20%.
- **H2 (Executor Inertia on False Critiques)**: Under condition C1, following a false rejection, the executor leaves its code unchanged in the majority of cases ($> 50\%$).
- **H3 (Cross-Family Superiority)**: Condition C2 achieves higher accuracy than baseline C0, evaluated by exact McNemar's test.
- **H4 (Deterministic Public Test Baseline)**: Condition C4 exhibits a false-rejection rate near zero ($\le 5\%$) and a false-accept count strictly greater than zero ($> 0$).

*Rule*: All pre-registered hypotheses will be explicitly reported as supported or not supported; no hypotheses will be silently dropped or altered.

---

## 6. Statistical Analysis Plan

All statistical comparisons are pre-specified and implemented in `analyze_code.py`:

1. **Paired Accuracy (Exact McNemar's Test)**:
   - Evaluated on discordant pairs $(b, c)$ across the 100 paired problems:
     - $C_1 \text{ vs } C_0$
     - $C_2 \text{ vs } C_0$
     - $C_3 \text{ vs } C_0$
     - $C_4 \text{ vs } C_0$
     - $C_4 \text{ vs } C_2$
   - Exact two-sided binomial p-values conditioned on discordant pairs.
2. **Multiplicity Correction**: Holm-Bonferroni step-down correction applied across the 5 primary pre-registered McNemar comparisons.
3. **Per-Rejection Proportions (Exact Fisher Tests)**:
   - Two-sided Fisher's exact test on repair rate ($\text{repairs} / \text{rejections}$) and no-change rate between active reviewer conditions.
4. **Confidence Intervals**: Wilson 95% score confidence intervals on all proportions (accuracy, recall, false-rejection rate, repair rate, damage rate, malformed rate).
