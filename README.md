# Reviewer Capability in LLM Execute–Review–Revise Pipelines

An empirical study of how reviewer capability affects the accuracy of an
Execute–Review–Revise (ERR) pipeline for olympiad mathematics, using a fixed
executor (Gemini 3.1 Flash-Lite) and three reviewer tiers on 100 Omni-MATH
problems.

## Headline Results

| Condition | Reviewer | Accuracy | Δ vs baseline | p (McNemar) |
|-----------|----------|----------|---------------|-------------|
| C0 | None (baseline) | 52% | — | — |
| C1 | Self-review (Gemini) | 58% | +6 pp | 0.1460 |
| C2 | Cross-family mid-tier (GPT-OSS-20B) | 64% | **+12 pp** | **0.0005** |
| C3 | Weak (Llama 3.1 8B) | 52% | 0 pp | 1.0000 |

The cross-family reviewer (C2) improves accuracy from 52% to 64%
(p = 0.0005, exact McNemar) with **zero damaged answers**. Self-review (C1)
has the highest detection rate (0.85 recall) but falsely rejects 35% of
correct answers; its low damage rate is an artifact of the executor ignoring
83% (15 of 18) of its false rejections. The weak reviewer (C3) changed zero of 100 final
answers while doubling token cost.

## Repository Layout

```
src/
  core.py            # Task dataclass, answer extraction, numeric grading
  harness.py         # Disk cache, JSONL logging, retry/backoff, run loop
  pipeline.py        # ERR pipeline, prompts, token accounting, analysis
  providers.py       # API wrappers (Google AI Studio, Groq); the only file
                     #   that touches real APIs
  tasks/
    omnimath_tasks.py    # Omni-MATH loader (HuggingFace datasets)
    math_tasks.py        # Procedural math-word-problem generator
    code_tasks.py        # Procedural code-debugging task generator
    constraint_tasks.py  # Procedural constraint-puzzle generator

tests/               # Offline tests (no API keys needed); run as scripts
  test_math.py           test_harness.py      test_pipeline.py
  test_code.py           test_constraint.py   test_math_robustness.py

run_experiment.py    # Main entry point for the experiment

mcnemar.py           # McNemar's exact test on paired condition comparisons
no_change.py         # Rejection-without-change analysis
rej_quality.py       # Rejection-quality breakdown (repair/damage/no-change)

run_experiment_data/
  log.jsonl          # Full experiment log (100 tasks × 4 conditions)
  tokens.jsonl       # Per-call token usage records
  cache/             # Response cache (gitignored; not needed to reproduce
                     #   the analysis, only to resume an interrupted run)

ANALYSIS_PLAN.md              # Pre-registered analysis plan (see below)
ANALYSIS_PLAN_AMENDMENT.md    # Amendment documenting the change of research
                              #   question after calibration
```

## How to Reproduce

```bash
# 1. Clone and set up a virtualenv
git clone https://github.com/faizan-tnvr004/llm-reviewer-capability.git
cd llm-reviewer-capability
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Set API keys (values not stored in the repo)
export GOOGLE_API_KEY=...       # Google AI Studio key for Gemini
export GROQ_API_KEY=...         # Groq key for Llama and GPT-OSS
# Windows PowerShell:
#   $env:GOOGLE_API_KEY="..."
#   $env:GROQ_API_KEY="..."

# 3. Run the experiment
python run_experiment.py
```

The harness **caches every API response to disk** and **resumes after
interruption**. Free-tier rate limits will cause daily stops; simply re-run
the next day and the cache picks up where it left off. No work is repeated.

## Reproduce the Analysis

The four analysis scripts read the committed `run_experiment_data/log.jsonl`
and require **no API access**:

```bash
python mcnemar.py        # McNemar's exact test on paired accuracy comparisons
python rej_quality.py    # Per-rejection outcome breakdown with Fisher tests
python no_change.py      # No-change rate (rejected but answer unchanged)
python run_experiment.py # Prints per-condition metrics (re-run reads the
                         #   existing log; only makes API calls if the log
                         #   is incomplete)
```

To run the offline test suite (no API keys needed):

```bash
python tests/test_math.py
python tests/test_code.py
python tests/test_constraint.py
python tests/test_harness.py
python tests/test_pipeline.py
python tests/test_math_robustness.py
```

## Pre-Registered Analysis Plan

[ANALYSIS_PLAN.md](ANALYSIS_PLAN.md) is the original pre-registered plan,
**committed before any experiments were run** (commit `6844da4`, 2026-07-21)
and preserved unchanged.
[ANALYSIS_PLAN_AMENDMENT.md](ANALYSIS_PLAN_AMENDMENT.md) documents the
change of research question made after calibration showed the original
design was infeasible with the available models. Both are retained as a
record of the project's methodological trajectory.

## Limitations

- **N = 100 problems.** Sufficient to detect the large C2 effect but
  underpowered for small effects; the non-significant C1 result may be a
  false negative.
- **Single executor.** All results are conditional on Gemini 3.1 Flash-Lite
  as executor.
- **Numeric-answer subset.** Only problems with clean integer/decimal
  answers are included (no LaTeX, no free-form); this skews toward
  number-theory and combinatorics.
- **Single difficulty band** (Omni-MATH difficulty 5–6).
- **Free-tier provider constraints.** Daily rate limits necessitated
  multi-day runs. In the reported N = 100 run (`run_experiment_data/log.jsonl`),
  16 of 100 GPT-OSS (C2) reviews were empty or unparseable and were
  conservatively scored as ACCEPT (`review_malformed = true`); C1 and C3 had
  none. The 29% empty rate mentioned in `ANALYSIS_PLAN_AMENDMENT.md` §9 came
  from the earlier run, before reviewer input was truncated to fit Groq's
  8 000 TPM cap; every review was regenerated after that change.

## Code Extension (LiveCodeBench)

The same execute–review–revise design, applied to executable code. The `code-extension` branch adds one condition, **C4**, in which the problem's public sample tests act as the reviewer.

- **Pre-registration**: [ANALYSIS_PLAN_CODE.md](ANALYSIS_PLAN_CODE.md), tagged `prereg-code-v1` before the main run. Deviations and every run incident are in [ANALYSIS_PLAN_CODE_AMENDMENT.md](ANALYSIS_PLAN_CODE_AMENDMENT.md).
- **Problems**: all 125 eligible LiveCodeBench `code_generation_lite` problems (stdin, medium or hard, contests from 2024-10-01 to 2025-04-06, no special judge, at least 10 hidden tests, calibration problems excluded). A program counts as correct only if it passes every hidden test, run in a network-less Docker sandbox with a 6 s limit per test.
- **Models**: the executor is `gemini-3.1-flash-lite`.
  - C1: self-review, by the same model.
  - C2: `openai/gpt-oss-20b` on Groq.
  - C3: Llama 3.1 8B Instruct (`llama3.1:8b-instruct-q8_0`), run locally in Ollama 0.40.16, because Groq retired `llama-3.1-8b-instant`.
  - C4: the public sample tests, with no model.
- **Results**: [run_code_data/RESULTS.md](run_code_data/RESULTS.md), produced by `analyze_code.py`. The second-pass audit (E3) is in [run_code_data/E3_AUDIT.md](run_code_data/E3_AUDIT.md).

| Condition | Reviewer | Accuracy | Δ vs C0 | Exact McNemar vs C0 (raw / Holm) |
|---|---|---|---|---|
| C0 | None | 45.6% (57/125) | — | — |
| C1 | Self-review (Gemini) | 48.0% (60/125) | +2.4 pts | 0.549 / 1.000 |
| C2 | Cross-family (GPT-OSS-20B) | 50.4% (63/125) | +4.8 pts | 0.109 / 0.438 |
| C3 | Weak (Llama 3.1 8B) | 45.6% (57/125) | 0.0 pts | 1.000 / 1.000 |
| C4 | Public sample tests | 50.4% (63/125) | +4.8 pts | 0.031 / 0.156 |

Pre-registered hypotheses:

- **H1, supported**: self-review falsely rejected 20 of 57 correct programs (35.1%, Wilson 95% CI 24–48%), the same rate as in the math study (35%).
- **H2, not supported**: after a false self-rejection, the executor rewrote 19 of 20 correct programs and left only 1 unchanged. In math it ignored 15 of 18. 4 of the 19 rewrites broke a correct program.
- **H3, not supported**: the cross-family reviewer's +4.8 point gain (2 programs broken, 8 fixed) is not significant (Holm p = 0.44). The math study's significant cross-family gain did not replicate on code.
- **H4, supported**: the public-test reviewer falsely rejected 0 of 57 correct programs (95% upper bound 6.3%; the pre-registered criterion of 5% or less applies to the point estimate), but 19 of 68 wrong programs (27.9%) passed every sample test.

No difference here reaches significance after Holm correction across the five pre-registered comparisons. With N = 125, effects of a few points cannot be ruled out; see the intervals in `RESULTS.md`.

To reproduce the analysis (no API calls):

```bash
python analyze_code.py
```
```bash
python scripts/audit_regrade.py
```

The math scripts also accept the code log, for example `python mcnemar.py --log run_code_data/log.jsonl --conds C0 C1 C2 C3 C4`.

### Code-study limitations

- **Single executor and benchmark**: all results depend on `gemini-3.1-flash-lite` and on LiveCodeBench problems from late 2024 to early 2025. The executor's stated knowledge cutoff is January 2025, so contamination cannot be excluded. The pre- and post-cutoff split (E2) found no difference in A0 accuracy (34/78 vs 23/47, Fisher p = 0.58).
- **C3 is not the math study's C3**: it uses the same Llama 3.1 8B Instruct weights, but served locally at Q8_0 instead of on Groq.
- **Empty gpt-oss reviews**: 12 of 125 C2 reviews (9.6%) were empty or unparseable after one retry and were scored as ACCEPT, as pre-registered.
- **Borderline timeouts**: two A0 programs have a slowest test of 5–7 s against the 6 s limit. They passed on the first (primary) pass, timed out on the audit pass, and passed again on a third run (E3).
- **Incomplete token counts**: exact provider counts are missing for 28 problems after a manual restart. Token cost is reported on the 97 problems with exact counts, and separately with chars/4 estimates for the rest.
- **Run interruptions**: four provider requests never returned and had to be killed and restarted, and there were three Gemini outages. None of these changed an outcome; all are listed in the amendment.
- **`rej_quality.py` on code**: its "corrected false-rejection rate" counts a correct program rewritten into a different correct one as "other", so it undercounts false rejections on code. Use the false-rejection rate in `RESULTS.md`.
- **E1 labels**: a blind LLM labeller's critique-specificity labels agreed with a human on only 19 of 30 items (Cohen's kappa = 0.046), so they are not used as a specificity measure. The E1 conclusion does not depend on labels: the executor revised its program after 153 of 154 rejections (`run_code_data/e1/E1_SUMMARY.md`).
