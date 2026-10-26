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
