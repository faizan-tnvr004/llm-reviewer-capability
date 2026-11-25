# Code Extension: Empirical Results

**Log source**: `run_code_data\log.jsonl`  
**Completed outcomes**: 625  
**Conditions**: C0, C1, C2, C3, C4  

## 1. Primary Paired Accuracy & McNemar Tests

| Condition | Reviewer | N | Correct | Accuracy | Wilson 95% CI | Δ vs C0 |
|---|---|---|---|---|---|---|
| **C0** | None (baseline) | 125 | 57 | **45.6%** | [0.371, 0.543] | +0.0% |
| **C1** | Self-review (Gemini 3.1 Flash-Lite) | 125 | 60 | **48.0%** | [0.394, 0.567] | +2.4% |
| **C2** | Cross-family (openai/gpt-oss-20b) | 125 | 63 | **50.4%** | [0.418, 0.590] | +4.8% |
| **C3** | Weak (llama-3.1-8b-instant) | 125 | 57 | **45.6%** | [0.371, 0.543] | +0.0% |
| **C4** | Public sample tests (deterministic) | 125 | 63 | **50.4%** | [0.418, 0.590] | +4.8% |

### Pre-registered Paired Comparisons (Exact McNemar)

| Comparison | Acc A | Acc B | Diff | Discordant (b / c) | Raw p | Holm-Adj p | Verdict |
|---|---|---|---|---|---|---|---|
| **C0 vs C1** | 45.6% | 48.0% | +2.4% | 4 / 7 | 0.5488 | 1.0000 | ns |
| **C0 vs C2** | 45.6% | 50.4% | +4.8% | 2 / 8 | 0.1094 | 0.4375 | ns |
| **C0 vs C3** | 45.6% | 45.6% | +0.0% | 0 / 0 | 1.0000 | 1.0000 | ns |
| **C0 vs C4** | 45.6% | 50.4% | +4.8% | 0 / 6 | 0.0312 | 0.1562 | ns |
| **C2 vs C4** | 50.4% | 50.4% | +0.0% | 5 / 5 | 1.0000 | 1.0000 | ns |

## 2. Reviewer Quality & Error Detection

| Condition | Recall on Wrong A0 | False Rejection Rate (FRR) | Repair Rate | Damage Rate | Malformed Review Rate |
|---|---|---|---|---|---|
| **C0** | 0.0% [0.00, 0.05] | 0.0% [0.00, 0.06] | 0.0% [0.00, 0.00] | 0.0% [0.00, 0.00] | 0.0% |
| **C1** | 80.9% [0.70, 0.88] | 35.1% [0.24, 0.48] | 12.7% [0.06, 0.24] | 20.0% [0.08, 0.42] | 0.0% |
| **C2** | 92.6% [0.84, 0.97] | 22.8% [0.14, 0.35] | 12.7% [0.07, 0.23] | 15.4% [0.04, 0.42] | 9.6% |
| **C3** | 4.4% [0.02, 0.12] | 0.0% [0.00, 0.06] | 0.0% [0.00, 0.56] | 0.0% [0.00, 0.00] | 0.0% |
| **C4** | 72.1% [0.60, 0.81] | 0.0% [0.00, 0.06] | 12.2% [0.06, 0.24] | 0.0% [0.00, 0.00] | 0.0% |

## 3. Rejection Outcomes & Quality Breakdown

Rejection classification uses normalized AST hash (`ast_hash`) as the primary definition of program identity (ignoring comments, docstrings, and formatting). Text-hash counts (`final_code_hash == a0_code_hash`) are reported alongside in brackets `[text]`.

In this table, "Other" means the program changed and its correctness did not flip: wrong to a different wrong program, *or correct to a different correct program*. In code the second case is common, because the executor nearly always rewrites a program it is told is wrong. For the same reason, `rej_quality.py`'s "corrected false-rejection rate" ((damage + nc-right) / correct A0) undercounts false rejections on this log, since a correct program rewritten into another correct one counts as "other". The pre-registered false-rejection rate is the one in Section 2 (rejected correct A0 / correct A0).

| Condition | Rejections | Repair (Useful) | Damage (Harmful) | No-Change Wrong (Ignored) [AST / Text] | No-Change Right (Harmless FR) [AST / Text] | Other |
|---|---|---|---|---|---|---|
| **C0** | 0 | - | - | - | - | - |
| **C1** | 75 | 7 (9%) | 4 (5%) | 0 (0%) [text: 0] | 1 (1%) [text: 0] | 63 (84%) |
| **C2** | 76 | 8 (11%) | 2 (3%) | 0 (0%) [text: 0] | 0 (0%) [text: 0] | 66 (87%) |
| **C3** | 3 | 0 (0%) | 0 (0%) | 0 (0%) [text: 0] | 0 (0%) [text: 0] | 3 (100%) |
| **C4** | 49 | 6 (12%) | 0 (0%) | 0 (0%) [text: 0] | 0 (0%) [text: 0] | 43 (88%) |

### Exact Fisher Tests on Per-Rejection Rates

| Comparison | Metric | Proportion A | Proportion B | Raw p | Holm-Adj p | Sig |
|---|---|---|---|---|---|---|
| **C1 vs C2** | repair rate | 7/75 (9%) | 8/76 (11%) | 1.0000 | 1.0000 | ns |
| **C1 vs C2** | no-change rate | 1/75 (1%) | 0/76 (0%) | 0.4967 | 1.0000 | ns |
| **C1 vs C3** | repair rate | 7/75 (9%) | 0/3 (0%) | 1.0000 | 1.0000 | ns |
| **C1 vs C3** | no-change rate | 1/75 (1%) | 0/3 (0%) | 1.0000 | 1.0000 | ns |
| **C1 vs C4** | repair rate | 7/75 (9%) | 6/49 (12%) | 0.7655 | 1.0000 | ns |
| **C1 vs C4** | no-change rate | 1/75 (1%) | 0/49 (0%) | 1.0000 | 1.0000 | ns |
| **C2 vs C3** | repair rate | 8/76 (11%) | 0/3 (0%) | 1.0000 | 1.0000 | ns |
| **C2 vs C3** | no-change rate | 0/76 (0%) | 0/3 (0%) | 1.0000 | 1.0000 | ns |
| **C2 vs C4** | repair rate | 8/76 (11%) | 6/49 (12%) | 0.7782 | 1.0000 | ns |
| **C2 vs C4** | no-change rate | 0/76 (0%) | 0/49 (0%) | 1.0000 | 1.0000 | ns |
| **C3 vs C4** | repair rate | 0/3 (0%) | 6/49 (12%) | 1.0000 | 1.0000 | ns |
| **C3 vs C4** | no-change rate | 0/3 (0%) | 0/49 (0%) | 1.0000 | 1.0000 | ns |

## 4. Code Execution & Resource Metrics

- **C4 False-Accept Count / Rate**: 19/68 (27.9%) wrong A0 programs passed all public sample tests.

Tokens use exact provider usage on the 97 problems whose records survived (Amendment 1, Section 2). The all-problems column adds re-estimated chars/4 records for the rest, which miss hidden reasoning tokens.

| Condition | Timeouts | Timeout Rate | Mean Tokens / Task (exact problems) | Cost / Extra Correct Answer (exact problems) | Mean Tokens / Task (all problems, incl. estimates) |
|---|---|---|---|---|---|
| **C0** | 11/125 | 8.8% | 1722 | inf | 1634 |
| **C1** | 13/125 | 10.4% | 4609 | 447033 | 4402 |
| **C2** | 13/125 | 10.4% | 7643 | 148273 | 6772 |
| **C3** | 13/125 | 10.4% | 3010 | inf | 2889 |
| **C4** | 13/125 | 10.4% | 2640 | 85360 | 2564 |

## 5. Olympiad Math vs Executable Code Comparison

The verdict column is computed from this log. Math figures are the published math-study results.

| Finding / Metric | Math Study (Olympiad) | Code Study (LiveCodeBench) | Replication Finding |
|---|---|---|---|
| Baseline Executor Accuracy (C0) | 52% (52/100) | 45.6% (57/125) | Baselines comparable |
| Self-Review Delta (C1 vs C0) | +6.0% (52% -> 58%, p=0.1460 ns) | +2.4% | Replicated: no significant gain in either study (+2.4 pts, b/c = 4/7, Holm p = 1.0000) |
| Cross-Family Delta (C2 vs C0) | +12.0% (52% -> 64%, p=0.0005 ***) | +4.8% | Not replicated (+4.8 pts, b/c = 2/8, Holm p = 0.4375) |
| Self-Review False Rejection Rate (C1 FRR) | 35% (18/52 correct answers) | 20/57 (35%) | Replicated (pre-registered threshold 20%) |
| Ignored False Rejections (code unchanged, AST hash) | 83% (15 of 18 unchanged) | 1/20 (5%) | Not replicated: the executor rewrote 19 of 20 correct programs |
| Damage on False Rejection (changed and broken) | 100% of changes broke the answer (3/3) | 4/19 (21%) of changed programs broke | Partly replicated: revisions still damage some correct programs, at a lower rate |
| Public Sample Tests Reviewer (C4) | N/A (code-only condition) | 50.4% accuracy; FRR 0/57 (0%); false accepts 19/68 | New in the code study |

## 6. Pre-Registered Hypotheses

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **H1** | supported | C1 FRR 20/57 = 35.1% |
| **H2** | not supported | unchanged after C1 false rejection: AST 1/20, text 0/20 |
| **H3** | not supported | C0 45.6% vs C2 50.4%, b=2 c=8, p=0.1094, Holm p=0.4375 |
| **H4** | supported | C4 FRR 0/57 = 0.0%; false accepts 19/68 |

## 7. Exploratory E2: Temporal Sensitivity (boundary 2025-01-31)

Exploratory and non-confirmatory. Post-cutoff hypothesis checks are low-n sensitivity checks.

| Stratum | Problems | A0 Correct | A0 Accuracy [Wilson 95% CI] | C1 FRR [Wilson 95% CI] |
|---|---|---|---|---|
| Pre (<= 2025-01-31) | 78 | 34 | 43.6% [0.331, 0.546] | 13/34 = 38.2% [0.239, 0.550] |
| Post (> 2025-01-31) | 47 | 23 | 48.9% [0.353, 0.628] | 7/23 = 30.4% [0.156, 0.509] |

- Two-sided exact Fisher on A0 accuracy, pre vs post: p = 0.5830

| Hypothesis (post-cutoff only) | Verdict | Evidence |
|---|---|---|
| H1 | supported | C1 FRR 7/23 = 30.4% |
| H2 | inconclusive | unchanged after C1 false rejection: AST 0/7, text 0/7 (low-n); fewer than 10 false rejections |
| H3 | not supported | C0 48.9% vs C2 51.1%, b=1 c=2, p=1.0000, Holm p=1.0000 |
| H4 | supported | C4 FRR 0/23 = 0.0%; false accepts 9/24 |

