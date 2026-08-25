# Analysis Plan, Amendment 1

**Status.** This document amends `ANALYSIS_PLAN.md`, which is retained unchanged
as a record of the original design. It was written before the reported
experiment was run. The original plan describes a study that was abandoned; the
reasons are set out in section 1 below, together with the calibration data that
motivated the change.

---

## 1. Why the study changed

The original plan tested whether cross-model error diversity predicts
multi-agent benefit. Its central metric, `D_across`, is the fraction of
model-pairs that produce *different* wrong answers on a task. Computing it
requires that the models actually produce wrong answers, and that at least two
of them are wrong on the same task often enough for the fraction to be stable.

Calibration probes falsified that premise for the models available. Each probe
ran 10 tasks against all three models at two samples per task:

| Task family | Difficulty | Gemini 3.1 Flash-Lite | gpt-oss-20b | Llama-3.1-8B |
|---|---|---|---|---|
| Generated math word problems | 2 | 20/20 | 20/20 | 11/20 |
| Generated math word problems | 3 | 20/20 | 19/20 | 9/20 |
| Generated code debugging | 2 | 20/20 | 20/20 | 20/20 |
| Generated constraint puzzles | 2 | 20/20 | 20/20 | 4/20 |
| Generated constraint puzzles | 5 | 20/20 | 6/20 | 2/20 |
| Omni-MATH, numeric subset | 5.0-6.0 | 12/20 | see note | 0/20 |

Note: gpt-oss returned empty responses on the first Omni-MATH probe because its
reasoning trace exhausted the output token budget before it emitted an answer.
This was a configuration fault, not a capability result, and was corrected
before the reported experiment.

Three consequences followed.

**The stronger models did not err on generated tasks.** Across four generated
families, including the hardest configuration the generators could produce,
Gemini answered every item correctly and gpt-oss answered all or nearly all
correctly. `D_across` requires both members of a model-pair to be wrong on the
same task; with two of three models effectively never wrong, such pairs almost
never occurred.

**The errors that did occur were the wrong kind.** On generated math, the weak
model's errors were rounding differences (for example 512.13 and 511.99 against
a reference of 512.15) that varied between samples of the same model on the same
task. The original hypothesis concerns models being confidently wrong in the
*same* way. Unstable rounding noise is the opposite, and would have inflated
`D_across` for reasons unrelated to the hypothesis.

**The capability spread was too wide on tasks that were hard enough.** Moving to
olympiad problems produced real errors from the strong model, but the weak model
then solved none of them. There was no difficulty at which all three models
erred sometimes without one of them erring always.

The original design was therefore not measurable with this model set on any task
type available. Rather than continue generating task families in the hope of
finding a band where all three models err comparably, the study was re-scoped to
treat the capability spread itself as the object of study.

## 2. Revised research question

In an Execute-Review-Revise pipeline with a fixed executor, how does replacing
the reviewer with progressively weaker models affect final accuracy, the outcome
of each individual rejection, and token cost?

Sub-questions:

- Does a review stage improve final accuracy over no review, and by how much per
  reviewer tier?
- Through which mechanism does each reviewer act or fail: detecting real errors,
  falsely rejecting correct answers, repairing on rejection, or damaging a
  correct answer on rejection?
- Can a reviewer that cannot solve the problems itself add value?
- What is the token cost per additional correct answer for each tier?

## 3. What is retained from the original plan

The pipeline mechanics, the paired design (the executor solves each problem once
and the identical solution is reviewed under every condition), exact-match
numeric grading with no model acting as judge, the harness and its caching, the
three models, temperature 0.7, and the commitment in the original plan to report
null results honestly.

## 4. What is replaced

`D_across` and its associated hypotheses are withdrawn. The set-A/set-B sample
split that existed to prevent circularity between predictor and outcome is no
longer needed, because the reviewer condition is now an experimental
manipulation rather than a measured property of the tasks.

The task source changes from programmatic generation to Omni-MATH, difficulty
band 5.0 to 6.0, restricted to problems with plain numeric answers and excluding
malformed problem types. N = 100.

## 5. Conditions

Executor fixed: Gemini 3.1 Flash-Lite. C0 no review; C1 self-review, same model
as executor; C2 cross-family mid-tier, gpt-oss-20b; C3 weak, Llama-3.1-8B.

## 6. Metrics

Final accuracy and change against C0. Recall, defined as the probability of
rejection given the initial answer was wrong. False-rejection rate, the
probability of rejection given the initial answer was correct. Every rejection
classified as repair, damage, no-change split by whether the initial answer was
correct, or miscorrection. Rate of unparseable reviews. Exact token usage per
task, and additional tokens per additional correct answer against C0.

## 7. Statistics, fixed in advance

Accuracy comparisons: exact McNemar on paired per-problem outcomes. Proportions
computed over rejections: Fisher's exact test. Conditions with fewer than five
rejections are described but not tested.

## 8. Pre-committed outcomes

Any of the following may occur and will be reported as found: no reviewer tier
improves on the baseline; a weaker reviewer degrades accuracy; a weaker reviewer
changes nothing at all. If a reviewer never rejects, its damage rate is
undefined and will be reported as such rather than as zero.

## 9. Pre-analysis adjustment, recorded here for completeness

After this amendment was written and before the reported analysis, one change
was made to the experimental setup. The mid-tier reviewer's reviews were being
truncated by its provider token ceiling on 29 percent of problems, returning
empty text that the pipeline counted as an accept. Because the ceiling cannot be
raised, reviewer *input* was reduced instead: the executor-facing answer-format
instructions were stripped from the problem text shown to the reviewer, and the
solution was truncated to its final 2,000 characters with an explicit elision
marker. This reduces reviewer input by approximately 66 percent on long
problems. The change was applied identically to all reviewer conditions and
every review was regenerated from scratch. The resulting rate of unparseable
reviews is reported per condition in the paper.

## 10. Limitations acknowledged in advance

Single executor. N = 100. One difficulty band. One benchmark. Numeric-answer
subset only, which favours number theory and combinatorics. One revision round.
One reviewer prompt template. Free-tier provider limits constrain both model
choice and sample size. Results describe this configuration and are not a
general claim about verification stages.
