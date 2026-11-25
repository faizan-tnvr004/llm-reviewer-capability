# E1: Critique specificity vs adoption (exploratory)

The procedure is in `ANALYSIS_PLAN_CODE_AMENDMENT.md`, Section 6; the outcome is in Section 6a.

- **Labels**: a blind LLM agent labelled all 154 C1–C3 critiques (`agent_labels.json`, `critiques_labeled_agent.csv`). It rated 147 `specific` and 7 `vague`.
- **Human check**: on a pre-drawn random 30, the owner's blind labels (`human_sample_30.csv`) agreed with the agent's 19/30 times (63.3%), **Cohen's kappa = 0.046**. The agent labels fail validation.
- **Adoption, independent of labels**: the executor changed its program (AST hash) after 153 of 154 rejections. The agent-label breakdown, which is not a valid specificity measure, is in `E1_RESULTS_agent.txt`.
- **Human labels, 30 items, descriptive only**:

  | | Items | Adopted |
  |---|---|---|
  | `specific` | 19 | 19 |
  | `vague` | 11 | 11 |

  Vague critiques by condition: C1 2/15, C2 7/13, C3 2/2.

**Conclusion**: specificity cannot explain adoption, because the executor adopts essentially every critique, whoever labels it. That repeats the H2 finding (19 of 20 falsely rejected correct programs were rewritten). An LLM labeller was not a reliable substitute for human specificity judgements here.
