# E3: Execution Stability Audit (exploratory)

Pre-registered in `ANALYSIS_PLAN_CODE.md`, Section 7. The audit re-graded every distinct program in `run_code_data/programs.jsonl` in a second Docker pass (`scripts/audit_regrade.py`), using the same sandbox limits and early-exit rules and bypassing the grade cache. The second fingerprints are compared with every first-pass fingerprint recorded for the same program. Primary results use the first pass only, as pre-registered.

- **Data**: `run_code_data/audit_regrade.json` (second pass) and `run_code_data/audit_regrade_recheck.json` (an extra third run of the disagreeing programs, described below).
- **When**: from 21:08 UTC on 24 November 2026 to about 15:50 UTC on 25 November 2026. The laptop was asleep from 21:54 UTC on 24 November to about 14:49 UTC on 25 November, during program 85.

## Result

**3 of 328 distinct programs (0.9%) disagreed, which is 21 of 1,250 first-pass fingerprint records (1.7%).** All three were `PASS` on the first pass and `TIMEOUT` on the second.

| Program | Role | Records | Slowest test: 1st pass | 2nd pass | 3rd pass (recheck) |
|---|---|---|---|---|---|
| `abc374_f` `3b4359de4f9e5f52` | A0, all of C0–C4 | 10 | 4.965 s, PASS | 7.031 s, TIMEOUT | 5.178 s, PASS |
| `abc390_d` `4d893e2ea802156d` | A0, all of C0–C4 | 10 | 4.911 s, PASS | 6.868 s, TIMEOUT | 5.831 s, PASS |
| `abc384_e` `8b59b32c3fcfab3a` | C2 final (revision) | 1 | 1.322 s, PASS | TIMEOUT, see below | 0.810 s, PASS |

## Interpretation

- **Borderline timeouts (`abc374_f`, `abc390_d`)**: both programs have a slowest test that took between 4.9 s and 7.0 s across the three runs, against the 6.0 s per-test limit. They pass or time out depending on timing noise. Calibration had already observed run-to-run wall-time variation of up to about 2x (`ANALYSIS_PLAN_CODE.md`, Section 2.3), and these two programs are that variation landing on the limit. Both are A0 programs shared by all five conditions, so the first-pass outcome counts the same way in every condition and does not favour any one of them.
- **Standby artifact (`abc384_e` `8b59b32c...`)**: the second-pass grade of this program took 60,945 s, because the laptop entered sleep while it was being graded. Its `TIMEOUT` is an infrastructure artifact, not program behaviour. The program passed in about 1 s on both other runs.

No disagreement involved wrong output, a crash, or a change between `PASS` and `FAIL`. The 325 other programs, including every `FAIL`, `CRASH` and `NO_CODE` result, gave the same fingerprint on both passes. The third-pass recheck of the three disagreeing programs was not pre-registered. It is reported only to separate the standby artifact from genuine borderline timing.
