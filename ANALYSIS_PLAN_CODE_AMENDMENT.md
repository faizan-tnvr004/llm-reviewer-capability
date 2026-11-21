# Analysis Plan (Code Extension), Amendment 1

**Date**: 20 November 2026
**Amends**: `ANALYSIS_PLAN_CODE.md` as tagged `prereg-code-v1` (commit `4f7999d`)
**Made**: during the main run, after 143 of 625 outcome rows had been logged

This amendment records one operational incident in the main run and one change to how token usage is written to disk. It changes no prompt, model, parameter, timeout, problem, grading rule, scoring rule, hypothesis or analysis. No logged outcome was edited, removed or regenerated.

## 1. Incident: a Groq request that never returned

- **What happened**: The C2 (`openai/gpt-oss-20b`) review request for problem `abc374_e`, sent at 08:45:10 UTC on 20 November 2026, never completed. Groq kept the connection open, so the Groq SDK's timeout, which counts the time between received bytes (60 s), never fired, and the runner waited without logging anything further.
- **Attempts to free it**: a 10-second Wi-Fi outage (the connection survived), then a Ctrl+C sent to the runner (not delivered, because the runner was a background process on Windows).
- **Resolution**: the runner was killed at 09:37:10 UTC and restarted at 09:37:30 UTC with the same command at the tagged commit `4f7999d`. The second entry in `run_code_data/run_manifest.json` records the restart (`git_describe = prereg-code-v1`).
- **Effect on outcomes**: none. Outcome rows are written last for each (problem, condition), so the 137 rows logged before the kill were complete and unchanged, and none were rescored. The restart resumed at `abc374_e` C2. That condition's review was requested again because the first request never returned a response, so nothing for it had been cached or logged.
- **Classification**: an operational interruption, like the quota and sandbox stops in Section 4.3, items 3 and 13. Unlike those, it needed a manual kill, so it is reported here.

## 2. Token records lost for the first 28 problems

`tokens.jsonl` was written only when the runner exited, so the kill discarded the provider-reported token counts for every call made before 09:37:10 UTC. These are the calls for the 137 rows of the first 28 problems in processing order: all five conditions of the first 27 problems, plus the A0, C0 and C1 calls of `abc374_e`:

arc185_d, abc395_c, abc396_c, abc395_f, abc389_f, abc377_d, abc387_f, arc189_c, abc396_f, abc391_f, arc190_d, arc186_d, abc379_g, arc192_a, abc380_d, abc388_d, abc390_d, abc399_e, abc394_f, abc388_c, abc386_c, abc374_f, abc388_g, abc391_d, arc187_b, abc395_e, abc375_f, abc374_e

**Reporting rule**: the token-cost metric (Section 6, item 4) is reported in two ways:
- primarily, on the problems with exact provider usage;
- separately, on all 125 problems, with these 28 problems' tokens re-estimated from the cached prompts and responses using the runner's own estimator. Those records are marked `exact: false`.

The two are never pooled without saying so.

## 3. Change: token records written as each call finishes

From the next runner start onward, `run_experiment_code.py` gives the `TokenMeter` a `StreamedTokenRecords` list. It appends each token record to `run_code_data/tokens.jsonl` and flushes it to disk the moment the call finishes, instead of writing all records when the runner exits. A killed run, or one that loses power, now keeps the usage of every finished call.

The record format, the usage source and the estimator fallback are unchanged. The change affects only when records reach disk; it touches no code that produces or scores an outcome.

The run in progress when this amendment was written keeps running the earlier code from memory. Its records are written when it exits normally. The first runner start on the amended commit appears as a new entry in `run_manifest.json`.

## 4. Further hangs (added 21 November 2026)

Three more requests never returned on 21 November 2026, all to Gemini (`gemini-3.1-flash-lite`). The Gemini SDK sets no overall request timeout by default, so the runner waited on an open connection with no CPU activity. Under the owner's standing instruction to restart on any hang, each was resolved by killing the runner and re-running the same command at commit `91a31d7` (`git_describe = prereg-code-v1-1-g91a31d7`). Each restart added an entry to `run_manifest.json`.

| # | Request that never returned | Sent (UTC) | Runner killed (UTC) | Restarted (UTC) | Manifest entry |
|---|---|---|---|---|---|
| 2 | `abc381_d` C1 revision | 09:25:58 | 10:08:16 | 10:08:25 | 4 |
| 3 | `abc390_f` C2 revision (after the C2 review completed) | about 11:07 | 11:27:11 | 11:27:18 | 5 |
| 4 | `arc189_d` C1 review | about 12:01:46 | 12:17:30 | 12:17:36 | 6 |

The effect on outcomes is the same as for the first incident: none. No logged row was edited or rescored. After each restart the run resumed at the interrupted (problem, condition), and only the request that had never returned was sent again. Token records were already being written per call (Section 3), so no token usage was lost in these three incidents.

These four hangs (one on Groq, three on Gemini) share one cause: neither SDK puts a total time limit on a request. Fixing that would change how the runner handles provider calls mid-run, so it was not done. The weakness is reported as a limitation of this setup, and the restarts are reported as above.
