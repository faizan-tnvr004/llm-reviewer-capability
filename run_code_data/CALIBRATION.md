# LiveCodeBench Calibration Log

**Target Executor Accuracy**: 40% – 60% on Gemini 3.1 Flash-Lite (`gemini-3.1-flash-lite`).

---

## 1. Calibration Batches

| Batch | Window | Difficulties | Seed | Evaluated | Correct | Accuracy | Fingerprints (PASS / FAIL / TIMEOUT / CRASH) | Status |
|---|---|---|---|---|---|---|---|---|
| **Batch 1** | `2024-10-01` to end (`2025-04-06`) | `medium` + `hard` | `0` | 20 | 8 | **40.0%** | PASS: 8, FAIL: 10, TIMEOUT: 2, CRASH: 0 | **TARGET MET (40.0%)** |

---

## 2. Batch 1 Outcome Breakdown

- **Date Window**: `2024-10-01` to `2025-04-06` (the most recent contest dates in LiveCodeBench-Lite, strictly after Gemini's training cutoff).
- **Difficulties**: `medium` (40 candidates) and `hard` (87 candidates), total 127 candidates in this window.
- **Accuracy**: 8 solved out of 20 = **40.0%** (target: 40% – 60%).
- **Fingerprint distribution**:
  - `PASS`: 8 (40.0%)
  - `FAIL`: 10 (50.0%)
  - `TIMEOUT`: 2 (10.0%)
  - `CRASH`: 0 (0.0%)

### Evaluated Problems in Batch 1 (Permanently Excluded from Main Set)
The following 20 problem IDs were calibrated and will **never** be reused in the main study ($N=100$):
1. `arc186_e` (hard) — TIMEOUT
2. `arc189_a` (medium) — FAIL:35/40
3. `abc390_c` (medium) — PASS
4. `arc196_d` (hard) — FAIL:32/40
5. `abc383_e` (hard) — FAIL:13/40
6. `arc195_a` (medium) — PASS
7. `abc382_d` (hard) — PASS
8. `abc376_b` (medium) — PASS
9. `abc377_f` (hard) — FAIL:38/40
10. `abc376_d` (medium) — PASS
11. `arc194_b` (hard) — FAIL:25/40
12. `abc394_c` (medium) — FAIL:1/40
13. `abc394_g` (hard) — PASS
14. `arc189_b` (hard) — FAIL:38/40
15. `arc195_c` (hard) — FAIL:4/4
16. `abc385_c` (medium) — PASS
17. `abc381_e` (hard) — FAIL:12/40
18. `abc378_g` (hard) — TIMEOUT
19. `abc386_e` (hard) — PASS
20. `abc392_g` (hard) — FAIL:39/40

---

## 3. Prompt Length Measurements (Candidate Pool)

Excluding the 20 calibration problems, the candidate pool in the `2024-10-01` window contains **135 problems** (more than sufficient for $N=100$):
- **Prompt Characters**: Min = 745, Median = 1,871, Mean = 2,006, Max = 4,135
- **Estimated Prompt Tokens** (`chars / 4`): Min = 186, Median = 468, Mean = 502, Max = 1,034
- **Groq TPM Safety**: With mean prompt length ~502 tokens (max 1,034 tokens), reviewer prompts remain comfortably below Groq's 8,000 tokens-per-minute ceiling when throttled to $\ge 40$ seconds.

---

## 4. Proposed Pre-Registration Parameters

- **Problem Window**: `date_from = "2024-10-01"`, `date_to = None`
- **Difficulties**: `("medium", "hard")`
- **Sample Size**: $N = 100$
- **Sampling Seed**: `42` (deterministically drawn from the 135 remaining candidate problems, strictly excluding the 20 Batch 1 problem IDs).
