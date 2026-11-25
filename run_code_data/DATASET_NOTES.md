# LiveCodeBench-Lite Dataset Notes

**Source**: `livecodebench/code_generation_lite` (HuggingFace)
**Inspection Date**: 2026-10-24 03:29:23
**Total Problems**: 1055
**Contest Date Range**: `2023-05-07T00:00:00` to `2025-04-06T00:00:00`

## 1. Pinned Files in Release

| File | Rows | Size (MB) | Contest Date Min | Contest Date Max |
|---|---|---|---|---|
| `test.jsonl` | 400 | 1194.6 MB | 2023-05-07 | 2024-03-02 |
| `test2.jsonl` | 111 | 680.3 MB | 2024-03-09 | 2024-05-25 |
| `test3.jsonl` | 101 | 594.5 MB | 2024-06-01 | 2024-08-10 |
| `test4.jsonl` | 101 | 1148.8 MB | 2023-08-26 | 2024-10-05 |
| `test5.jsonl` | 167 | 531.9 MB | 2024-09-22 | 2025-01-04 |
| `test6.jsonl` | 175 | 128.1 MB | 2025-01-04 | 2025-04-06 |

## 2. Schema and Expected Fields

| Field | Type | Description / Sample |
|---|---|---|
| `question_title` | `str` | A. Short Sort |
| `question_content` | `str` | There are three cards with letters $\texttt{a}$, $\texttt{b}$, $\texttt{c}$ placed in a row in some ... |
| `platform` | `str` | codeforces |
| `question_id` | `str` | 1873_A |
| `contest_id` | `str` | 1873 |
| `contest_date` | `str` | 2023-08-21T00:00:00 |
| `starter_code` | `str` |  |
| `difficulty` | `str` | easy |
| `public_test_cases` | `str` | [{"input": "6\nabc\nacb\nbac\nbca\ncab\ncba\n", "output": "YES\nYES\nYES\nNO\nNO\nYES\n", "testtype"... |
| `private_test_cases` | `str` | eJxrYJmaz8gABhEZQEZ0tVJmXkFpiZKVgpJhTF5iUnJMnpKOglJ+aQlUNNI1GCJUklpcUlJZkAoSLC5JycxTqtVRQNJuDNWOILAY... |
| `metadata` | `str` | {} |

## 3. Platform Distribution

| Platform | Count | Primary Test Type |
|---|---|---|
| atcoder | 602 | stdin |
| codeforces | 9 | stdin |
| leetcode | 444 | functional (class Solution) |

## 4. Difficulty Distribution

| Difficulty | Count | Percentage |
|---|---|---|
| easy | 322 | 30.5% |
| hard | 350 | 33.2% |
| medium | 383 | 36.3% |

## 5. Candidate Problems per Month (Stdin-Only, Medium & Hard, Non-Special-Judge)

| Month | Total Problems | Kept Candidates | Special-Judge Dropped |
|---|---|---|---|
| 2023-05 | 34 | **12** | 0 |
| 2023-06 | 39 | **12** | 0 |
| 2023-07 | 54 | **17** | 1 |
| 2023-08 | 36 | **6** | 4 |
| 2023-09 | 44 | **16** | 1 |
| 2023-10 | 44 | **16** | 2 |
| 2023-11 | 35 | **11** | 2 |
| 2023-12 | 40 | **11** | 1 |
| 2024-01 | 36 | **11** | 0 |
| 2024-02 | 32 | **13** | 0 |
| 2024-03 | 45 | **11** | 1 |
| 2024-04 | 37 | **12** | 1 |
| 2024-05 | 36 | **12** | 0 |
| 2024-06 | 46 | **15** | 0 |
| 2024-07 | 43 | **15** | 1 |
| 2024-08 | 66 | **31** | 3 |
| 2024-09 | 47 | **20** | 2 |
| 2024-10 | 53 | **28** | 2 |
| 2024-11 | 57 | **27** | 0 |
| 2024-12 | 49 | **20** | 1 |
| 2025-01 | 51 | **20** | 2 |
| 2025-02 | 51 | **22** | 1 |
| 2025-03 | 68 | **30** | 3 |
| 2025-04 | 12 | **8** | 0 |
| **Total** | **1055** | **396** | **28** |

## 6. Sample Problem Test Case Decoding (5 Problems)

### Problem 1: `1883_B` (codeforces, medium, 2023-09-22)
- **Public Tests (1)**: type=`stdin`
  - Sample Input: `'14\n1 0\na\n2 0\nab\n2 1\nba\n3 1\nabb\n3 2\nabc\n6 2\nbacacd\n6 2\nfagbza\n6 2\nzwaafa\n7 2\ntaagaak\n14 3\nttrraakkttoorr\n5 3\ndebdb\n5 4\nec'`
  - Sample Expected: `'YES\nNO\nYES\nYES\nYES\nYES\nNO\nNO\nYES\nYES\nYES\nYES\nNO\nYES\n'`
- **Private Hidden Tests (3)**: type=`stdin`
  - Sample Input: `'5\n10 3\naaabbbcccd\n10 1\naaabbccddd\n10 0\naaabbccddd\n10 9\nabcdefghij\n10 2\naabbccddee\n'`
  - Sample Expected: `'YES\nYES\nNO\nYES\nYES'`

### Problem 2: `abc326_c` (atcoder, medium, 2023-10-28)
- **Public Tests (3)**: type=`stdin`
  - Sample Input: `'8 6\n2 3 5 7 11 13 17 19\n'`
  - Sample Expected: `'4\n'`
- **Private Hidden Tests (12)**: type=`stdin`
  - Sample Input: `'1 1\n1\n'`
  - Sample Expected: `'1\n'`

### Problem 3: `abc354_c` (atcoder, medium, 2024-05-18)
- **Public Tests (3)**: type=`stdin`
  - Sample Input: `'3\n2 4\n1 1\n3 2\n'`
  - Sample Expected: `'2\n2 3\n'`
- **Private Hidden Tests (14)**: type=`stdin`
  - Sample Input: `'200000\n2194 999997802\n4140 999994470\n4558 999987430\n14295 999985126\n17774 999976965\n18961 999961693\n22578 999952992\n3344'`
  - Sample Expected: `'1\n200000\n'`

### Problem 4: `arc182_c` (atcoder, hard, 2024-08-11)
- **Public Tests (3)**: type=`stdin`
  - Sample Input: `'1 7\n'`
  - Sample Expected: `'16\n'`
- **Private Hidden Tests (25)**: type=`stdin`
  - Sample Input: `'10 11\n'`
  - Sample Expected: `'797173552\n'`

### Problem 5: `arc189_d` (atcoder, hard, 2024-12-08)
- **Public Tests (2)**: type=`stdin`
  - Sample Input: `'6\n4 13 2 3 2 6'`
  - Sample Expected: `'4 30 2 13 2 13'`
- **Private Hidden Tests (40)**: type=`stdin`
  - Sample Input: `'1772\n329091090 696407456 230198842 529706435 495965766 185322586 677456060 691258097 961598048 716237508 626673105 63024'`
  - Sample Expected: `'329091090 875300735583 230198842 875300735583 875300735583 185322586 875300735583 875300735583 875300735583 875300735583'`

## 7. Differences and Nuances Compared to Section 4 of Implementation Plan

1. **Private Test Encoding**: Confirmed `base64 -> zlib -> pickle -> JSON`. Decoding verified across all problems.
2. **Public Test Encoding**: Confirmed direct JSON string of list of test dicts with `input`, `output`, `testtype`.
3. **Dataset Partitioning**: The HuggingFace dataset contains 6 disjoint JSONL files (`test.jsonl` through `test6.jsonl`). `test5.jsonl` and `test6.jsonl` cover September 2024 through April 2025, which postdate typical training cutoffs.
4. **Platform Filtering**: AtCoder and Codeforces use standard stdin/stdout (`testtype == 'stdin'`). LeetCode problems use functional signatures with `class Solution:` and must be filtered out as specified.
5. **Special Judge Filtering**: In addition to `is_special_judge`, keyword filters (`'any of them'`, `'print any'`, `'if there are multiple'`, `'absolute or relative error'`) successfully identify ambiguous-output problems.
