"""Omni-MATH task loader.

Loads olympiad problems from the KbsdJames/Omni-MATH HuggingFace dataset,
filters to a difficulty band AND to problems with clean numeric answers (so
grading stays exact and automated, no LLM judge), EXCLUDES malformed problem
types, and maps them into the project's Task objects.

Why numeric-only: Omni-MATH answers include LaTeX expressions, free variables,
and yes/no answers that cannot be graded programmatically without an LLM judge.
Restricting to plain integer/decimal answers keeps grading clean. Stated
limitation: skews toward number-theory/combinatorics answers.

Why the junk filter: the smoke run surfaced unanswerable problems that pollute
every condition -- e.g. problems that reference another problem not included
("as in problem 15"), guts-round chains, and open-ended "estimate" problems
with no exact answer. These are excluded (see _JUNK_PATTERNS).

Difficulty band: 1-10 scale. 5-6 is where the models fail SOME but not ALL,
the regime the reviewer study needs.
"""

from __future__ import annotations

import re
from typing import List

from core import Task, grade_numeric, NUMERIC_ANSWER_INSTRUCTION

# A "clean" answer: an optionally-signed integer or decimal, nothing else.
_CLEAN_NUMERIC = re.compile(r"-?[0-9]+(\.[0-9]+)?$")

# Problem-text patterns that make a task unanswerable in isolation or non-exact.
# Matched case-insensitively against the problem text.
_JUNK_PATTERNS = [
    r"\bas in (?:problem|question)\b",          # references another problem
    r"\bprevious (?:problem|question|part)\b",
    r"\bproblem\s+\d+\b",                        # "problem 15", guts-round chains
    r"\bpreceding (?:problem|question)\b",
    r"\bestimate\b",                             # open-ended estimation, no exact answer
    r"\bguts round\b",
    r"\bhow close\b",                            # estimation-style scoring
    r"\byour answer to\b",                        # references another answer
]
_JUNK_RE = re.compile("|".join(_JUNK_PATTERNS), re.IGNORECASE)


def _is_junk(problem_text: str) -> bool:
    return bool(_JUNK_RE.search(problem_text))


def load_omnimath_tasks(
    n: int | None = None,
    min_difficulty: float = 5.0,
    max_difficulty: float = 6.0,
    seed: int = 0,
    report: bool = True,
) -> List[Task]:
    """Load Omni-MATH problems filtered to [min_difficulty, max_difficulty]
    with clean numeric answers and no malformed problem types. Returns up to n
    Task objects (all if n is None), shuffled deterministically by seed.

    Requires: pip install datasets
    """
    from datasets import load_dataset
    import random

    ds = load_dataset("KbsdJames/Omni-MATH", split="test")

    kept, n_diff, n_nonnumeric, n_junk = [], 0, 0, 0
    for row in ds:
        diff = float(row["difficulty"])
        if not (min_difficulty <= diff <= max_difficulty):
            n_diff += 1
            continue
        ans = row["answer"].strip()
        if not _CLEAN_NUMERIC.fullmatch(ans):
            n_nonnumeric += 1
            continue
        if _is_junk(row["problem"]):
            n_junk += 1
            continue
        kept.append(row)

    rng = random.Random(seed)
    rng.shuffle(kept)
    if n is not None:
        kept = kept[:n]

    if report:
        print(f"[omnimath] band {min_difficulty}-{max_difficulty}: "
              f"{len(kept)} kept "
              f"(excluded: {n_junk} junk, {n_nonnumeric} non-numeric, "
              f"{n_diff} out-of-band). Returning {len(kept)}.")

    tasks: List[Task] = []
    for i, row in enumerate(kept):
        prompt = (
            f"Solve the following competition mathematics problem.\n\n"
            f"{row['problem']}"
            + NUMERIC_ANSWER_INSTRUCTION
        )
        tasks.append(Task(
            task_id=f"omni_{i:04d}",
            family="omnimath",
            prompt=prompt,
            ground_truth=float(row["answer"].strip()),
            meta={
                "difficulty": float(row["difficulty"]),
                "domain": row.get("domain", ""),
                "source": row.get("source", ""),
                "round_to": 6,
            },
        ))
    return tasks


def grade_omnimath(task: Task, response: str) -> dict:
    return grade_numeric(task, response)