"""Core task structure and grading logic shared by all task families.

Every task, regardless of family, carries:
  - a unique id
  - the prompt shown to the model
  - a ground-truth answer
  - a `fingerprint(response)` function that maps a model response to a
    canonical representation of *what answer it gave* (used both for grading
    and for the "same wrong answer" comparison in D_across).

Models are instructed to end their response with a line:
    ANSWER: <value>
The extractor is deliberately forgiving (models don't always comply exactly),
but grading itself is strict: exact numeric match within a small tolerance.
"""

from __future__ import annotations

import re, math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# Numeric tolerance for "exact" match: covers float noise, not wrong answers.
NUMERIC_TOL = 1e-6
# Money answers are compared after rounding to 2 decimal places.
MONEY_DECIMALS = 2


@dataclass
class Task:
    task_id: str
    family: str                # "math" | "code" | "constraint" | "data"
    prompt: str
    ground_truth: Any
    meta: dict = field(default_factory=dict)  # template name, params, difficulty knobs


# ---------------------------------------------------------------------------
# Answer extraction (shared by math + data families: final-number fingerprint)
# ---------------------------------------------------------------------------
#
# Real model output taught us these cases (from the live-run checkpoint):
#   - "ANSWER: 31/5"          -> a fraction; must evaluate to 6.2, not 31
#   - LaTeX-wrapped answers    -> "\[ ... \]", "\(...\)"; strip the markup
#   - display-rounded trailing numbers when no ANSWER line is present
# We therefore (a) capture the whole answer token after ANSWER:, including a
# possible /denominator, and (b) evaluate fractions.

# Answer line: capture everything after ANSWER: up to end of line, so we can
# then parse a fraction, a plain number, currency, etc.
_ANSWER_LINE = re.compile(r"ANSWER\s*[:=]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
# A fraction like 31/5 or -7/2 (integers only, which is how models write them).
_FRACTION = re.compile(r"^\s*(-?\d+)\s*/\s*(-?\d+)\s*$")
# A plain number (optionally with $ , and decimals). Used for answer-line
# parsing and as the last-number fallback.
_NUMBER = re.compile(r"-?\$?\s*[\d,]+(?:\.\d+)?")


def _parse_value(token: str) -> Optional[float]:
    """Parse one answer token that may be a fraction, currency, or plain number.
    Returns the float, or None if the token is present but unparseable
    (e.g. divide-by-zero) — the caller must NOT then scavenge stray numbers."""
    token = token.strip().strip("*").strip()           # drop markdown bold, spaces
    token = token.replace("\\", "").replace("$", "")    # drop LaTeX backslashes, $
    frac = _FRACTION.match(token)
    if frac:
        num, den = int(frac.group(1)), int(frac.group(2))
        if den == 0:
            return None
        return num / den
    m = _NUMBER.search(token)
    if m:
        return _to_float(m.group(0))
    return None


def _answer_line_is_fraction_like(token: str) -> bool:
    """True if the token looks like it was meant as a fraction (contains '/')."""
    return "/" in token


def extract_final_number(response: str) -> Optional[float]:
    """Pull the model's final numeric answer out of its response text.

    Priority:
      1. An explicit 'ANSWER: <value>' line (what the prompt asks for);
         value may be a fraction (31/5), currency ($12.50), or plain number.
         If several ANSWER lines exist (self-correction), the LAST wins.
      2. Fallback: the last number appearing in the response (LaTeX stripped).
    Returns None if no value can be found (counts as incorrect, fingerprint
    'NO_ANSWER').
    """
    lines = _ANSWER_LINE.findall(response)
    if lines:
        token = lines[-1]
        val = _parse_value(token)
        if val is not None:
            return val
        # ANSWER line was present but unparseable. If it looked like a fraction
        # (e.g. '5/0'), do NOT fall through and grab stray numbers — that would
        # invent an answer. Treat it as no valid answer.
        if _answer_line_is_fraction_like(token):
            return None
    # Fallback: strip LaTeX delimiters, take the last number in the text.
    cleaned = response.replace("\\[", " ").replace("\\]", " ")
    cleaned = cleaned.replace("\\(", " ").replace("\\)", " ").replace("\\", " ")
    nums = _NUMBER.findall(cleaned)
    if nums:
        return _to_float(nums[-1])
    return None


def _to_float(s: str) -> Optional[float]:
    try:
        v = float(s.replace(",", "").replace("$", "").strip())
    except ValueError:
        return None
    return v if math.isfinite(v) else None


# ---------------------------------------------------------------------------
# Grading + fingerprinting for numeric-answer tasks
# ---------------------------------------------------------------------------

def grade_numeric(task: Task, response: str) -> dict:
    """Grade a numeric-answer task (math / data families).

    Returns a dict with:
      correct      : bool
      fingerprint  : canonical string of the answer given (or 'NO_ANSWER') —
                     this is what D_across compares between two wrong attempts
      extracted    : the raw extracted float (or None)
    """
    extracted = extract_final_number(response)
    truth = float(task.ground_truth)

    if extracted is None:
        return {"correct": False, "fingerprint": "NO_ANSWER", "extracted": None}

    # Canonicalise: money-style answers round to 2dp; both sides same treatment.
    decimals = task.meta.get("round_to", MONEY_DECIMALS)
    canon_ans = round(extracted, decimals)
    canon_truth = round(truth, decimals)

    correct = abs(canon_ans - canon_truth) <= NUMERIC_TOL
    return {
        "correct": correct,
        "fingerprint": f"{canon_ans:.{decimals}f}",
        "extracted": extracted,
    }


# Instruction appended to every numeric-answer prompt.
NUMERIC_ANSWER_INSTRUCTION = (
    "\n\nWork through the problem step by step, then give your final answer on "
    "the last line in exactly this format:\nANSWER: <number>\n"
    "Give the number as a plain decimal (not a fraction, not in words, no units, "
    "no LaTeX). For example: ANSWER: 6.20"
)