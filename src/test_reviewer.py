"""C4 public-test reviewer — a ToolReviewer, not an LLM.

The PublicTestReviewer runs the executor's program against the problem's
sample test cases (the ones visible in the problem statement).  It does NOT
use an LLM and therefore costs zero tokens.

On failure it emits a deterministic VERDICT: REJECT + a critique built from
the C4 templates in prompts_code.py.  On success it emits VERDICT: ACCEPT.

The ToolReviewer protocol:
  - Attribute: model_id (str) — recorded in the log.
  - Method:    review(task, solution_text) -> str — the review text.

The pipeline detects isinstance(reviewer, ToolReviewer) and calls
reviewer.review() directly instead of calling an LLM.
"""

from __future__ import annotations

from core import Task
from tasks.livecodebench_tasks import extract_program, public_check_with_input
from prompts_code import (
    C4_FAIL_TEMPLATE,
    C4_CRASH_TEMPLATE,
    C4_TIMEOUT_TEMPLATE,
    C4_NO_PROGRAM_TEMPLATE,
)

# Maximum length for any single field in a C4 critique
_FIELD_CAP = 1000


def _cap(text: str, limit: int = _FIELD_CAP) -> str:
    """Cap a field to `limit` characters, appending [truncated] if cut."""
    if len(text) <= limit:
        return text
    return text[: limit - len("[truncated]")] + "[truncated]"


class ToolReviewer:
    """Protocol base class for non-LLM reviewers.

    Subclasses must set model_id and implement review(task, solution_text).
    The pipeline checks isinstance(reviewer, ToolReviewer) to dispatch.
    """
    model_id: str = "tool-reviewer"

    def review(self, task: Task, solution_text: str) -> str:
        raise NotImplementedError


class PublicTestReviewer(ToolReviewer):
    """C4 condition: review by running public sample tests."""

    model_id = "public-test-runner"

    def review(self, task: Task, solution_text: str) -> str:
        """Run the program against public tests.  Return a verdict string."""
        code = extract_program(solution_text)

        if code is None:
            return (
                "VERDICT: REJECT\n"
                f"CRITIQUE: {C4_NO_PROGRAM_TEMPLATE}"
            )

        result = public_check_with_input(task, code)

        if result["all_pass"]:
            return "VERDICT: ACCEPT"

        fail = result["first_failure"]
        kind = fail["kind"]

        if kind == "wrong":
            critique = C4_FAIL_TEMPLATE.format(
                input=_cap(fail["input"]),
                expected=_cap(fail["expected"]),
                actual=_cap(fail["actual"]),
            )
        elif kind == "crash":
            critique = C4_CRASH_TEMPLATE.format(
                input=_cap(fail["input"]),
                stderr_tail=_cap(fail.get("stderr_tail", "")),
            )
        elif kind == "timeout":
            critique = C4_TIMEOUT_TEMPLATE.format(
                input=_cap(fail["input"]),
            )
        else:
            # Unexpected kind — fall back to generic
            critique = f"The program produced unexpected behavior: {kind}"

        return f"VERDICT: REJECT\nCRITIQUE: {critique}"
