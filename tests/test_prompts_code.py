"""Tests for prompts_code.py — frozen prompt constants and hash stability.

Run:  pytest tests/test_prompts_code.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prompts_code import (
    CODE_ANSWER_INSTRUCTION,
    CODE_REVIEW_PROMPT,
    CODE_REVISE_PROMPT,
    C4_FAIL_TEMPLATE,
    C4_CRASH_TEMPLATE,
    C4_TIMEOUT_TEMPLATE,
    C4_NO_PROGRAM_TEMPLATE,
    prompts_hash,
)


def test_prompts_contain_required_placeholders():
    """Review prompt has {problem} and {solution}; revise has {critique} too."""
    assert "{problem}" in CODE_REVIEW_PROMPT
    assert "{solution}" in CODE_REVIEW_PROMPT
    assert "{problem}" in CODE_REVISE_PROMPT
    assert "{solution}" in CODE_REVISE_PROMPT
    assert "{critique}" in CODE_REVISE_PROMPT


def test_c4_templates_have_placeholders():
    """C4 templates have the expected fields."""
    assert "{input}" in C4_FAIL_TEMPLATE
    assert "{expected}" in C4_FAIL_TEMPLATE
    assert "{actual}" in C4_FAIL_TEMPLATE
    assert "{input}" in C4_CRASH_TEMPLATE
    assert "{stderr_tail}" in C4_CRASH_TEMPLATE
    assert "{input}" in C4_TIMEOUT_TEMPLATE


def test_review_prompt_verdict_format():
    """Review prompt asks for VERDICT: ACCEPT or VERDICT: REJECT."""
    assert "VERDICT: ACCEPT" in CODE_REVIEW_PROMPT
    assert "VERDICT: REJECT" in CODE_REVIEW_PROMPT


def test_prompts_hash_deterministic():
    """Hash is stable across calls (frozen prompts)."""
    h1 = prompts_hash()
    h2 = prompts_hash()
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_prompts_hash_is_sha256():
    """Hash is a valid hex string."""
    h = prompts_hash()
    assert all(c in "0123456789abcdef" for c in h)


def test_answer_instruction_mentions_stdin_stdout():
    """Executor instruction asks for stdin/stdout programs."""
    assert "standard input" in CODE_ANSWER_INSTRUCTION
    assert "standard output" in CODE_ANSWER_INSTRUCTION


def test_review_prompt_forbids_running_code():
    """The reviewer prompt explicitly says 'You cannot run the code'."""
    assert "You cannot run the code" in CODE_REVIEW_PROMPT


def run():
    test_prompts_contain_required_placeholders()
    test_c4_templates_have_placeholders()
    test_review_prompt_verdict_format()
    test_prompts_hash_deterministic()
    test_prompts_hash_is_sha256()
    test_answer_instruction_mentions_stdin_stdout()
    test_review_prompt_forbids_running_code()
    print("[ok] all prompts_code tests passed")


if __name__ == "__main__":
    run()
