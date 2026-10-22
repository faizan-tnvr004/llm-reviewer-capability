"""Tests for code_exec.py — output normalization and sandboxed execution.

Run:  pytest tests/test_code_exec.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from code_exec import normalize_output, code_hash


def test_normalize_output_basic():
    """CRLF -> LF, trailing whitespace stripped, trailing blanks dropped."""
    assert normalize_output("hello\r\nworld\r\n") == "hello\nworld\n"
    assert normalize_output("hello  \nworld  \n") == "hello\nworld\n"
    assert normalize_output("hello\n\n\n") == "hello\n"
    assert normalize_output("") == ""
    assert normalize_output("\n\n") == ""


def test_normalize_output_preserves_internal_blanks():
    """Internal blank lines are preserved."""
    assert normalize_output("a\n\nb\n") == "a\n\nb\n"


def test_normalize_output_adds_trailing_newline():
    """Non-empty output always ends with exactly one newline."""
    assert normalize_output("hello") == "hello\n"
    assert normalize_output("hello\n") == "hello\n"


def test_code_hash_deterministic():
    """Same code -> same hash, different code -> different hash."""
    code1 = "print('hello')\n"
    code2 = "print('world')\n"
    assert code_hash(code1) == code_hash(code1)
    assert code_hash(code1) != code_hash(code2)


def test_code_hash_normalizes_whitespace():
    """Leading/trailing whitespace and multiple blank lines don't change hash."""
    code_a = "  print('hello')  \n"
    code_b = "print('hello')\n\n\n\n"
    assert code_hash(code_a) == code_hash(code_b)


def test_code_hash_length():
    """Hash is 16 hex characters."""
    h = code_hash("print('x')")
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


def run():
    test_normalize_output_basic()
    test_normalize_output_preserves_internal_blanks()
    test_normalize_output_adds_trailing_newline()
    test_code_hash_deterministic()
    test_code_hash_normalizes_whitespace()
    test_code_hash_length()
    print("[ok] all code_exec tests passed")


if __name__ == "__main__":
    run()
