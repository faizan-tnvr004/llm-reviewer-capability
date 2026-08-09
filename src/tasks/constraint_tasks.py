"""Constraint-puzzle task family.

WHY THIS FAMILY: unlike arithmetic, logic puzzles produce *conceptual* errors
(a model applies the constraints wrongly) with *discrete* answers (an ordering),
so there is no rounding noise. Two models that get it wrong can be wrong in the
SAME way (same incorrect ordering) or DIFFERENT ways — which is exactly the
signal D_across needs.

PUZZLE TYPE: linear ordering / ranking. N items must be placed in positions
1..N subject to constraints like "A is immediately before B", "C is somewhere
before D", "E is not in position k". Every generated puzzle has EXACTLY ONE
solution (verified by brute force at generation time — a puzzle with 0 or >1
solutions is discarded), so the ground truth is unambiguous.

ANSWER FORMAT: the model outputs the final ordering as a comma-separated list
on an ANSWER line, e.g.  ANSWER: C, A, E, B, D

GRADING: exact match of the ordering. FINGERPRINT: the model's ordering
(canonicalised), so two wrong attempts giving the same wrong ordering share a
fingerprint — this is what lets D_across detect correlated vs. diverse errors.
"""

from __future__ import annotations

import itertools
import random
import re
from typing import List, Optional, Tuple

from core import Task

# Answer line: "ANSWER: C, A, E, B, D" (letters/names separated by commas)
_CP_ANSWER = re.compile(r"ANSWER\s*[:=]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


# ---------------------------------------------------------------------------
# Constraint types (each is a predicate over a candidate ordering)
# ---------------------------------------------------------------------------

def _c_immediately_before(a, b):
    def pred(order):
        ia, ib = order.index(a), order.index(b)
        return ib == ia + 1
    return pred, f"{a} is immediately before {b}"


def _c_before(a, b):
    def pred(order):
        return order.index(a) < order.index(b)
    return pred, f"{a} is somewhere before {b}"


def _c_not_in_position(a, pos):  # pos is 1-indexed
    def pred(order):
        return order.index(a) != pos - 1
    return pred, f"{a} is not in position {pos}"


def _c_in_position(a, pos):
    def pred(order):
        return order.index(a) == pos - 1
    return pred, f"{a} is in position {pos}"


def _c_adjacent(a, b):
    def pred(order):
        return abs(order.index(a) - order.index(b)) == 1
    return pred, f"{a} and {b} are next to each other (in either order)"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def _solve(items, constraints) -> List[tuple]:
    """Return all orderings satisfying every constraint (brute force)."""
    sols = []
    for perm in itertools.permutations(items):
        if all(pred(perm) for pred, _ in constraints):
            sols.append(perm)
    return sols


def generate_constraint_tasks(n: int = 20, difficulty: int = 2, seed: int = 0) -> List[Task]:
    """Generate n linear-ordering puzzles, each with a unique solution.

    difficulty controls N (number of items) and thus the search space:
      1 -> 4 items, 2 -> 5 items, 3 -> 6 items.
    More items = harder for models to track all constraints = more errors.
    """
    rng = random.Random(seed)
    n_items = {1: 4, 2: 5, 3: 6, 4: 7, 5: 8}.get(difficulty, 5)
    names = ["A", "B", "C", "D", "E", "F", "G", "H"][:n_items]

    tasks: List[Task] = []
    attempts = 0
    while len(tasks) < n and attempts < n * 200:
        attempts += 1
        task = _make_one_puzzle(f"cp_{len(tasks):03d}", rng, names, difficulty)
        if task is not None:
            tasks.append(task)
    if len(tasks) < n:
        raise RuntimeError(f"only generated {len(tasks)}/{n} unique-solution puzzles; "
                           "loosen constraints or raise attempt cap")
    return tasks


def _make_one_puzzle(task_id, rng, names, difficulty) -> Optional[Task]:
    """Build one puzzle; return None if it doesn't have exactly one solution."""
    # Pick a target solution, then generate constraints consistent with it.
    target = list(names)
    rng.shuffle(target)

    n_constraints = {1: 3, 2: 4, 3: 5, 4: 6, 5: 7}.get(difficulty, 4)
    builders = []
    for _ in range(n_constraints * 3):  # over-generate, we'll trim
        kind = rng.choice(["imm_before", "before", "not_pos", "adjacent"])
        if kind == "imm_before":
            i = rng.randrange(len(target) - 1)
            builders.append(_c_immediately_before(target[i], target[i + 1]))
        elif kind == "before":
            i, j = sorted(rng.sample(range(len(target)), 2))
            builders.append(_c_before(target[i], target[j]))
        elif kind == "not_pos":
            a = rng.choice(names)
            # a "not in position" constraint that's TRUE for the target
            wrong_pos = rng.randrange(len(names)) + 1
            if target.index(a) != wrong_pos - 1:
                builders.append(_c_not_in_position(a, wrong_pos))
        elif kind == "adjacent":
            i = rng.randrange(len(target) - 1)
            builders.append(_c_adjacent(target[i], target[i + 1]))

    # Greedily add constraints until the solution is unique.
    chosen = []
    for b in builders:
        chosen.append(b)
        sols = _solve(names, chosen)
        if len(sols) == 1:
            break
    else:
        return None  # never narrowed to a unique solution

    sols = _solve(names, chosen)
    if len(sols) != 1 or list(sols[0]) != target:
        return None  # sanity: unique AND equals our intended target

    # De-duplicate constraint text and require a minimum count so puzzles
    # aren't trivially short.
    seen, clue_texts = set(), []
    for _, text in chosen:
        if text not in seen:
            seen.add(text)
            clue_texts.append(text)
    if len(clue_texts) < 2:
        return None

    positions = ", ".join(str(i + 1) for i in range(len(names)))
    prompt = (
        f"There are {len(names)} items ({', '.join(names)}) to be placed in "
        f"positions {positions} (position 1 is first). The following clues all hold:\n"
        + "\n".join(f"  - {t}" for t in clue_texts)
        + "\n\nThere is exactly one valid ordering. Work out the ordering, then "
        "give your final answer on the last line as the items in order from "
        "position 1 to the last, separated by commas, in exactly this format:\n"
        "ANSWER: X, Y, Z, ..."
    )

    return Task(task_id, "constraint", prompt, tuple(target),
                meta={"difficulty": difficulty, "n_items": len(names),
                      "n_clues": len(clue_texts), "names": names})


# ---------------------------------------------------------------------------
# Extraction + grading
# ---------------------------------------------------------------------------

def extract_ordering(response: str, names: List[str]) -> Optional[Tuple[str, ...]]:
    """Pull the model's final ordering from its response.
    Returns a tuple of item names, or None if unparseable."""
    lines = _CP_ANSWER.findall(response)
    if not lines:
        return None
    token = lines[-1]
    # Strip markdown emphasis that can cling to the line or items.
    token = token.replace("*", "").replace("`", "")
    # Split on commas (and stray whitespace); keep only recognised item names,
    # uppercased, preserving order.
    parts = [p.strip().strip(".").upper() for p in re.split(r"[,\s]+", token) if p.strip()]
    nameset = set(names)
    ordering = [p for p in parts if p in nameset]
    if not ordering:
        return None
    return tuple(ordering)


def grade_constraint(task: Task, response: str) -> dict:
    """Grade a constraint puzzle by exact-ordering match.

    fingerprint = the model's ordering as 'A>B>C...' (or 'NO_ANSWER'/'MALFORMED'),
    so two wrong attempts with the same wrong ordering share a fingerprint.
    """
    names = task.meta["names"]
    ordering = extract_ordering(response, names)

    if ordering is None:
        return {"correct": False, "fingerprint": "NO_ANSWER", "extracted": None}

    # A well-formed answer must be a permutation of exactly the item set.
    if sorted(ordering) != sorted(names):
        # Model gave a malformed/incomplete ordering. Keep what it said in the
        # fingerprint so identical malformed answers still match.
        fp = "MALFORMED:" + ">".join(ordering)
        return {"correct": False, "fingerprint": fp, "extracted": ordering}

    correct = tuple(ordering) == tuple(task.ground_truth)
    return {"correct": correct, "fingerprint": ">".join(ordering), "extracted": ordering}