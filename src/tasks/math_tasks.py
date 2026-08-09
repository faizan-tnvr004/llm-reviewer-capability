"""Math word-problem generator.

Five template types (per the analysis plan), each with randomized parameters
and a difficulty knob. Ground truth is computed programmatically, so grading
is exact by construction.

Templates:
  1. discount_tax   — price with a percentage discount, then sales tax
  2. unit_rate      — speed/distance/time with a mid-journey change
  3. proportion     — recipe/mixture scaling with a unit conversion
  4. work_rate      — two workers at different rates, one leaves early
  5. simple_interest— principal, rate, term; compare two options

Difficulty knob (1=easiest .. 3=hardest) controls number magnitude, how many
steps the problem takes, and whether "distractor" values are included. This is
the lever for the pre-run difficulty calibration (target: 20-70% single-agent
error band).

All money answers round to 2 decimals; counts/other answers as noted per task.
"""

from __future__ import annotations

import random
from typing import List

from core import Task, NUMERIC_ANSWER_INSTRUCTION


def generate_math_tasks(n: int = 25, difficulty: int = 2, seed: int = 0) -> List[Task]:
    """Generate n math word-problem tasks, cycling through the 5 templates."""
    rng = random.Random(seed)
    templates = [_discount_tax, _unit_rate, _proportion, _work_rate, _simple_interest]
    tasks: List[Task] = []
    for i in range(n):
        template = templates[i % len(templates)]
        tasks.append(template(f"math_{i:03d}", rng, difficulty))
    return tasks


# ---------------------------------------------------------------------------
# Template 1: discount then tax
# ---------------------------------------------------------------------------

def _discount_tax(task_id: str, rng: random.Random, difficulty: int) -> Task:
    price = rng.randint(40, 90) if difficulty == 1 else rng.randint(137, 899) + rng.choice([0, 0.5, 0.25])
    discount = rng.choice([10, 20, 25]) if difficulty == 1 else rng.choice([12, 17, 23, 34])
    tax = rng.choice([5, 10]) if difficulty == 1 else rng.choice([7, 8.25, 9.5, 13])

    discounted = price * (1 - discount / 100)
    final = round(discounted * (1 + tax / 100), 2)

    distractor = ""
    if difficulty >= 3:
        member_fee = rng.randint(5, 20)
        distractor = (
            f" The store also charges a one-time membership fee of ${member_fee} "
            f"per year, but the customer is already a member."
        )

    prompt = (
        f"A jacket is priced at ${price}. It is on sale at {discount}% off. "
        f"After the discount, a sales tax of {tax}% is applied.{distractor} "
        f"What is the final amount the customer pays for the jacket, in dollars?"
    ) + NUMERIC_ANSWER_INSTRUCTION

    return Task(task_id, "math", prompt, final,
                meta={"template": "discount_tax", "difficulty": difficulty, "round_to": 2})


# ---------------------------------------------------------------------------
# Template 2: unit rate with a mid-journey change
# ---------------------------------------------------------------------------

def _unit_rate(task_id: str, rng: random.Random, difficulty: int) -> Task:
    v1 = rng.randint(40, 70)
    t1 = rng.choice([1.0, 1.5, 2.0]) if difficulty == 1 else rng.choice([1.25, 1.75, 2.25, 2.6])
    v2 = v1 + rng.choice([-15, -10, 10, 20])
    t2 = rng.choice([0.5, 1.0]) if difficulty == 1 else rng.choice([0.4, 0.75, 1.2, 1.8])

    total_dist = v1 * t1 + v2 * t2
    total_time = t1 + t2
    avg = round(total_dist / total_time, 2)

    prompt = (
        f"A truck drives at {v1} km/h for {t1} hours, then at {v2} km/h for "
        f"{t2} hours. What is its average speed over the whole journey, in km/h?"
    ) + NUMERIC_ANSWER_INSTRUCTION

    return Task(task_id, "math", prompt, avg,
                meta={"template": "unit_rate", "difficulty": difficulty, "round_to": 2})


# ---------------------------------------------------------------------------
# Template 3: proportion / mixture scaling with unit conversion
# ---------------------------------------------------------------------------

def _proportion(task_id: str, rng: random.Random, difficulty: int) -> Task:
    base_servings = rng.choice([4, 6, 8])
    target_servings = rng.choice([10, 14, 18, 22])
    grams = rng.randint(120, 480)

    if difficulty >= 2:
        # answer requested in kilograms -> forces a unit conversion step
        answer = round(grams * target_servings / base_servings / 1000, 3)
        unit_clause = "in kilograms"
        round_to = 3
    else:
        answer = round(grams * target_servings / base_servings, 2)
        unit_clause = "in grams"
        round_to = 2

    prompt = (
        f"A recipe for {base_servings} servings uses {grams} grams of flour. "
        f"How much flour is needed for {target_servings} servings, {unit_clause}?"
    ) + NUMERIC_ANSWER_INSTRUCTION

    return Task(task_id, "math", prompt, answer,
                meta={"template": "proportion", "difficulty": difficulty, "round_to": round_to})


# ---------------------------------------------------------------------------
# Template 4: work rate, one worker leaves early
# ---------------------------------------------------------------------------

def _work_rate(task_id: str, rng: random.Random, difficulty: int) -> Task:
    a_hours = rng.choice([4, 5, 6, 8])
    b_hours = a_hours + rng.choice([2, 3, 4])
    together = rng.choice([1, 1.5, 2])

    done_together = together * (1 / a_hours + 1 / b_hours)
    remaining = 1 - done_together
    b_finish = round(remaining * b_hours, 2)  # hours B needs alone

    hours_word = "hour" if together == 1 else "hours"
    prompt = (
        f"Worker A can paint a fence alone in {a_hours} hours; worker B alone in "
        f"{b_hours} hours. They work together for {together} {hours_word}, then A leaves. "
        f"How many more hours does B need to finish the fence alone?"
    ) + NUMERIC_ANSWER_INSTRUCTION

    return Task(task_id, "math", prompt, b_finish,
                meta={"template": "work_rate", "difficulty": difficulty, "round_to": 2})


# ---------------------------------------------------------------------------
# Template 5: simple interest, compare two options
# ---------------------------------------------------------------------------

def _simple_interest(task_id: str, rng: random.Random, difficulty: int) -> Task:
    principal = rng.randrange(2000, 9000, 500)
    while True:
        r1 = rng.choice([4, 5, 6])
        r2 = r1 + rng.choice([1, 1.5, 2])
        years1 = rng.choice([3, 4, 5])
        years2 = years1 - rng.choice([1, 2])
        if abs(r1 * years1 - r2 * years2) > 1e-9:
            break  # avoid degenerate tasks where both options are equal

    total1 = principal * (1 + r1 / 100 * years1)
    total2 = principal * (1 + r2 / 100 * years2)
    diff = round(abs(total1 - total2), 2)

    prompt = (
        f"You can invest ${principal} at {r1}% simple annual interest for {years1} years, "
        f"or at {r2}% simple annual interest for {years2} years. What is the difference, "
        f"in dollars, between the total amounts (principal plus interest) of the two options?"
    ) + NUMERIC_ANSWER_INSTRUCTION

    return Task(task_id, "math", prompt, diff,
                meta={"template": "simple_interest", "difficulty": difficulty, "round_to": 2})
