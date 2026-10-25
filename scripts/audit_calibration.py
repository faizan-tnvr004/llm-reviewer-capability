"""Audit script for calibration numbers and mathematical projections.

Reproduces all statistical numbers in the pre-registered analysis plan:
- Calibration strata Wilson 95% intervals
- Stratified expected C0 accuracy for N = 125 (40 medium, 85 hard)
- Plausible range under separate stratum Wilson bounds
- Expected correct programs
- Expected C1 false rejections at 20% and 35%
- H2 testability assessment
"""

from __future__ import annotations

import math


def wilson_interval(k: int, n: int, z: float = 1.95996) -> tuple[float, float]:
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    diff = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return centre - diff, centre + diff


def main():
    # 1. Calibration Batch 1 numbers
    k_med, n_med = 5, 7
    k_hard, n_hard = 3, 13
    k_tot, n_tot = 8, 20

    ci_med = wilson_interval(k_med, n_med)
    ci_hard = wilson_interval(k_hard, n_hard)
    ci_tot = wilson_interval(k_tot, n_tot)

    print("=== CALIBRATION BATCH 1 STRATA ===")
    print(f"Medium (5/7)  : {k_med/n_med*100:.1f}% | 95% Wilson CI: [{ci_med[0]*100:.1f}%, {ci_med[1]*100:.1f}%]")
    print(f"Hard   (3/13) : {k_hard/n_hard*100:.1f}% | 95% Wilson CI: [{ci_hard[0]*100:.1f}%, {ci_hard[1]*100:.1f}%]")
    print(f"Overall (8/20): {k_tot/n_tot*100:.1f}% | 95% Wilson CI: [{ci_tot[0]*100:.1f}%, {ci_tot[1]*100:.1f}%]")

    # 2. Main study N = 125 projections
    n_total_problems = 125
    n_main_med = 40
    n_main_hard = 85

    exp_correct_med = n_main_med * (k_med / n_med)
    exp_correct_hard = n_main_hard * (k_hard / n_hard)
    exp_correct_total = exp_correct_med + exp_correct_hard
    exp_acc = exp_correct_total / n_total_problems

    # Plausible range using separate stratum Wilson bounds
    low_correct = n_main_med * ci_med[0] + n_main_hard * ci_hard[0]
    high_correct = n_main_med * ci_med[1] + n_main_hard * ci_hard[1]
    low_acc = low_correct / n_total_problems
    high_acc = high_correct / n_total_problems

    print("\n=== STRATIFIED MAIN STUDY PROJECTIONS (N = 125) ===")
    print(f"Medium problems: {n_main_med} ({n_main_med/n_total_problems*100:.1f}%)")
    print(f"Hard problems  : {n_main_hard} ({n_main_hard/n_total_problems*100:.1f}%)")
    print(f"Expected correct programs : {exp_correct_total:.2f} out of {n_total_problems} ({exp_acc*100:.2f}%)")
    print(f"Plausible range (strata)  : [{low_acc*100:.1f}%, {high_acc*100:.1f}%] ({low_correct:.1f} to {high_correct:.1f} correct)")

    # 3. Expected false rejections under C1
    fr_at_20 = exp_correct_total * 0.20
    fr_at_35 = exp_correct_total * 0.35
    print("\n=== C1 FALSE REJECTION PROJECTIONS ===")
    print(f"At 20% false rejection rate: {fr_at_20:.2f} false rejections (floor={math.floor(fr_at_20)}, round={round(fr_at_20)})")
    print(f"At 35% false rejection rate: {fr_at_35:.2f} false rejections (floor={math.floor(fr_at_35)}, round={round(fr_at_35)})")

    # 4. H2 testability assessment
    print("\n=== H2 TESTABILITY ASSESSMENT ===")
    print(f"Low-n threshold for H2 testability: n_false_rej >= 10")
    print(f"At 20% FR rate: {fr_at_20:.1f} false rejections -> borderline ({'TESTABLE' if round(fr_at_20) >= 10 else 'BORDERLINE/BELOW'})")
    print(f"At 35% FR rate: {fr_at_35:.1f} false rejections -> TESTABLE ({fr_at_35:.1f} >= 10)")
    min_fr_rate_needed = 10 / exp_correct_total
    print(f"Minimum false rejection rate needed for >= 10 false rejections: {min_fr_rate_needed*100:.1f}%")


if __name__ == "__main__":
    main()
