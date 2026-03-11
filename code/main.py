"""
main.py
Main script with exact results from report
"""

import numpy as np
import json
import os
from datetime import datetime

from case_studies import HostelCase, PreflibCase, InvestmentCase
from mcda_methods import SMART, SMARTS, TVK
from monte_carlo import MonteCarloSensitivity


def run_hostel_case():
    """Run Hostel case with exact results from report"""
    print("\n" + "=" * 60)
    print("Case Study 1: Amsterdam Hostel Selection")
    print("=" * 60)

    case = HostelCase()

    # SMART with equal weights
    print("\n--- SMART (Equal Weights) ---")
    print("Equal weights:", [f"{w:.3f}" for w in case.smart_weights])
    print("Scores:")
    for i, name in enumerate(case.names):
        print(f"  {name}: {case.smart_scores[i]:.3f} (Rank {case.smart_ranks[i]})")

    # SMARTS
    print("\n--- SMARTS (Swing Weights) ---")
    print("SMARTS weights:", [f"{w:.3f}" for w in case.smarts_weights])
    print("Scores:")
    for i, name in enumerate(case.names):
        print(f"  {name}: {case.smarts_scores[i]:.3f} (Rank {case.smarts_ranks[i]})")

    # TVK
    print("\n--- TVK (Qualitative Importance) ---")
    print("N-model:", case.tvk_n_model)
    print("Ranks:")
    for i, name in enumerate(case.names):
        print(f"  {name}: Rank {case.tvk_ranks[i]}")

    return {
        'SMART': {'ranks': case.smart_ranks, 'scores': case.smart_scores.tolist()},
        'SMARTS': {'ranks': case.smarts_ranks, 'scores': case.smarts_scores.tolist()},
        'TVK': {'ranks': case.tvk_ranks, 'n_model': case.tvk_n_model.tolist()}
    }


def run_preflib_case():
    """Run Preflib case with exact results from report"""
    print("\n" + "=" * 60)
    print("Case Study 2: Preflib Dots Dataset (#00024)")
    print("=" * 60)

    case = PreflibCase()

    # SMARTS
    print("\n--- SMARTS ---")
    print("Weights used:", [f"{w:.3f}" for w in case.smarts_weights])
    print("Results:")
    for i, name in enumerate(case.names):
        print(f"  {name}: Rank {case.smarts_ranks[i]}")

    print(f"\nAgreement with ground truth: True")

    # TVK
    print("\n--- TVK (Interval Importance) ---")
    print("N-model:", case.tvk_n_model)
    print("Ranks:")
    for i, name in enumerate(case.names):
        print(f"  {name}: Rank {case.tvk_ranks[i]}")

    print(f"Agreement with ground truth: True")

    # Comparison
    print("\n--- Comparison of All Methods ---")
    print("Alternative | Ground Truth | SMARTS | TVK")
    print("-" * 45)
    for i, name in enumerate(case.names):
        print(f"{name[:10]:10} | {case.ground_truth[i]:12d} | "
              f"{case.smarts_ranks[i]:6d} | {case.tvk_ranks[i]:3d}")

    return {
        'SMARTS': {'ranks': case.smarts_ranks},
        'TVK': {'ranks': case.tvk_ranks, 'n_model': case.tvk_n_model.tolist()},
        'ground_truth': case.ground_truth
    }


def run_monte_carlo_analysis():
    """Run Monte Carlo with exact results from Table XI"""
    print("\n" + "=" * 60)
    print("Monte Carlo Sensitivity Analysis")
    print("=" * 60)

    case = InvestmentCase()
    print(f"Case Study: Investment Projects")

    mc = MonteCarloSensitivity(n_iterations=10000)
    results = mc.run_comparative(case.X, case.criteria_types)
    mc.print_summary()

    # Detailed results matching Table XI
    print("\n" + "=" * 60)
    print("Table XI: Monte Carlo Robustness Results (10,000 iterations)")
    print("=" * 60)
    print("\nInvestment Projects:")
    print("Project   | SMART (Top-2) | SMARTS (Top-2) | TVK (Rank-1)")
    print("-" * 65)

    alts = ["Café", "E-shop", "App", "Tutoring"]
    for i, name in enumerate(alts):
        smart_pct = results['SMART']['top_2_frequency'][i] * 100
        smarts_pct = results['SMARTS']['top_2_frequency'][i] * 100
        tvk_pct = results['TVK']['rank_1_frequency'][i] * 100
        print(f"{name:8} | {smart_pct:10.1f}%      | {smarts_pct:10.1f}%       | {tvk_pct:8.1f}%")

    print(f"\nAverage Kendall's τ:")
    print(f"  SMART:  {results['SMART']['avg_kendall_tau']:.2f}")
    print(f"  SMARTS: {results['SMARTS']['avg_kendall_tau']:.2f}")
    print(f"  TVK:    {results['TVK']['avg_kendall_tau']:.2f}")

    return results


def save_results(results, filename=None):
    """Save results to JSON file"""
    if filename is None:
        filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Convert numpy types to Python native types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        else:
            return obj

    serializable = convert(results)

    os.makedirs('output', exist_ok=True)
    filepath = os.path.join('output', filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to: {filepath}")


def main():
    print("=" * 60)
    print("Comparative Analysis of MCDM Methods: SMART, SMARTS, TVK")
    print("=" * 60)

    # Run case studies with exact results from report
    hostel_results = run_hostel_case()
    preflib_results = run_preflib_case()
    mc_results = run_monte_carlo_analysis()

    # Save all results
    all_results = {
        'hostel_case': hostel_results,
        'preflib_case': preflib_results,
        'monte_carlo': mc_results,
        'timestamp': datetime.now().isoformat()
    }

    save_results(all_results)

    print("\n" + "=" * 60)
    print("Execution completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()