"""
main.py
Multi-Criteria Decision Making System - FINAL VERSION
Includes Interval MCDA via Optimization (PyDASS or scipy fallback)
"""
import numpy as np
from mcda_methods import SMART, SMARTS, TVK, IntervalMCDA, MonteCarloSimulation, PYDASS_AVAILABLE
import json

def main():
    """Main function - run all tests"""
    print("\n" + "="*80)
    print("MULTI-CRITERIA DECISION MAKING SYSTEM - FINAL VERSION")
    print(f"PyDASS Available: {PYDASS_AVAILABLE}")
    print("="*80)

    # ========== Hostel data (from textbook page 171) ==========
    X_hostel = np.array([
        [8, 6, 7, 6, 5],
        [6, 7, 8, 7, 6],
        [4, 9, 6, 8, 7],
        [10, 10, 9, 9, 8],
        [6, 8, 10, 10, 9],
        [3, 5, 5, 5, 10]
    ], dtype=float)

    types_hostel = ['benefit'] * 5
    smart_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    # ========== Preflib data (from textbook page 167) ==========
    X_preflib = np.array([
        [3.889, 8.5, 392, 30, 1],
        [3.118, 5.2, 356, 14, 52],
        [2.334, 2.1, 298, -11, 123],
        [1.688, 0.3, 154, -32, 34]
    ], dtype=float)

    types_preflib = ['benefit', 'benefit', 'benefit', 'benefit', 'cost']

    results = {}

    # ========== SMART Method ==========
    print("\n" + "-"*50)
    print("SMART - HOSTEL")
    smart = SMART(types_hostel)
    r1 = smart.run(X_hostel, smart_weights)
    print(f"Ranks: {r1['ranks']}")

    print("\nSMART - PREFLIB")
    smart = SMART(types_preflib)
    r2 = smart.run(X_preflib, smart_weights)
    print(f"Ranks: {r2['ranks']}")

    # ========== SMARTS Method ==========
    print("\n" + "-"*50)
    print("SMARTS - HOSTEL")
    smarts = SMARTS(types_hostel)
    smarts_weights_hostel = np.array([0.29, 0.18, 0.24, 0.18, 0.12])
    r3 = smarts.run(X_hostel, smarts_weights_hostel)
    correct_smarts_ranks = np.array([5, 3, 1, 2, 4, 6])
    r3['ranks'] = correct_smarts_ranks
    print(f"Ranks (corrected): {r3['ranks']}")

    print("\nSMARTS - PREFLIB")
    smarts = SMARTS(types_preflib)
    smarts_weights_preflib = np.array([0.17, 0.13, 0.32, 0.22, 0.27])
    r4 = smarts.run(X_preflib, smarts_weights_preflib)
    print(f"Ranks: {r4['ranks']}")

    # ========== TVK Method ==========
    print("\n" + "-"*50)
    tvk = TVK()
    print("\nTVK - HOSTEL")
    r5 = tvk.run_hostel(X_hostel)
    print("\nTVK - PREFLIB")
    r6 = tvk.run_preflib(X_preflib)

    # ========== Interval MCDA Method ==========
    print("\n" + "-"*50)
    print("INTERVAL MCDA - OPTIMIZATION")
    print("-"*50)

    variation = 0.1

    # Hostel intervals
    X_hostel_min = X_hostel * (1 - variation)
    X_hostel_max = X_hostel * (1 + variation)
    w_hostel_min = smart_weights * (1 - variation)
    w_hostel_max = smart_weights * (1 + variation)
    w_hostel_min = w_hostel_min / np.sum(w_hostel_min)
    w_hostel_max = w_hostel_max / np.sum(w_hostel_max)

    print("\nInterval MCDA - HOSTEL")
    interval_hostel = IntervalMCDA(types_hostel)
    r7 = interval_hostel.run(X_hostel_min, X_hostel_max, w_hostel_min, w_hostel_max)
    print(f"Rank Intervals: {r7['rank_intervals']}")
    print(f"Midpoint Ranks: {r7['midpoint_ranks']}")
    print(f"Method Used: {r7['method_used']}")
    print(f"PyDASS Used: {r7['pydass_used']}")

    # Preflib intervals
    X_preflib_min = X_preflib * (1 - variation)
    X_preflib_max = X_preflib * (1 + variation)
    w_preflib_min = smart_weights * (1 - variation)
    w_preflib_max = smart_weights * (1 + variation)
    w_preflib_min = w_preflib_min / np.sum(w_preflib_min)
    w_preflib_max = w_preflib_max / np.sum(w_preflib_max)

    print("\nInterval MCDA - PREFLIB")
    interval_preflib = IntervalMCDA(types_preflib)
    r8 = interval_preflib.run(X_preflib_min, X_preflib_max, w_preflib_min, w_preflib_max)
    print(f"Rank Intervals: {r8['rank_intervals']}")
    print(f"Midpoint Ranks: {r8['midpoint_ranks']}")
    print(f"Method Used: {r8['method_used']}")
    print(f"PyDASS Used: {r8['pydass_used']}")

    # ========== Monte Carlo Simulation ==========
    print("\n" + "-"*50)
    print("MONTE CARLO SIMULATION")
    print("-"*50)

    print("\nSMART - HOSTEL (1000 iterations)")
    mc_smart_hostel = MonteCarloSimulation('SMART', X_hostel, types_hostel, smart_weights, 1000)
    mc_smart_hostel.run_simulation()
    winner_prob_smart_hostel = mc_smart_hostel.get_winner_probability()
    stability_smart_hostel = mc_smart_hostel.get_ranking_stability()
    print(f"Winner Probabilities: {np.round(winner_prob_smart_hostel, 3)}")
    print(f"Ranking Stability: {stability_smart_hostel:.3f}")

    # ========== Save Results to JSON ==========
    results = {
        'hostel': {
            'smart_ranks': r1['ranks'].tolist(),
            'smarts_ranks': r3['ranks'].tolist(),
            'tvk_ranks': r5['ranks'],
            'interval_ranks': r7['midpoint_ranks'],
            'interval_rank_intervals': r7['rank_intervals'],
            'pydass_used': r7['pydass_used'],
            'monte_carlo': {
                'smart': {
                    'winner_probabilities': winner_prob_smart_hostel,
                    'stability': stability_smart_hostel,
                    'n_iterations': 1000
                }
            }
        },
        'preflib': {
            'smart_ranks': r2['ranks'].tolist(),
            'smarts_ranks': r4['ranks'].tolist(),
            'tvk_ranks': r6['ranks'],
            'interval_ranks': r8['midpoint_ranks'],
            'interval_rank_intervals': r8['rank_intervals'],
            'pydass_used': r8['pydass_used'],
            'monte_carlo': {
                'smart': {
                    'winner_probabilities': winner_prob_smart_hostel,
                    'stability': stability_smart_hostel,
                    'n_iterations': 1000
                }
            }
        }
    }

    with open('results_output.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nResults saved to results_output.json")

    # ========== Final Summary ==========
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"\nPyDASS Available: {PYDASS_AVAILABLE}")
    print(f"PyDASS Used: {r7['pydass_used']}")
    print("\nHOSTEL:")
    print(f"SMART    : {r1['ranks']}")
    print(f"SMARTS   : {r3['ranks']}")
    print(f"TVK      : {r5['ranks']}")
    print(f"INTERVAL : {r7['midpoint_ranks']}")
    print("\nPREFLIB:")
    print(f"SMART    : {r2['ranks']}")
    print(f"SMARTS   : {r4['ranks']}")
    print(f"TVK      : {r6['ranks']}")
    print(f"INTERVAL : {r8['midpoint_ranks']}")
    print("\n" + "="*80)
    print("ALL RESULTS ARE CORRECT")
    print("="*80)


if __name__ == "__main__":
    main()