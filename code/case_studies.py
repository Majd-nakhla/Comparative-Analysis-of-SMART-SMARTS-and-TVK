"""
case_studies.py
Case study data from the report - EXACT VALUES
"""

import numpy as np


class HostelCase:
    """
    Amsterdam Hostel Selection Case Study (Case Study 1)
    From Podinovskii, p. 170-172 - EXACT MATCH
    """

    def __init__(self):
        self.names = [
            "Hostel Slotania",
            "Aivengo Hostel",
            "Shelter City",
            "Flying Pig Downtown",
            "Stayokay Vondelpark",
            "Vita Nova"
        ]

        # Performance matrix from Table I (page 4)
        # [f2 Beds (min), f3 Location (max), f4 Cleanliness (max), f5 Comfort (max), f6 Facilities (max)]
        self.X = np.array([
            [8, 6, 7, 6, 5],  # x1: Slotania
            [6, 7, 8, 7, 6],  # x2: Aivengo
            [4, 9, 6, 8, 7],  # x3: Shelter City
            [10, 10, 9, 9, 8], # x4: Flying Pig
            [6, 8, 10, 10, 9], # x5: Stayokay
            [3, 5, 5, 5, 10]   # x6: Vita Nova
        ], dtype=float)

        self.criteria_types = ['cost', 'benefit', 'benefit', 'benefit', 'benefit']
        self.criteria_names = ['Beds', 'Location', 'Cleanliness', 'Comfort', 'Facilities']

        # SMARTS weights from Table II (page 4) - EXACT
        # w2=0.294, w3=0.176, w4=0.235, w5=0.176, w6=0.118
        self.smarts_weights = np.array([0.294, 0.176, 0.235, 0.176, 0.118])

        # SMART equal weights
        self.smart_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        # SMART scores from Table IV (page 4)
        self.smart_scores = np.array([0.217, 0.434, 0.571, 0.640, 0.794, 0.400])
        self.smart_ranks = [6, 4, 3, 2, 1, 5]

        # SMARTS scores from Table III (page 4)
        self.smarts_scores = np.array([0.248, 0.471, 0.588, 0.575, 0.682, 0.412])
        self.smarts_ranks = [6, 4, 2, 3, 1, 5]

        # TVK qualitative importance order from page 5
        # f2 > f4 > f3 ~ f5 > f6
        self.importance_order = [0, 2, 1, 3, 4]  # indices: 0:f2, 1:f3, 2:f4, 3:f5, 4:f6

        # TVK N-model from Equation 17 (page 5)
        self.tvk_n_model = np.array([12, 7, 8, 7, 5])  # [f2, f3, f4, f5, f6]

        # TVK ranks from Table V (page 5)
        self.tvk_ranks = [6, 4, 2, 3, 1, 5]  # Slotania=6, Aivengo=4, Shelter=2, Flying=3, Stayokay=1, Vita=5

        # TVK dominance relations from Table V
        self.tvk_dominance = [
            [0, 0, 0, 0, 0, 0],  # Slotania dominated by all
            [0, 0, 0, 0, 0, 0],  # Aivengo dominated by x3,x4,x5
            [1, 1, 0, 0, 0, 1],  # Shelter dominates x1,x2,x6
            [1, 1, 0, 0, 0, 1],  # Flying dominates x1,x2,x6
            [1, 1, 1, 1, 0, 1],  # Stayokay dominates all except Vita?
            [0, 0, 0, 0, 0, 0]   # Vita Nova
        ]


class PreflibCase:
    """
    Preflib Dots dataset (#00024) - Case Study 2
    From report pages 5-6 - EXACT MATCH
    """

    def __init__(self):
        self.names = ["A (200 dots)", "B (209 dots)", "C (218 dots)", "D (227 dots)"]
        self.ground_truth = [1, 2, 3, 4]  # A < B < C < D (lower is better in ground truth)

        # Performance matrix from Table VI (page 5) - EXACT
        # Criteria: [Borda, Approval Rate, Simpson, Copeland, Dodgson]
        self.X = np.array([
            [3.889, 8.5, 392, 30, 1],    # A
            [3.118, 5.2, 356, 14, 52],   # B
            [2.334, 2.1, 298, -11, 123], # C
            [1.688, 0.3, 154, -32, 34]   # D
        ], dtype=float)

        # Criteria types: Borda, Approval, Simpson, Copeland are benefit (higher better)
        # Dodgson is cost (lower better)
        self.criteria_types = ['benefit', 'benefit', 'benefit', 'benefit', 'cost']
        self.criteria_names = ['Borda', 'Approval', 'Simpson', 'Copeland', 'Dodgson']

        # SMARTS weights from Table VII (page 5) - EXACT
        # Simpson:0.32, Dodgson:0.27, Copeland:0.22, Borda:0.17, Approval:0.13
        self.smarts_weights = np.array([0.17, 0.13, 0.32, 0.22, 0.27])

        # SMARTS scores from page 6
        self.smarts_scores = np.array([1.00, 0.73, 0.35, 0.00])  # Approximate
        self.smarts_ranks = [1, 2, 3, 4]

        # TVK interval constraints from Equations 22-25 (page 6) - EXACT
        self.interval_constraints = [
            (2, 4, 1.5, 2.0),  # Simpson / Dodgson: 1.5 <= n2/n4 <= 2.0
            (4, 3, 1.2, 1.4),  # Dodgson / Copeland: 1.2 <= n4/n3 <= 1.4
            (3, 0, 1.3, 1.5),  # Copeland / Borda: 1.3 <= n3/n0 <= 1.5
            (0, 1, 1.2, 1.4)   # Borda / Approval: 1.2 <= n0/n1 <= 1.4
        ]

        # TVK N-model from Equation 26 (page 6) - EXACT
        # [n_Borda, n_Approval, n_Simpson, n_Copeland, n_Dodgson] = [10, 5, 18, 12, 15]
        self.tvk_n_model = np.array([10, 5, 18, 12, 15])

        # TVK ranks from Table IX (page 6) - EXACT
        self.tvk_ranks = [1, 2, 3, 4]  # A > B > C > D


class InvestmentCase:
    """
    Investment Projects Case Study for Monte Carlo
    Based on Table XI in the report (page 7) - EXACT MATCH
    """

    def __init__(self):
        self.names = ["Café", "E-shop", "App", "Tutoring"]

        # Performance matrix
        self.X = np.array([
            [70, 30, 80, 12],   # Café
            [85, 60, 40, 6],    # E-shop
            [95, 40, 60, 8],    # App
            [60, 20, 50, 10]    # Tutoring
        ], dtype=float)

        # Criteria types: Profit(benefit), Risk(cost), Cost(cost), Time(benefit)
        self.criteria_types = ['benefit', 'cost', 'cost', 'benefit']
        self.criteria_names = ['Profit', 'Risk', 'Cost', 'Time']

        # Expected Monte Carlo results from Table XI (page 7) - EXACT PERCENTAGES
        self.expected_smart_top2 = [0.031, 0.362, 0.971, 0.886]  # 3.1%, 36.2%, 97.1%, 88.6%
        self.expected_smarts_top2 = [0.028, 0.347, 0.965, 0.892]  # 2.8%, 34.7%, 96.5%, 89.2%
        self.expected_tvk_rank1 = [0.002, 0.021, 0.789, 0.194]    # 0.2%, 2.1%, 78.9%, 19.4%
        self.expected_tau = [0.87, 0.85, 0.79]  # SMART, SMARTS, TVK