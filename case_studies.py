"""
case_studies.py
Case study data from the report - EXACT VALUES
"""
import numpy as np


class HostelCase:
    """
    Amsterdam Hostel Selection Case Study
    From textbook pages 170-172

    6 hostels in Amsterdam evaluated on 5 criteria:
    - Beds
    - Location
    - Cleanliness
    - Comfort
    - Facilities
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

        # Performance matrix from table
        # [f2 Beds, f3 Location, f4 Cleanliness, f5 Comfort, f6 Facilities]
        self.X = np.array([
            [8, 6, 7, 6, 5],   # x1: Slotania
            [6, 7, 8, 7, 6],   # x2: Aivengo
            [4, 9, 6, 8, 7],   # x3: Shelter
            [10, 10, 9, 9, 8], # x4: Flying
            [6, 8, 10, 10, 9], # x5: Stayokay
            [3, 5, 5, 5, 10]   # x6: Vita
        ], dtype=float)

        self.criteria_types = ['benefit', 'benefit', 'benefit', 'benefit', 'benefit']
        self.criteria_names = ['Beds', 'Location', 'Cleanliness', 'Comfort', 'Facilities']

        # SMARTS weights from textbook
        self.smarts_weights = np.array([0.294, 0.176, 0.235, 0.176, 0.118])

        # SMART equal weights
        self.smart_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        # Expected TVK ranks from textbook page 171
        self.tvk_ranks = [6, 4, 2, 3, 1, 5]

        # Importance coefficients from hostel.json
        self.ic_coefficients = [12, 7, 8, 7, 5]


class PreflibCase:
    """
    Preflib Dots dataset (#00024)
    From textbook page 167

    4 alternatives (A, B, C, D) evaluated on 5 voting criteria:
    - Borda
    - Approval
    - Simpson
    - Copeland
    - Dodgson
    """
    def __init__(self):
        self.names = ["A (200 dots)", "B (209 dots)", "C (218 dots)", "D (227 dots)"]

        # Performance matrix
        # [Borda, Approval, Simpson, Copeland, Dodgson]
        self.X = np.array([
            [3.889, 8.5, 392, 30, 1],    # A
            [3.118, 5.2, 356, 14, 52],   # B
            [2.334, 2.1, 298, -11, 123], # C
            [1.688, 0.3, 154, -32, 34]   # D
        ], dtype=float)

        self.criteria_types = ['benefit', 'benefit', 'benefit', 'benefit', 'cost']
        self.criteria_names = ['Borda', 'Approval', 'Simpson', 'Copeland', 'Dodgson']

        # SMARTS weights from textbook
        self.smarts_weights = np.array([0.17, 0.13, 0.32, 0.22, 0.27])

        # Expected TVK ranks from textbook page 167
        self.tvk_ranks = [1, 2, 3, 4]

        # Importance coefficients from preflib.json
        self.ic_coefficients = [10, 5, 18, 12, 15]