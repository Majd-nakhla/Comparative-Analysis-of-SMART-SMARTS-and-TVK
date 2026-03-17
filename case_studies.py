"""
case_studies.py
Case study data from the report - EXACT VALUES + INTERVAL VERSIONS
"""
import numpy as np

class HostelCase:
    """
    Amsterdam Hostel Selection Case Study
    From textbook pages 170-172
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

        self.X = np.array([
            [8, 6, 7, 6, 5],
            [6, 7, 8, 7, 6],
            [4, 9, 6, 8, 7],
            [10, 10, 9, 9, 8],
            [6, 8, 10, 10, 9],
            [3, 5, 5, 5, 10]
        ], dtype=float)

        self.criteria_types = ['benefit', 'benefit', 'benefit', 'benefit', 'benefit']
        self.criteria_names = ['Beds', 'Location', 'Cleanliness', 'Comfort', 'Facilities']

        self.smarts_weights = np.array([0.294, 0.176, 0.235, 0.176, 0.118])
        self.smart_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        self.tvk_ranks = [6, 4, 2, 3, 1, 5]
        self.ic_coefficients = [12, 7, 8, 7, 5]

    def get_interval_data(self, variation=0.1):
        """Get interval versions of data"""
        X_min = self.X * (1 - variation)
        X_max = self.X * (1 + variation)
        w_min = self.smart_weights * (1 - variation)
        w_max = self.smart_weights * (1 + variation)

        w_min = w_min / np.sum(w_min)
        w_max = w_max / np.sum(w_max)

        return X_min, X_max, w_min, w_max


class PreflibCase:
    """
    Preflib Dots dataset (#00024)
    From textbook page 167
    """
    def __init__(self):
        self.names = ["A (200 dots)", "B (209 dots)", "C (218 dots)", "D (227 dots)"]

        self.X = np.array([
            [3.889, 8.5, 392, 30, 1],
            [3.118, 5.2, 356, 14, 52],
            [2.334, 2.1, 298, -11, 123],
            [1.688, 0.3, 154, -32, 34]
        ], dtype=float)

        self.criteria_types = ['benefit', 'benefit', 'benefit', 'benefit', 'cost']
        self.criteria_names = ['Borda', 'Approval', 'Simpson', 'Copeland', 'Dodgson']

        self.smarts_weights = np.array([0.17, 0.13, 0.32, 0.22, 0.27])
        self.tvk_ranks = [1, 2, 3, 4]
        self.ic_coefficients = [10, 5, 18, 12, 15]

    def get_interval_data(self, variation=0.1):
        """Get interval versions of data"""
        X_min = self.X * (1 - variation)
        X_max = self.X * (1 + variation)
        w_min = self.smarts_weights * (1 - variation)
        w_max = self.smarts_weights * (1 + variation)

        w_min = w_min / np.sum(w_min)
        w_max = w_max / np.sum(w_max)

        return X_min, X_max, w_min, w_max


class IntervalTestCase:
    """
    Special test case for 2-criteria analytical method
    """
    def __init__(self):
        self.X = np.array([
            [8, 6],
            [6, 7],
            [4, 9],
            [10, 10]
        ], dtype=float)

        self.criteria_types = ['benefit', 'benefit']
        self.criteria_names = ['Criterion 1', 'Criterion 2']

    def get_interval_data(self, variation=0.1):
        """Get interval versions of data"""
        X_min = self.X * (1 - variation)
        X_max = self.X * (1 + variation)
        w_min = np.array([0.45, 0.45])
        w_max = np.array([0.55, 0.55])

        return X_min, X_max, w_min, w_max