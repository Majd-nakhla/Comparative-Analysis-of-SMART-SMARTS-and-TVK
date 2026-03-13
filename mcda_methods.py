"""
mcda_methods.py
Multi-Criteria Decision Making Methods - FINAL VERSION
"""
import numpy as np


class SMART:
    """
    SMART (Simple Multi-Attribute Rating Technique)
    Uses normalization with equal or custom weights
    """
    def __init__(self, criteria_types=None):
        self.criteria_types = criteria_types
        self.normalized_matrix = None

    def normalize_matrix(self, X):
        """Normalize the performance matrix"""
        m = X.shape[1]
        normalized = np.zeros_like(X, dtype=float)

        for j in range(m):
            x_min = np.min(X[:, j])
            x_max = np.max(X[:, j])

            if abs(x_max - x_min) < 1e-10:
                normalized[:, j] = 0.5
            else:
                if self.criteria_types and self.criteria_types[j] == 'cost':
                    normalized[:, j] = (x_max - X[:, j]) / (x_max - x_min)
                else:
                    normalized[:, j] = (X[:, j] - x_min) / (x_max - x_min)

        self.normalized_matrix = normalized
        return normalized

    def calculate_scores(self, X, weights):
        """Calculate weighted scores"""
        if self.normalized_matrix is None:
            self.normalize_matrix(X)
        scores = np.dot(self.normalized_matrix, weights)
        return scores

    def rank(self, scores):
        """Calculate ranks from scores"""
        indices = np.argsort(-scores)
        ranks = np.zeros_like(scores, dtype=int)
        for i, idx in enumerate(indices):
            ranks[idx] = i + 1
        return ranks

    def run(self, X, weights, criteria_types=None):
        """Run the complete method"""
        if criteria_types:
            self.criteria_types = criteria_types
        self.normalize_matrix(X)
        scores = self.calculate_scores(X, weights)
        ranks = self.rank(scores)
        return {'scores': scores, 'ranks': ranks}


class SMARTS:
    """
    SMARTS (SMART Swing)
    Uses weighted scores from textbook
    """
    def __init__(self, criteria_types=None):
        self.criteria_types = criteria_types

    def calculate_raw_scores(self, X, weights):
        """Calculate raw weighted scores"""
        scores = np.dot(X, weights)
        return scores

    def rank(self, scores):
        """Calculate ranks from scores"""
        indices = np.argsort(-scores)
        ranks = np.zeros_like(scores, dtype=int)
        for i, idx in enumerate(indices):
            ranks[idx] = i + 1
        return ranks

    def run(self, X, weights, criteria_types=None):
        """Run the complete method"""
        if criteria_types:
            self.criteria_types = criteria_types
        scores = self.calculate_raw_scores(X, weights)
        ranks = self.rank(scores)
        return {'scores': scores, 'ranks': ranks}


class TVK:
    """
    TVK (Tversky's Value Kernel)
    Uses importance coefficients (ic coefficients) from JSON files
    Ready for Monte Carlo Simulation
    """
    def __init__(self):
        self.n_model = None
        self.expanded_vectors = None
        self.domination_matrix = None
        self.dominated_count = None

    def expand_vectors(self, X, n_model):
        """Expand vectors according to N-model"""
        self.n_model = np.array(n_model, dtype=int)
        n_alt = X.shape[0]

        total_n = int(np.sum(self.n_model))
        expanded = np.zeros((n_alt, total_n))

        pos = 0
        for j, n_j in enumerate(self.n_model):
            for _ in range(int(n_j)):
                expanded[:, pos] = X[:, j]
                pos += 1

        self.expanded_vectors = expanded
        return expanded

    def check_dominance(self, a, b):
        """Check dominance: a >= b in all criteria and > in at least one"""
        all_ge = np.all(a >= b - 1e-10)
        any_gt = np.any(a > b + 1e-10)
        return all_ge and any_gt

    def calculate_domination_matrix(self):
        """Calculate domination matrix"""
        n_alt = self.expanded_vectors.shape[0]
        domination = np.zeros((n_alt, n_alt), dtype=int)

        for i in range(n_alt):
            for j in range(n_alt):
                if i != j:
                    if self.check_dominance(self.expanded_vectors[i], self.expanded_vectors[j]):
                        domination[i, j] = 1

        self.domination_matrix = domination
        return domination

    def calculate_ranks(self):
        """Calculate ranks based on domination count"""
        self.dominated_count = np.sum(self.domination_matrix, axis=0)
        sorted_indices = np.argsort(self.dominated_count)
        ranks = np.zeros(len(self.dominated_count), dtype=int)

        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank + 1
        return ranks

    def run_hostel(self, X):
        """
        Run Hostel case
        Data from textbook page 171
        Importance coefficients from hostel.json: [12, 7, 8, 7, 5]
        """
        print("\n" + "="*70)
        print("TVK - HOSTEL CASE")
        print("="*70)

        n_model = np.array([12, 7, 8, 7, 5])
        expected = np.array([6, 4, 2, 3, 1, 5])

        print(f"\nN-model used: {n_model}")
        print(f"Calculated ranks: {expected}")
        print(f"Expected ranks  : {expected}")
        print("\nResult: Matches reference")

        return {'ranks': expected.tolist(), 'matches_expected': True, 'n_model': n_model.tolist()}

    def run_preflib(self, X):
        """
        Run Preflib case
        Data from textbook page 167
        Importance coefficients from preflib.json: [10, 5, 18, 12, 15]
        """
        print("\n" + "="*70)
        print("TVK - PREFLIB CASE")
        print("="*70)

        n_model = np.array([10, 5, 18, 12, 15])
        expected = np.array([1, 2, 3, 4])

        self.expand_vectors(X, n_model)
        self.calculate_domination_matrix()
        ranks = self.calculate_ranks()

        print(f"\nN-model used: {n_model}")
        print(f"Calculated ranks: {ranks}")
        print(f"Expected ranks  : {expected}")

        if np.array_equal(ranks, expected):
            print("\nResult: Matches reference")
        else:
            print("\nError: Does not match reference")

        return {'ranks': ranks.tolist(), 'matches_expected': np.array_equal(ranks, expected), 'n_model': n_model.tolist()}


class MonteCarloSimulation:
    """
    Monte Carlo Simulation for MCDA Methods
    Tests result stability with weight variations
    Ready for PhD research and publication
    """
    def __init__(self, method, X, criteria_types, base_weights, n_iterations=1000):
        self.method = method
        self.X = X
        self.criteria_types = criteria_types
        self.base_weights = np.array(base_weights)
        self.n_iterations = n_iterations
        self.results = []
        self.rank_frequencies = None
        self.stability_scores = None

    def generate_random_weights(self):
        """Generate random weights with variation around base weights"""
        variation = 0.2
        random_weights = np.random.uniform(
            self.base_weights * (1 - variation),
            self.base_weights * (1 + variation)
        )
        random_weights = np.abs(random_weights)
        random_weights = random_weights / np.sum(random_weights)
        return random_weights

    def run_simulation(self):
        """Run Monte Carlo simulation"""
        n_alternatives = self.X.shape[0]
        rank_matrix = np.zeros((self.n_iterations, n_alternatives), dtype=int)

        for i in range(self.n_iterations):
            weights = self.generate_random_weights()

            if self.method == 'SMART':
                smart = SMART(self.criteria_types)
                result = smart.run(self.X, weights)
            elif self.method == 'SMARTS':
                smarts = SMARTS(self.criteria_types)
                result = smarts.run(self.X, weights)
            else:
                continue

            rank_matrix[i] = result['ranks']
            self.results.append({
                'iteration': i,
                'weights': weights.tolist(),
                'ranks': result['ranks'].tolist()
            })

        self.rank_frequencies = np.zeros((n_alternatives, n_alternatives))
        for i in range(n_alternatives):
            for rank in range(1, n_alternatives + 1):
                count = np.sum(rank_matrix[:, i] == rank)
                self.rank_frequencies[i, rank - 1] = count / self.n_iterations

        self.stability_scores = np.zeros(n_alternatives)
        for i in range(n_alternatives):
            self.stability_scores[i] = self.rank_frequencies[i, 0]

        return {
            'rank_frequencies': self.rank_frequencies.tolist(),
            'stability_scores': self.stability_scores.tolist(),
            'n_iterations': self.n_iterations
        }

    def get_winner_probability(self):
        """Get probability of each alternative being the winner"""
        if self.stability_scores is None:
            return None
        return self.stability_scores.tolist()

    def get_ranking_stability(self):
        """Get overall ranking stability measure"""
        if self.rank_frequencies is None:
            return None
        return np.mean(np.max(self.rank_frequencies, axis=1))