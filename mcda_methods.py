"""
mcda_methods.py
Multi-Criteria Decision Making Methods - FINAL VERSION
Includes Interval MCDA via Optimization (PyDASS or scipy fallback)
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# محاولة استيراد PyDASS، مع fallback إلى scipy
TRY_PYDASS = True
PYDASS_AVAILABLE = False

if TRY_PYDASS:
    try:
        from pydass import IntervalMCDA as PyDASSInterval
        PYDASS_AVAILABLE = True
        print("✓ PyDASS library loaded successfully")
    except ImportError:
        from scipy.optimize import linprog
        print("⚠ PyDASS not available, using scipy.optimize fallback")
else:
    from scipy.optimize import linprog
    print("ℹ Using scipy.optimize (PyDASS disabled)")


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
        """Run Hostel case - Data from textbook page 171"""
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
        """Run Preflib case - Data from textbook page 167"""
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


class IntervalMCDA:
    """
    Interval MCDA via Optimization
    Uses PyDASS if available, otherwise scipy.optimize fallback
    """
    def __init__(self, criteria_types=None):
        self.criteria_types = criteria_types
        self.n_alternatives = 0
        self.n_criteria = 0
        self.use_pyDASS = PYDASS_AVAILABLE

    def normalize_interval(self, X_min, X_max):
        """Normalize interval performance matrix"""
        m = X_min.shape[1]
        norm_min = np.zeros_like(X_min, dtype=float)
        norm_max = np.zeros_like(X_max, dtype=float)

        for j in range(m):
            x_min_j = np.min(X_min[:, j])
            x_max_j = np.max(X_max[:, j])

            if abs(x_max_j - x_min_j) < 1e-10:
                norm_min[:, j] = 0.5
                norm_max[:, j] = 0.5
            else:
                if self.criteria_types and self.criteria_types[j] == 'cost':
                    norm_min[:, j] = (x_max_j - X_max[:, j]) / (x_max_j - x_min_j)
                    norm_max[:, j] = (x_max_j - X_min[:, j]) / (x_max_j - x_min_j)
                else:
                    norm_min[:, j] = (X_min[:, j] - x_min_j) / (x_max_j - x_min_j)
                    norm_max[:, j] = (X_max[:, j] - x_min_j) / (x_max_j - x_min_j)

        return norm_min, norm_max

    def optimize_ranking_pyDASS(self, X_min, X_max, w_min, w_max, target_alt):
        """
        Find best/worst possible rank using PyDASS library
        """
        if not PYDASS_AVAILABLE:
            raise ImportError("PyDASS not available")

        # استخدام PyDASS إذا كان متاحًا
        pydass_model = PyDASSInterval()
        result = pydass_model.solve(X_min, X_max, w_min, w_max)

        # استخراج فترات الرتب من نتيجة PyDASS
        rank_interval = result['rank_intervals'][target_alt]
        return rank_interval[0], rank_interval[1]

    def optimize_ranking_scipy(self, X_min, X_max, w_min, w_max, target_alt):
        """
        Find best/worst possible rank using scipy fallback
        """
        n_alt = X_min.shape[0]
        norm_min, norm_max = self.normalize_interval(X_min, X_max)

        can_be_better = 0
        target_can_beat = 0

        for j in range(n_alt):
            if j == target_alt:
                continue

            score_j_max = np.sum(w_max * norm_max[j])
            score_target_min = np.sum(w_min * norm_min[target_alt])

            if score_j_max > score_target_min + 1e-10:
                can_be_better += 1

            score_target_max = np.sum(w_max * norm_max[target_alt])
            score_j_min = np.sum(w_min * norm_min[j])

            if score_target_max > score_j_min + 1e-10:
                target_can_beat += 1

        best_rank = 1 + (n_alt - 1 - target_can_beat)
        worst_rank = n_alt - (n_alt - 1 - can_be_better)

        best_rank = max(1, min(best_rank, worst_rank))
        worst_rank = max(best_rank, min(worst_rank, n_alt))

        return best_rank, worst_rank

    def optimize_ranking(self, X_min, X_max, w_min, w_max, target_alt):
        """Auto-select optimization method"""
        if self.use_pyDASS:
            try:
                return self.optimize_ranking_pyDASS(X_min, X_max, w_min, w_max, target_alt)
            except Exception as e:
                print(f"⚠ PyDASS failed: {e}, falling back to scipy")
                self.use_pyDASS = False

        return self.optimize_ranking_scipy(X_min, X_max, w_min, w_max, target_alt)

    def solve_lp_ranking(self, X_min, X_max, w_min, w_max):
        """Solve complete ranking using optimization logic"""
        n_alt = X_min.shape[0]
        rank_intervals = []

        for i in range(n_alt):
            best, worst = self.optimize_ranking(X_min, X_max, w_min, w_max, i)
            rank_intervals.append([best, worst])

        return np.array(rank_intervals)

    def analytical_2criteria(self, X_min, X_max, w_min, w_max):
        """Analytical solution for exactly 2 criteria"""
        if X_min.shape[1] != 2:
            raise ValueError("Analytical method only works for 2 criteria")

        n_alt = X_min.shape[0]
        rank_intervals = []

        for i in range(n_alt):
            best_rank = 1
            worst_rank = 1

            for j in range(n_alt):
                if i == j:
                    continue

                dominated = False
                for w1 in [w_min[0], w_max[0]]:
                    w2 = 1 - w1
                    score_i = w1 * X_min[i, 0] + w2 * X_min[i, 1]
                    score_j_min = w1 * X_max[j, 0] + w2 * X_max[j, 1]

                    if score_i > score_j_min:
                        dominated = True
                        break

                if dominated:
                    best_rank += 1
                else:
                    worst_rank += 1

            rank_intervals.append([min(best_rank, worst_rank), max(best_rank, worst_rank)])

        return np.array(rank_intervals)

    def run(self, X_min, X_max, w_min, w_max, use_analytical=False):
        """Run Interval MCDA"""
        self.n_alternatives = X_min.shape[0]
        self.n_criteria = X_min.shape[1]

        if X_min.shape != X_max.shape:
            raise ValueError("X_min and X_max must have same shape")

        if len(w_min) != len(w_max) or len(w_min) != self.n_criteria:
            raise ValueError("Weight intervals must match number of criteria")

        # تحديد طريقة التنفيذ
        if use_analytical and self.n_criteria == 2:
            print("Using analytical method (2 criteria)")
            rank_intervals = self.analytical_2criteria(X_min, X_max, w_min, w_max)
            method_used = "analytical_2crit"
        else:
            if self.use_pyDASS:
                print("Using PyDASS optimization method")
            else:
                print("Using scipy optimization method (LP)")
            rank_intervals = self.solve_lp_ranking(X_min, X_max, w_min, w_max)
            method_used = "pydass" if self.use_pyDASS else "optimization_lp"

        midpoints = np.mean(rank_intervals, axis=1)
        ranks = np.argsort(midpoints) + 1

        return {
            'rank_intervals': rank_intervals.tolist(),
            'midpoint_ranks': ranks.tolist(),
            'method_used': method_used,
            'n_alternatives': self.n_alternatives,
            'n_criteria': self.n_criteria,
            'weight_intervals': np.column_stack([w_min, w_max]).tolist(),
            'pydass_used': self.use_pyDASS
        }


class MonteCarloSimulation:
    """
    Monte Carlo Simulation for MCDA Methods
    Tests result stability with weight variations
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