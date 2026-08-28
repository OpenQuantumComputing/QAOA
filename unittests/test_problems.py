"""
Unit tests for problem classes.

Covers:
- ExactCover: cost, isFeasible, create_circuit
- QUBO: cost, create_circuit, lower-triangular
- PortfolioOptimization: cost, isFeasible, penalty behavior
- MaxKCutOneHot: binstringToLabels, cost, create_circuit
- BucketExactCover: cost, isFeasible, circuit, brute force, scaling, IO
"""

import math
import unittest

import numpy as np
import networkx as nx


class TestExactCover(unittest.TestCase):
    """Tests for the ExactCover problem."""

    def _make_columns(self):
        # Columns represent subsets: {0,1}, {1,2}, {2,3} over elements {0,1,2,3}
        return np.array([
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
        ])

    def test_exact_cover_cost_feasible(self):
        from qaoa.problems import ExactCover
        # Exact cover: subsets 0 and 2 cover all elements exactly once
        # columns[:,0] = {0,1}, columns[:,2] = {2,3}
        columns = self._make_columns()
        problem = ExactCover(columns)
        # "101": use subsets 0 and 2 → covers elements {0,1} and {2,3} exactly once
        cost = problem.cost("101")
        self.assertAlmostEqual(cost, 0.0)

    def test_exact_cover_cost_infeasible(self):
        from qaoa.problems import ExactCover
        columns = self._make_columns()
        problem = ExactCover(columns)
        # "000": no subsets selected → elements uncovered → penalty > 0
        cost = problem.cost("000")
        self.assertLess(cost, 0.0)

    def test_exact_cover_is_feasible_true(self):
        from qaoa.problems import ExactCover
        columns = self._make_columns()
        problem = ExactCover(columns)
        self.assertTrue(problem.isFeasible("101"))

    def test_exact_cover_is_feasible_false(self):
        from qaoa.problems import ExactCover
        columns = self._make_columns()
        problem = ExactCover(columns)
        self.assertFalse(problem.isFeasible("100"))
        self.assertFalse(problem.isFeasible("111"))

    def test_exact_cover_circuit_has_parameters(self):
        from qaoa.problems import ExactCover
        columns = self._make_columns()
        problem = ExactCover(columns)
        problem.create_circuit()
        self.assertIsNotNone(problem.circuit)
        self.assertGreater(len(problem.circuit.parameters), 0)

    def test_exact_cover_n_qubits(self):
        from qaoa.problems import ExactCover
        columns = self._make_columns()
        problem = ExactCover(columns)
        # N_qubits = number of columns = 3
        self.assertEqual(problem.N_qubits, 3)

    def test_exact_cover_with_weights(self):
        from qaoa.problems import ExactCover
        columns = self._make_columns()
        weights = np.array([1.0, 1.0, 1.0])
        problem = ExactCover(columns, weights=weights)
        # Feasible solution "101" should have non-negative cost (weights shift it)
        cost = problem.cost("101")
        self.assertIsInstance(cost, (float, np.floating))


class TestQUBO(unittest.TestCase):
    """Tests for the QUBO problem."""

    def _make_simple_qubo(self):
        """Simple 2-var QUBO: min x0^2 - x1^2 = min x0 - x1 (binary)."""
        from qaoa.problems import QUBO
        Q = np.diag([1.0, -1.0])
        return QUBO(Q=Q)

    def test_qubo_cost_zeros(self):
        problem = self._make_simple_qubo()
        # x = [0, 0] → cost = 0
        self.assertAlmostEqual(problem.cost("00"), 0.0)

    def test_qubo_cost_minimizer(self):
        problem = self._make_simple_qubo()
        # x = [0, 1] → cost = -(-1) = 1... wait, we want min x0 - x1
        # QUBO.cost returns -(x^T Q x) because we maximize
        # Q = diag(1, -1), so x^T Q x = x0 - x1
        # cost("01") = -(0 - 1) = 1  (best: we want minimum cost = most negative)
        cost_01 = problem.cost("01")
        cost_10 = problem.cost("10")
        cost_11 = problem.cost("11")
        cost_00 = problem.cost("00")
        # The QUBO is negated for maximization: QUBO.cost = -(x^T Q x)
        # "01": -(0 - 1) = 1, "10": -(1 - 0) = -1, "11": -(1 - 1) = 0
        self.assertAlmostEqual(cost_01, 1.0)
        self.assertAlmostEqual(cost_10, -1.0)
        self.assertAlmostEqual(cost_00, 0.0)

    def test_qubo_with_linear_terms(self):
        from qaoa.problems import QUBO
        Q = np.diag([0.0, 0.0])
        c = np.array([1.0, 0.0])
        problem = QUBO(Q=Q, c=c)
        # cost = -(c^T x) => cost("10") = -1, cost("01") = 0
        self.assertAlmostEqual(problem.cost("10"), -1.0)
        self.assertAlmostEqual(problem.cost("01"), 0.0)

    def test_qubo_circuit_parametrized(self):
        problem = self._make_simple_qubo()
        problem.create_circuit()
        self.assertIsNotNone(problem.circuit)
        self.assertGreater(len(problem.circuit.parameters), 0)

    def test_qubo_n_qubits(self):
        problem = self._make_simple_qubo()
        self.assertEqual(problem.N_qubits, 2)

    def test_qubo_invalid_Q_raises(self):
        from qaoa.problems import QUBO
        with self.assertRaises(AssertionError):
            QUBO(Q=np.array([1.0, 2.0]))  # 1D, not square

    def test_qubo_non_square_raises(self):
        from qaoa.problems import QUBO
        with self.assertRaises(AssertionError):
            QUBO(Q=np.array([[1.0, 0.0]]))  # 1×2, not square


class TestPortfolioOptimization(unittest.TestCase):
    """Tests for the PortfolioOptimization problem."""

    def _make_portfolio(self):
        from qaoa.problems import PortfolioOptimization
        # 2-asset portfolio
        cov = np.array([[1.0, 0.2], [0.2, 1.0]])
        exp_ret = np.array([0.1, 0.3])
        return PortfolioOptimization(risk=0.5, budget=1, cov_matrix=cov,
                                     exp_return=exp_ret, penalty=2.0)

    def test_portfolio_n_qubits(self):
        problem = self._make_portfolio()
        self.assertEqual(problem.N_qubits, 2)

    def test_portfolio_feasible(self):
        problem = self._make_portfolio()
        # Budget=1: select exactly 1 asset → "01" and "10" are feasible
        self.assertTrue(problem.isFeasible("01"))
        self.assertTrue(problem.isFeasible("10"))

    def test_portfolio_infeasible(self):
        problem = self._make_portfolio()
        self.assertFalse(problem.isFeasible("00"))
        self.assertFalse(problem.isFeasible("11"))

    def test_portfolio_cost_is_float(self):
        problem = self._make_portfolio()
        cost = problem.cost("01")
        self.assertIsInstance(cost, (float, np.floating))

    def test_portfolio_cost_penalized(self):
        problem = self._make_portfolio()
        # cost() should return a float for any bitstring
        cost = problem.cost("01")
        self.assertIsInstance(cost, (float, np.floating))

    def test_portfolio_penalty_lowers_infeasible_cost(self):
        from qaoa.problems import PortfolioOptimization
        cov = np.array([[1.0, 0.2], [0.2, 1.0]])
        exp_ret = np.array([0.1, 0.3])
        # Without penalty: infeasible "11" gets no budget penalty
        problem_no_pen = PortfolioOptimization(risk=0.5, budget=1, cov_matrix=cov,
                                               exp_return=exp_ret, penalty=0)
        # With penalty: infeasible "11" is penalized → lower (more negative) cost
        problem_with_pen = PortfolioOptimization(risk=0.5, budget=1, cov_matrix=cov,
                                                 exp_return=exp_ret, penalty=2.0)
        cost_no_pen = problem_no_pen.cost("11")
        cost_with_pen = problem_with_pen.cost("11")
        self.assertLess(cost_with_pen, cost_no_pen)

    def test_portfolio_circuit_created(self):
        problem = self._make_portfolio()
        problem.create_circuit()
        self.assertIsNotNone(problem.circuit)


class TestMaxKCutOneHot(unittest.TestCase):
    """Tests for the MaxKCutOneHot problem."""

    def _make_graph(self):
        G = nx.Graph()
        G.add_nodes_from([0, 1])
        G.add_edge(0, 1, weight=1.0)
        return G

    def test_maxkcut_one_hot_n_qubits(self):
        from qaoa.problems import MaxKCutOneHot
        G = self._make_graph()
        problem = MaxKCutOneHot(G, k_cuts=2)
        # 2 nodes × 2 colors = 4 qubits
        self.assertEqual(problem.N_qubits, 4)

    def test_maxkcut_one_hot_invalid_k(self):
        from qaoa.problems import MaxKCutOneHot
        G = self._make_graph()
        with self.assertRaises(ValueError):
            MaxKCutOneHot(G, k_cuts=1)
        with self.assertRaises(ValueError):
            MaxKCutOneHot(G, k_cuts=9)

    def test_maxkcut_one_hot_cost_cut(self):
        from qaoa.problems import MaxKCutOneHot
        G = self._make_graph()
        problem = MaxKCutOneHot(G, k_cuts=2)
        # nodes in different colors → cut: "1001" means node0=color0, node1=color1
        cost = problem.cost("1001")
        self.assertGreater(cost, 0.0)

    def test_maxkcut_one_hot_cost_no_cut(self):
        from qaoa.problems import MaxKCutOneHot
        G = self._make_graph()
        problem = MaxKCutOneHot(G, k_cuts=2)
        # nodes same color → no cut: "1010" means both=color0
        cost = problem.cost("1010")
        self.assertAlmostEqual(cost, 0.0)

    def test_maxkcut_one_hot_binstring_to_labels(self):
        from qaoa.problems import MaxKCutOneHot
        G = self._make_graph()
        problem = MaxKCutOneHot(G, k_cuts=2)
        # "1001": node0=color0, node1=color1
        labels = problem.binstringToLabels("1001")
        self.assertIsInstance(labels, str)

    def test_maxkcut_one_hot_circuit(self):
        from qaoa.problems import MaxKCutOneHot
        G = self._make_graph()
        problem = MaxKCutOneHot(G, k_cuts=2)
        problem.create_circuit()
        self.assertIsNotNone(problem.circuit)
        self.assertGreater(len(problem.circuit.parameters), 0)

    def test_maxkcut_one_hot_3cuts(self):
        from qaoa.problems import MaxKCutOneHot
        G = nx.path_graph(3)
        # MaxKCutOneHot requires weighted edges
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0
        problem = MaxKCutOneHot(G, k_cuts=3)
        self.assertEqual(problem.N_qubits, 9)  # 3 nodes × 3 colors
        problem.create_circuit()
        self.assertIsNotNone(problem.circuit)


class TestBucketExactCover(unittest.TestCase):
    """Tests for the BucketExactCover problem (HUBO formulation)."""

    # columns (6 rows × 7 cols):
    #          c0  c1  c2  c3  c4  c5  c6
    # boat1  [  1   1   1   0   0   0   0 ]
    # boat2  [  0   0   0   1   1   1   1 ]
    # order1 [  1   0   0   1   0   1   1 ]
    # order2 [  0   1   0   0   1   0   1 ]
    # order3 [  0   1   0   1   1   0   1 ]
    # order4 [  1   0   1   0   0   1   0 ]
    #
    # weights = [20, 40, 30, 35, 10, 25, 40], num_buckets = 2
    # bucket 0 (boat1): [c0, c1, c2], n_k=3, b_k=ceil(log2(3))=2 → qubits 0,1
    # bucket 1 (boat2): [c3, c4, c5, c6], n_k=4, b_k=ceil(log2(4))=2 → qubits 2,3
    # N_qubits = 4
    #
    # Bitstring layout (LSB first per bucket):
    #   bucket0: string[0]=z0, string[1]=z1; v=z0+2*z1; idx=v%3
    #   bucket1: string[2]=z0, string[3]=z1; v=z0+2*z1; idx=v%4
    #
    # Feasible solutions (all 4 orders covered exactly once):
    #   [c0, c4]: weights 20+10=30 → "0010"  (optimal, least cost)
    #   [c1, c5]: weights 40+25=65 → "1001"
    #   [c2, c6]: weights 30+40=70 → "0111"

    def _make_problem(self, **kwargs):
        from qaoa.problems import BucketExactCover
        columns = np.array([
            [1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1],
            [1, 0, 0, 1, 0, 1, 1],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 1, 0, 1, 1, 0, 1],
            [1, 0, 1, 0, 0, 1, 0],
        ], dtype=float)
        weights = np.array([20., 40., 30., 35., 10., 25., 40.])
        return BucketExactCover(columns, num_buckets=2, weights=weights, **kwargs)

    # Bitstrings for known feasible solutions
    # bucket0: v=0→idx=0→c0, bucket1: v=1→idx=1→c4
    _S_C0C4 = "0010"
    # bucket0: v=1→idx=1→c1, bucket1: v=2→idx=2→c5
    _S_C1C5 = "1001"
    # bucket0: v=2→idx=2→c2, bucket1: v=3→idx=3→c6
    _S_C2C6 = "0111"
    # "0000" → bucket0 v=0→c0 (covers orders 1,4), bucket1 v=0→c3 (covers orders 1,3)
    # → order 1 covered twice, order 2 not covered at all → infeasible
    _S_INFEASIBLE = "0000"

    def test_n_qubits(self):
        bec = self._make_problem()
        self.assertEqual(bec.N_qubits, 4)

    def test_cost_feasible_solution(self):
        bec = self._make_problem()
        self.assertAlmostEqual(bec.unscaled_cost(self._S_C0C4), -30.0)

    def test_cost_suboptimal_feasible(self):
        bec = self._make_problem()
        self.assertAlmostEqual(bec.unscaled_cost(self._S_C1C5), -65.0)

    def test_is_feasible_true(self):
        bec = self._make_problem()
        self.assertTrue(bec.isFeasible(self._S_C0C4))
        self.assertTrue(bec.isFeasible(self._S_C1C5))
        self.assertTrue(bec.isFeasible(self._S_C2C6))

    def test_is_feasible_false(self):
        bec = self._make_problem()
        self.assertFalse(bec.isFeasible(self._S_INFEASIBLE))

    def test_brute_force_finds_cheapest_solution(self):
        bec = self._make_problem()
        opt_sol = bec.brute_force_solve()
        self.assertAlmostEqual(bec.unscaled_cost(opt_sol), -30.0)

    def test_create_circuit_not_none(self):
        bec = self._make_problem()
        circ = bec.create_circuit()
        self.assertIsNotNone(circ)

    def test_encoding_degeneracy_stats(self):
        bec = self._make_problem()
        st = bec.get_encoding_degeneracy_stats()
        # bucket 0: n_k=3, b_k=2 → M=4; v%3 yields multiplicities [2,1,1]
        b0 = st["per_bucket"][0]
        self.assertEqual(b0["n_routes"], 3)
        self.assertEqual(b0["num_encoded_states"], 4)
        self.assertEqual(b0["multiplicities"], [2, 1, 1])
        self.assertEqual(b0["redundant_encodings"], 1)
        # bucket 1: n_k=4, b_k=2 → M=4; perfect fit
        b1 = st["per_bucket"][1]
        self.assertEqual(b1["multiplicities"], [1, 1, 1, 1])
        self.assertEqual(b1["redundant_encodings"], 0)
        self.assertEqual(st["total_encoded_bitstrings"], 16)
        self.assertEqual(st["total_decoded_assignments"], 12)
        self.assertEqual(st["excess_encodings_over_assignments"], 4)


    def test_circuit_has_parameter(self):
        bec = self._make_problem()
        circ = bec.create_circuit()
        self.assertGreater(len(circ.parameters), 0)

    def test_validate_circuit(self):
        bec = self._make_problem()
        bec.create_circuit()
        ok, report = bec.validate_circuit()
        self.assertTrue(ok, msg=str(report))

    def test_unscaled_cost_equals_cost_when_no_scaling(self):
        bec = self._make_problem(scale_problem=False)
        for s in [self._S_C0C4, self._S_C1C5, self._S_C2C6, self._S_INFEASIBLE]:
            self.assertAlmostEqual(bec.cost(s), bec.unscaled_cost(s))

    def test_penalty_auto_computed(self):
        bec = self._make_problem()
        self.assertGreater(bec.penalty_factor, 0)

    def test_no_bucket_columns_ignored(self):
        """Columns with all zeros in the top num_buckets rows are ignored."""
        from qaoa.problems import BucketExactCover
        columns = np.array([
            [1, 1, 1, 0, 0, 0, 0, 0],   # boat1: c0,c1,c2
            [0, 0, 0, 1, 1, 1, 1, 0],   # boat2: c3,c4,c5,c6
            [1, 0, 0, 1, 0, 1, 1, 0],   # order1
            [0, 1, 0, 0, 1, 0, 1, 0],   # order2
            [0, 1, 0, 1, 1, 0, 1, 0],   # order3
            [1, 0, 1, 0, 0, 1, 0, 0],   # order4
        ], dtype=float)
        weights = np.array([20., 40., 30., 35., 10., 25., 40., 99.])
        # c7 has all zeros in top 2 rows → ignored
        bec = BucketExactCover(columns, num_buckets=2, weights=weights)
        # N_qubits should be the same as the 7-column problem (c7 ignored)
        self.assertEqual(bec.N_qubits, 4)
        self.assertNotIn(7, bec._valid_columns)


if __name__ == "__main__":
    unittest.main()
