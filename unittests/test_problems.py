"""
Unit tests for problem classes.

Covers:
- ExactCover: cost, isFeasible, create_circuit
- QUBO: cost, create_circuit, lower-triangular
- PortfolioOptimization: cost, isFeasible, cost_nonQUBO
- MaxKCutOneHot: binstringToLabels, cost, create_circuit
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

    def test_portfolio_cost_nonqubo(self):
        problem = self._make_portfolio()
        # cost_nonQUBO should return a float for feasible bitstrings
        cost = problem.cost_nonQUBO("01", penalize=False)
        self.assertIsInstance(cost, (float, np.floating))

    def test_portfolio_cost_nonqubo_with_penalty(self):
        problem = self._make_portfolio()
        # Infeasible "11": penalized cost should be worse (lower, since we negate)
        cost_with = problem.cost_nonQUBO("11", penalize=True)
        cost_without = problem.cost_nonQUBO("11", penalize=False)
        self.assertLess(cost_with, cost_without)

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


if __name__ == "__main__":
    unittest.main()
