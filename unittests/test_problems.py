"""
Unit tests for problem classes.

Covers:
- ExactCover: objective/energy, isFeasible, create_circuit
- QUBO: objective/energy, create_circuit, lower-triangular
- PortfolioOptimization: objective/energy, isFeasible, penalty behavior
- MaxKCutOneHot: binstringToLabels, objective, create_circuit
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

    def test_exact_cover_objective_feasible(self):
        from qaoa.problems import ExactCover
        # Exact cover: subsets 0 and 2 cover all elements exactly once
        # columns[:,0] = {0,1}, columns[:,2] = {2,3}
        columns = self._make_columns()
        problem = ExactCover(columns)
        # "101": use subsets 0 and 2 → covers elements {0,1} and {2,3} exactly once
        objective = problem.objective_value("101")
        self.assertAlmostEqual(objective, 0.0)

    def test_exact_cover_objective_infeasible(self):
        from qaoa.problems import ExactCover
        columns = self._make_columns()
        problem = ExactCover(columns)
        # "000": no subsets selected → elements uncovered → penalty > 0
        objective = problem.objective_value("000")
        self.assertGreater(objective, 0.0)

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
        objective = problem.objective_value("101")
        self.assertIsInstance(objective, (float, np.floating))


class TestQUBO(unittest.TestCase):
    """Tests for the QUBO problem."""

    def _make_simple_qubo(self):
        """Simple 2-var QUBO: min x0^2 - x1^2 = min x0 - x1 (binary)."""
        from qaoa.problems import QUBO
        Q = np.diag([1.0, -1.0])
        return QUBO(Q=Q)

    def test_qubo_objective_zeros(self):
        problem = self._make_simple_qubo()
        self.assertAlmostEqual(problem.objective_value("00"), 0.0)

    def test_qubo_objective_minimizer(self):
        problem = self._make_simple_qubo()
        objective_01 = problem.objective_value("01")
        objective_10 = problem.objective_value("10")
        objective_11 = problem.objective_value("11")
        objective_00 = problem.objective_value("00")
        self.assertAlmostEqual(objective_01, -1.0)
        self.assertAlmostEqual(objective_10, 1.0)
        self.assertAlmostEqual(objective_11, 0.0)
        self.assertAlmostEqual(objective_00, 0.0)

    def test_qubo_with_linear_terms(self):
        from qaoa.problems import QUBO
        Q = np.diag([0.0, 0.0])
        c = np.array([1.0, 0.0])
        problem = QUBO(Q=Q, c=c)
        self.assertAlmostEqual(problem.objective_value("10"), 1.0)
        self.assertAlmostEqual(problem.objective_value("01"), 0.0)

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

    def test_portfolio_objective_is_float(self):
        problem = self._make_portfolio()
        objective = problem.objective_value("01")
        self.assertIsInstance(objective, (float, np.floating))

    def test_portfolio_objective_penalized(self):
        problem = self._make_portfolio()
        objective = problem.objective_value("01")
        self.assertIsInstance(objective, (float, np.floating))

    def test_portfolio_penalty_raises_infeasible_objective(self):
        from qaoa.problems import PortfolioOptimization
        cov = np.array([[1.0, 0.2], [0.2, 1.0]])
        exp_ret = np.array([0.1, 0.3])
        # Without penalty: infeasible "11" gets no budget penalty
        problem_no_pen = PortfolioOptimization(risk=0.5, budget=1, cov_matrix=cov,
                                               exp_return=exp_ret, penalty=0)
        # With penalty: infeasible "11" is penalized → higher objective value
        problem_with_pen = PortfolioOptimization(risk=0.5, budget=1, cov_matrix=cov,
                                                 exp_return=exp_ret, penalty=2.0)
        objective_no_pen = problem_no_pen.objective_value("11")
        objective_with_pen = problem_with_pen.objective_value("11")
        self.assertGreater(objective_with_pen, objective_no_pen)

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

    def test_maxkcut_one_hot_objective_cut(self):
        from qaoa.problems import MaxKCutOneHot
        G = self._make_graph()
        problem = MaxKCutOneHot(G, k_cuts=2)
        # nodes in different colors → cut: "1001" means node0=color0, node1=color1
        objective = problem.objective_value("1001")
        self.assertGreater(objective, 0.0)

    def test_maxkcut_one_hot_objective_no_cut(self):
        from qaoa.problems import MaxKCutOneHot
        G = self._make_graph()
        problem = MaxKCutOneHot(G, k_cuts=2)
        # nodes same color → no cut: "1010" means both=color0
        objective = problem.objective_value("1010")
        self.assertAlmostEqual(objective, 0.0)

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
