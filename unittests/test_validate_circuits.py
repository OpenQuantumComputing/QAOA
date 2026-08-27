import unittest
import os
import numpy as np
import networkx as nx
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

from qaoa import QAOA, problems, mixers, initialstates
from qaoa.problems.base_problem import ObjectiveSense, Problem
from qaoa.utils.validation import check_phase_separator_exact_problem


class TestValidateMaxCut(unittest.TestCase):

    def test_small_maxcut(self):
        G = nx.Graph()
        G.add_nodes_from(np.arange(0, 2, 1))
        G.add_weighted_edges_from([(0, 1, 1.0)])

        qaoa_inst = QAOA(
            initialstate=initialstates.Plus(),
            problem=problems.MaxKCutBinaryPowerOfTwo(G=G, k_cuts=2),
            mixer=mixers.X(),
        )
        ok, report = qaoa_inst.validate_circuit()
        self.assertTrue(ok, f"Small MaxCut failed: {report}")

    def test_small_maxcut_different_t(self):
        G = nx.Graph()
        G.add_nodes_from(np.arange(0, 2, 1))
        G.add_weighted_edges_from([(0, 1, 1.0)])

        qaoa_inst = QAOA(
            initialstate=initialstates.Plus(),
            problem=problems.MaxKCutBinaryPowerOfTwo(G=G, k_cuts=2),
            mixer=mixers.X(),
        )
        ok, report = qaoa_inst.validate_circuit(t=1.5)
        self.assertTrue(ok, f"Small MaxCut (t=1.5) failed: {report}")

    def test_larger_maxcut(self):
        G = nx.Graph()
        G.add_nodes_from(np.arange(0, 5, 1))
        G.add_weighted_edges_from(
            [(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0),
             (3, 2, 1.0), (3, 4, 1.0), (4, 2, 1.0)]
        )

        qaoa_inst = QAOA(
            initialstate=initialstates.Plus(),
            problem=problems.MaxKCutBinaryPowerOfTwo(G=G, k_cuts=2),
            mixer=mixers.X(),
        )
        ok, report = qaoa_inst.validate_circuit()
        self.assertTrue(ok, f"Larger MaxCut failed: {report}")

    def test_weighted_maxcut(self):
        gml_path = os.path.join(
            os.path.dirname(__file__), "..", "examples", "MaxCut", "data", "w_ba_n10_k4_0.gml"
        )
        if not os.path.exists(gml_path):
            self.skipTest(f"GML data file not found: {gml_path}")

        G = nx.read_gml(gml_path)
        qaoa_inst = QAOA(
            initialstate=initialstates.Plus(),
            problem=problems.MaxKCutBinaryPowerOfTwo(G=G, k_cuts=2),
            mixer=mixers.X(),
        )
        ok, report = qaoa_inst.validate_circuit()
        self.assertTrue(ok, f"Weighted MaxCut failed: {report}")


class TestValidateQUBO(unittest.TestCase):

    def test_qubo_with_linear_and_offset(self):
        Q = -np.array([[-3, 0, 0], [2, 1, 0], [3, 0, 3]])
        c = -np.array([1, -2, 3])
        b = 0.0
        qaoa_inst = QAOA(
            initialstate=initialstates.Plus(),
            problem=problems.QUBO(Q, c, b),
            mixer=mixers.X(),
        )
        ok, report = qaoa_inst.validate_circuit()
        self.assertTrue(ok, f"QUBO (Q+c+b) failed: {report}")

    def test_qubo_combined(self):
        Q_comb = -np.array([[-2, 0, 0], [2, -1, 0], [3, 0, 6]])
        qaoa_inst = QAOA(
            initialstate=initialstates.Plus(),
            problem=problems.QUBO(Q_comb),
            mixer=mixers.X(),
        )
        ok, report = qaoa_inst.validate_circuit()
        self.assertTrue(ok, f"QUBO combined failed: {report}")

    def test_qubo_2x2_diagonal(self):
        Q = -np.array([[1, 0], [0, 2]])
        qaoa_inst = QAOA(
            initialstate=initialstates.Plus(),
            problem=problems.QUBO(Q),
            mixer=mixers.X(),
        )
        ok, report = qaoa_inst.validate_circuit()
        self.assertTrue(ok, f"QUBO 2x2 diagonal failed: {report}")

    def test_qubo_2x2_full(self):
        Q = np.array([[1, 3], [3, 2]])
        qaoa_inst = QAOA(
            initialstate=initialstates.Plus(),
            problem=problems.QUBO(Q),
            mixer=mixers.X(),
        )
        ok, report = qaoa_inst.validate_circuit()
        self.assertTrue(ok, f"QUBO 2x2 full failed: {report}")

    def test_qubo_large_symmetric(self):
        rng = np.random.RandomState(42)
        Q = rng.rand(8, 8)
        Q_sym = 10 * (Q + Q.T)
        qaoa_inst = QAOA(
            initialstate=initialstates.Plus(),
            problem=problems.QUBO(Q_sym),
            mixer=mixers.X(),
        )
        ok, report = qaoa_inst.validate_circuit()
        self.assertTrue(ok, f"QUBO large symmetric failed: {report}")

    def test_qubo_large_asymmetric(self):
        rng = np.random.RandomState(42)
        Q = rng.rand(8, 8)
        qaoa_inst = QAOA(
            initialstate=initialstates.Plus(),
            problem=problems.QUBO(Q),
            mixer=mixers.X(),
        )
        ok, report = qaoa_inst.validate_circuit()
        self.assertTrue(ok, f"QUBO large asymmetric failed: {report}")

    def test_qubo_large_asymmetric_with_linear(self):
        rng = np.random.RandomState(42)
        Q = rng.rand(8, 8)
        c = 3 * (rng.rand(8) - 0.5)
        qaoa_inst = QAOA(
            initialstate=initialstates.Plus(),
            problem=problems.QUBO(Q, c),
            mixer=mixers.X(),
        )
        ok, report = qaoa_inst.validate_circuit()
        self.assertTrue(ok, f"QUBO large asymmetric+c failed: {report}")


class TestValidatePortfolioOptimization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            from qiskit_finance.data_providers import RandomDataProvider
        except ImportError:
            raise unittest.SkipTest("qiskit-finance not installed")

        import datetime
        number_of_assets = 4
        start = datetime.datetime(2020, 1, 1)
        end = start + datetime.timedelta(101)
        tickers = [f"TICKER{i}" for i in range(number_of_assets)]
        provider = RandomDataProvider(tickers=tickers, start=start, end=end, seed=0)
        provider.run()

        cls.cov_matrix = provider.get_period_return_covariance_matrix()
        cls.exp_return = provider.get_period_return_mean_vector()
        cls.budget = 2
        cls.gamma_scale = 50

    def _make_problem(self, penalty=None):
        if penalty is not None:
            return problems.PortfolioOptimization(
                risk=0.5 * self.gamma_scale,
                budget=self.budget,
                cov_matrix=self.cov_matrix,
                exp_return=self.exp_return * self.gamma_scale,
                penalty=penalty,
            )
        else:
            return problems.PortfolioOptimization(
                risk=0.5 * self.gamma_scale,
                budget=self.budget,
                cov_matrix=self.cov_matrix,
                exp_return=self.exp_return * self.gamma_scale,
            )

    def test_penalty_method(self):
        qaoa_inst = QAOA(
            initialstate=initialstates.Plus(),
            problem=self._make_problem(penalty=4 * self.gamma_scale),
            mixer=mixers.X(),
        )
        ok, report = qaoa_inst.validate_circuit()
        self.assertTrue(ok, f"PortOpt penalty failed: {report}")

    def test_xy_mixer_chain(self):
        qaoa_inst = QAOA(
            initialstate=initialstates.Dicke(self.budget),
            problem=self._make_problem(),
            mixer=mixers.XY(case="chain"),
        )
        ok, report = qaoa_inst.validate_circuit()
        self.assertTrue(ok, f"PortOpt XY chain failed: {report}")

    def test_xy_mixer_ring(self):
        qaoa_inst = QAOA(
            initialstate=initialstates.Dicke(self.budget),
            problem=self._make_problem(),
            mixer=mixers.XY(case="ring"),
        )
        ok, report = qaoa_inst.validate_circuit()
        self.assertTrue(ok, f"PortOpt XY ring failed: {report}")

    def test_grover_mixer(self):
        qaoa_inst = QAOA(
            initialstate=initialstates.Dicke(self.budget),
            problem=self._make_problem(),
            mixer=mixers.Grover(initialstates.Dicke(self.budget)),
        )
        ok, report = qaoa_inst.validate_circuit()
        self.assertTrue(ok, f"PortOpt Grover failed: {report}")


class MaxFeasibleOnlyToyProblem(Problem):
    def __init__(self):
        super().__init__(objective_sense=ObjectiveSense.MAXIMIZE)
        self.N_qubits = 1

    def objective_value(self, string):
        return 2.0

    def energy(self, string):
        if string == "1":
            raise ValueError("energy must not be evaluated for infeasible states")
        return super().energy(string)

    def isFeasible(self, string):
        return string == "0"

    def create_circuit(self):
        q = QuantumRegister(1)
        gamma = Parameter("x_gamma")
        self.circuit = QuantumCircuit(q)
        self.circuit.global_phase = 2 * gamma
        self.circuit.p(0.7 * gamma, q[0])


class TestValidateInfeasibleStates(unittest.TestCase):
    def test_omit_infeasible_states(self):
        problem = MaxFeasibleOnlyToyProblem()
        problem.create_circuit()
        with self.assertRaises(ValueError):
            problem.energy("1")

        ok_omit, report_omit = check_phase_separator_exact_problem(
            problem, omit_infeasible_states=True
        )
        self.assertTrue(ok_omit, f"Expected pass when omitting infeasible states: {report_omit}")

        ok_all, report_all = check_phase_separator_exact_problem(
            problem, omit_infeasible_states=False
        )
        self.assertFalse(
            ok_all,
            f"Expected failure when infeasible states are included: {report_all}",
        )


if __name__ == "__main__":
    unittest.main()
