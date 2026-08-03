import tempfile
import unittest

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

from qaoa import QAOA, initialstates, mixers, problems
from qaoa.problems.base_problem import ObjectiveSense, Problem
from qaoa.utils import BitFlip, Statistic, qaoaIO, compute_approx_ratio


class MinToyProblem(Problem):
    def __init__(self):
        super().__init__(objective_sense=ObjectiveSense.MINIMIZE)
        self.N_qubits = 2

    def objective_value(self, string):
        return float(sum(int(b) for b in string))

    def create_circuit(self):
        q = QuantumRegister(2)
        self.circuit = QuantumCircuit(q)
        gamma = Parameter("x_gamma")
        self.circuit.p(-gamma, q[0])
        self.circuit.p(-gamma, q[1])


class MaxToyProblem(Problem):
    def __init__(self):
        super().__init__(objective_sense=ObjectiveSense.MAXIMIZE)
        self.N_qubits = 2

    def objective_value(self, string):
        return float(sum(int(b) for b in string))

    def create_circuit(self):
        q = QuantumRegister(2)
        self.circuit = QuantumCircuit(q)
        gamma = Parameter("x_gamma")
        self.circuit.p(gamma, q[0])
        self.circuit.p(gamma, q[1])


class WrongSignMinProblem(MinToyProblem):
    def create_circuit(self):
        q = QuantumRegister(2)
        self.circuit = QuantumCircuit(q)
        gamma = Parameter("x_gamma")
        self.circuit.p(gamma, q[0])
        self.circuit.p(gamma, q[1])


class TestObjectiveSenseMigration(unittest.TestCase):
    def test_objective_sense_enum(self):
        self.assertEqual(ObjectiveSense.MINIMIZE.value, "minimize")
        self.assertEqual(ObjectiveSense.MAXIMIZE.value, "maximize")
        with self.assertRaises(ValueError):
            MinToyProblem().objective_sense = ObjectiveSense("invalid")

    def test_energy_objective_invariants(self):
        pmin = MinToyProblem()
        pmax = MaxToyProblem()
        self.assertEqual(pmin.energy("10"), pmin.objective_value("10"))
        self.assertEqual(pmax.energy("10"), -pmax.objective_value("10"))

    def test_removed_compatibility_apis(self):
        self.assertFalse(hasattr(MinToyProblem(), "cost"))
        self.assertFalse(hasattr(MinToyProblem(), "computeMinMaxCosts"))

    def test_maxcut_objective_and_energy(self):
        G = nx.Graph()
        G.add_edge(0, 1, weight=2.0)
        problem = problems.MaxCut(G)
        self.assertEqual(problem.objective_value("01"), 2.0)
        self.assertEqual(problem.energy("01"), -2.0)

    def test_qubo_default_minimize_and_objective(self):
        Q = np.array([[1.0, 0.0], [0.0, 2.0]])
        problem = problems.QUBO(Q)
        self.assertEqual(problem.objective_sense, ObjectiveSense.MINIMIZE)
        self.assertEqual(problem.objective_value("11"), 3.0)
        self.assertEqual(problem.energy("11"), 3.0)

    def test_exact_cover_objective(self):
        columns = np.array([[1, 0], [0, 1]])
        weights = np.array([2.0, 3.0])
        problem = problems.ExactCover(columns, weights=weights, penalty_factor=5.0)
        self.assertEqual(problem.objective_value("11"), 5.0)

    def test_portfolio_objective(self):
        cov = np.array([[1.0, 0.2], [0.2, 1.0]])
        exp_ret = np.array([0.1, 0.3])
        problem = problems.PortfolioOptimization(
            risk=0.5, budget=1, cov_matrix=cov, exp_return=exp_ret, penalty=2.0
        )
        # x = 01 => 0.5*1 - 0.3 + 0 penalty = 0.2
        self.assertAlmostEqual(problem.objective_value("01"), 0.2)

    def test_lower_tail_cvar_and_maximize_mapping(self):
        stat = Statistic(cvar=0.5)
        for v in [1.0, 2.0, 3.0, 4.0]:
            stat.add_sample(v, 1.0, str(v))
        self.assertAlmostEqual(stat.get_CVaR(), 1.5)

        max_problem = MaxToyProblem()
        stat2 = Statistic(cvar=0.5)
        for v in [-1.0, -2.0, -3.0, -4.0]:
            stat2.add_sample(v, 1.0, str(v))
        energy_cvar = stat2.get_CVaR()
        self.assertAlmostEqual(energy_cvar, -3.5)
        self.assertAlmostEqual(max_problem.objective_from_energy(energy_cvar), 3.5)

    def test_deterministic_selection_by_energy(self):
        pmin = MinToyProblem()
        pmax = MaxToyProblem()
        self.assertLess(pmin.energy("00"), pmin.energy("11"))
        self.assertLess(pmax.energy("11"), pmax.energy("00"))
        self.assertLess(pmin.objective_value("00"), pmin.objective_value("11"))
        self.assertGreater(pmax.objective_value("11"), pmax.objective_value("00"))

    def test_phase_validation_pass_and_fail(self):
        ok_min, _ = MinToyProblem().validate_circuit()
        ok_max, _ = MaxToyProblem().validate_circuit()
        ok_bad, _ = WrongSignMinProblem().validate_circuit()
        self.assertTrue(ok_min)
        self.assertTrue(ok_max)
        self.assertFalse(ok_bad)

    def test_flip_boosting_both_senses(self):
        flipper = BitFlip(2)
        np.random.seed(0)
        pmin = MinToyProblem()
        s0 = "11"
        s1 = flipper.boost_samples(problem=pmin, string=s0, K=10)
        self.assertLessEqual(pmin.energy(s1[::-1]), pmin.energy(s0[::-1]))

        np.random.seed(0)
        pmax = MaxToyProblem()
        s0 = "00"
        s1 = flipper.boost_samples(problem=pmax, string=s0, K=10)
        self.assertLessEqual(pmax.energy(s1[::-1]), pmax.energy(s0[::-1]))

    def test_objective_and_energy_bounds(self):
        pmin = MinToyProblem()
        pmax = MaxToyProblem()
        self.assertEqual(pmin.objective_bounds(), (0.0, 2.0))
        self.assertEqual(pmax.objective_bounds(), (0.0, 2.0))
        self.assertEqual(pmin.energy_bounds(), (0.0, 2.0))
        self.assertEqual(pmax.energy_bounds(), (-2.0, -0.0))
        self.assertEqual(pmin.optimal_objective(), 0.0)
        self.assertEqual(pmax.optimal_objective(), 2.0)

    def test_qaoa_energy_and_objective(self):
        from qiskit_aer import AerSimulator

        G = nx.path_graph(3)
        for u, v in G.edges():
            G[u][v]["weight"] = 1.0
        q = QAOA(
            problems.MaxCut(G),
            mixers.X(),
            initialstates.Plus(),
            backend=AerSimulator(),
            shots=128,
        )
        q.optimize(depth=1, angles={"gamma": [0, np.pi, 3], "beta": [0, np.pi, 3]})
        self.assertFalse(hasattr(q, "get_Exp"))
        self.assertAlmostEqual(
            q.get_objective(depth=1), -q.get_energy(depth=1)
        )

    def test_serialization_roundtrip_preserves_objective_sense(self):
        pd = qaoaIO.ExactCoverProblemData(
            columns=np.array([[1, 0], [0, 1]]),
            weights=np.array([1.0, 2.0]),
            solution=np.array([1, 1]),
            hamming_weight=1,
            objective_sense="minimize",
        )
        params = qaoaIO.QAOAParameters(
            cvar=0.5,
            init_method=qaoaIO.InitMethod.PLUS,
            mixer_method=qaoaIO.MixerMethod.X,
            backend="sim",
            optimizer="COBYLA",
            N_qubits=2,
            depths={1: qaoaIO.DepthResult([0.1, 0.2], {"00": 1}, 0.01, 0.0, 0.0)},
        )
        result = qaoaIO.QAOAResult(problem=pd, qaoa_params=params)
        with tempfile.NamedTemporaryFile(suffix=".json") as fp:
            result.save(fp.name)
            loaded = qaoaIO.QAOAResult.load(fp.name)
        self.assertEqual(loaded.schema_version, 3)
        self.assertEqual(loaded.problem.objective_sense, "minimize")

    def test_legacy_serialization_is_rejected(self):
        legacy_payload = {
            "problem": {
                "problem_type": "ExactCover",
                "columns": [[1, 0], [0, 1]],
                "weights": [1.0, 2.0],
                "solution": [1, 1],
                "hamming_weight": 1,
            },
            "qaoa_params": {
                "cvar": 0.5,
                "init_method": "PLUS",
                "mixer_method": "X",
                "backend": "sim",
                "optimizer": "COBYLA",
                "N_qubits": 2,
                "depths": {"1": {"optimal_angles": [0.1, 0.2], "histogram": {"00": 1}, "opt_time": 0.01}},
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as fp:
            import json

            json.dump(legacy_payload, fp)
            fp.flush()
            with self.assertRaises(ValueError):
                qaoaIO.QAOAResult.load(fp.name)

    def test_approximation_ratio_mapping_formula(self):
        """Legacy inline-formula tests – kept for regression."""
        best, worst = 10.0, 2.0
        value_best, value_worst = 10.0, 2.0
        ratio_max_best = (value_best - worst) / (best - worst)
        ratio_max_worst = (value_worst - worst) / (best - worst)
        self.assertAlmostEqual(ratio_max_best, 1.0)
        self.assertAlmostEqual(ratio_max_worst, 0.0)

        best, worst = 2.0, 10.0
        value_best, value_worst = 2.0, 10.0
        ratio_min_best = (worst - value_best) / (worst - best)
        ratio_min_worst = (worst - value_worst) / (worst - best)
        self.assertAlmostEqual(ratio_min_best, 1.0)
        self.assertAlmostEqual(ratio_min_worst, 0.0)

    def test_compute_approx_ratio_maximize(self):
        # best value = max_objective (10), worst = min_objective (2)
        self.assertAlmostEqual(compute_approx_ratio(10.0, 2.0, 10.0, ObjectiveSense.MAXIMIZE), 1.0)
        self.assertAlmostEqual(compute_approx_ratio(2.0, 2.0, 10.0, ObjectiveSense.MAXIMIZE), 0.0)
        self.assertAlmostEqual(compute_approx_ratio(6.0, 2.0, 10.0, ObjectiveSense.MAXIMIZE), 0.5)
        # also accepts string sense
        self.assertAlmostEqual(compute_approx_ratio(10.0, 2.0, 10.0, "maximize"), 1.0)

    def test_compute_approx_ratio_minimize(self):
        # best value = min_objective (2), worst = max_objective (10)
        self.assertAlmostEqual(compute_approx_ratio(2.0, 2.0, 10.0, ObjectiveSense.MINIMIZE), 1.0)
        self.assertAlmostEqual(compute_approx_ratio(10.0, 2.0, 10.0, ObjectiveSense.MINIMIZE), 0.0)
        self.assertAlmostEqual(compute_approx_ratio(6.0, 2.0, 10.0, ObjectiveSense.MINIMIZE), 0.5)
        # also accepts string sense
        self.assertAlmostEqual(compute_approx_ratio(2.0, 2.0, 10.0, "minimize"), 1.0)

    def test_compute_approx_ratio_trivial_landscape(self):
        self.assertAlmostEqual(compute_approx_ratio(5.0, 5.0, 5.0, ObjectiveSense.MAXIMIZE), 1.0)
        self.assertAlmostEqual(compute_approx_ratio(5.0, 5.0, 5.0, ObjectiveSense.MINIMIZE), 1.0)

    def test_compute_approx_ratio_array(self):
        values = np.array([2.0, 6.0, 10.0])
        expected_max = np.array([0.0, 0.5, 1.0])
        expected_min = np.array([1.0, 0.5, 0.0])
        np.testing.assert_allclose(
            compute_approx_ratio(values, 2.0, 10.0, ObjectiveSense.MAXIMIZE), expected_max
        )
        np.testing.assert_allclose(
            compute_approx_ratio(values, 2.0, 10.0, ObjectiveSense.MINIMIZE), expected_min
        )

    def test_missing_sense_raises(self):
        """Problem.__init__ must raise ValueError when objective_sense is omitted."""
        with self.assertRaises(ValueError) as ctx:
            # Anonymous concrete Problem subclass – no sense provided.
            class _NakedProblem(Problem):
                def objective_value(self, s):
                    return 0.0
                def create_circuit(self):
                    pass
            _NakedProblem()
        self.assertIn("objective_sense", str(ctx.exception))
