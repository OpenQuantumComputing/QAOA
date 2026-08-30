"""Unit tests for the uploaded unified MIS/MVC components."""

import unittest

import networkx as nx
import numpy as np
from qiskit.quantum_info import Operator, Statevector

from qaoa.initialstates import MIS_MVC_InitialState
from qaoa.mixers import VertexSubsetLXMixer
from qaoa.problems import MIS_MVC_Problem


def q0_first(index, num_qubits):
    """Convert a Qiskit basis index to the package's q0-first bitstring."""

    return format(index, f"0{num_qubits}b")[::-1]


class TestProblem(unittest.TestCase):
    def setUp(self):
        self.graph = nx.path_graph(4)

    def test_rejects_invalid_problem_kind(self):
        with self.assertRaises(TypeError):
            MIS_MVC_Problem(self.graph, problem_kind="invalid")

    def test_default_objective_sense(self):
        mis = MIS_MVC_Problem(self.graph, problem_kind="mis")
        mvc = MIS_MVC_Problem(self.graph, problem_kind="mvc")

        self.assertEqual(mis.objective_sense.value, "maximize")
        self.assertEqual(mvc.objective_sense.value, "minimize")

    def test_objective_sense_can_be_overridden(self):
        mis_as_minimization = MIS_MVC_Problem(
            self.graph,
            problem_kind="mis",
            objective_sense="minimize",
        )
        mvc_as_maximization = MIS_MVC_Problem(
            self.graph,
            problem_kind="mvc",
            objective_sense="maximize",
        )

        self.assertEqual(mis_as_minimization.objective_sense.value, "minimize")
        self.assertEqual(mvc_as_maximization.objective_sense.value, "maximize")

    def test_weighted_objective_and_selected_nodes(self):
        weights = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
        problem = MIS_MVC_Problem(
            self.graph,
            problem_kind="mis",
            weights=weights,
        )

        self.assertEqual(problem.objective_value("1010"), 4.0)
        self.assertEqual(problem.selected_nodes("1010"), [0, 2])

    def test_problem_kind_selects_feasibility_rule(self):
        mis = MIS_MVC_Problem(self.graph, problem_kind="mis")
        mvc = MIS_MVC_Problem(self.graph, problem_kind="mvc")

        self.assertTrue(mis.isFeasible("1010"))
        self.assertFalse(mis.isFeasible("1100"))
        self.assertTrue(mvc.isFeasible("1010"))
        self.assertFalse(mvc.isFeasible("1000"))

    def test_rejects_invalid_bitstrings(self):
        problem = MIS_MVC_Problem(self.graph, problem_kind="mis")

        with self.assertRaises(ValueError):
            problem.objective_value("101")
        with self.assertRaises(ValueError):
            problem.isFeasible("10x0")
        with self.assertRaises(TypeError):
            problem.selected_nodes([1, 0, 1, 0])

    def test_phase_circuit_matches_energy(self):
        weights = [1.0, 2.0, 0.5, 1.5]
        for problem_kind in ("mis", "mvc"):
            with self.subTest(problem_kind=problem_kind):
                problem = MIS_MVC_Problem(
                    self.graph,
                    problem_kind=problem_kind,
                    weights=weights,
                )
                problem.create_circuit()
                ok, report = problem.validate_circuit()
                self.assertTrue(ok, report)

    def test_default_mis_and_mvc_phase_signs_are_opposite(self):
        mis = MIS_MVC_Problem(self.graph, problem_kind="mis")
        mvc = MIS_MVC_Problem(self.graph, problem_kind="mvc")
        mis.create_circuit()
        mvc.create_circuit()

        gamma_mis = next(iter(mis.circuit.parameters))
        gamma_mvc = next(iter(mvc.circuit.parameters))
        u_mis = Operator(mis.circuit.assign_parameters({gamma_mis: 0.37})).data
        u_mvc = Operator(mvc.circuit.assign_parameters({gamma_mvc: 0.37})).data

        np.testing.assert_allclose(u_mvc, u_mis.conj(), atol=1e-10)


class TestInitialState(unittest.TestCase):
    def setUp(self):
        self.graph = nx.path_graph(4)

    def assert_feasible_support(self, initial_state, problem):
        initial_state.create_circuit()
        state = Statevector.from_instruction(initial_state.circuit)
        nonzero_amplitudes = []

        for index, amplitude in enumerate(state.data):
            bitstring = q0_first(index, problem.N_qubits)
            if problem.isFeasible(bitstring):
                if abs(amplitude) > 1e-10:
                    nonzero_amplitudes.append(amplitude)
            else:
                self.assertAlmostEqual(abs(amplitude), 0.0)

        self.assertTrue(nonzero_amplitudes)
        return nonzero_amplitudes

    def test_rejects_invalid_problem_kind(self):
        with self.assertRaises(TypeError):
            MIS_MVC_InitialState(self.graph, problem_kind="invalid")

    def test_degree_order_and_node_specific_angles(self):
        angles = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
        initial = MIS_MVC_InitialState(
            self.graph,
            problem_kind="mis",
            angle=angles,
        )

        self.assertEqual(initial.vertex_order, (1, 2, 0, 3))
        np.testing.assert_allclose(initial.angles, [0.1, 0.2, 0.3, 0.4])

    def test_scalar_angle_is_broadcast(self):
        initial = MIS_MVC_InitialState(
            self.graph,
            problem_kind="mis",
            angle=0.23,
        )

        np.testing.assert_allclose(initial.angles, [0.23] * 4)

    def test_rejects_incompatible_qubit_count(self):
        initial = MIS_MVC_InitialState(self.graph, problem_kind="mis")

        with self.assertRaises(ValueError):
            initial.setNumQubits(5)

    def test_mis_and_mvc_states_have_only_feasible_support(self):
        for problem_kind in ("mis", "mvc"):
            with self.subTest(problem_kind=problem_kind):
                initial = MIS_MVC_InitialState(
                    self.graph,
                    problem_kind=problem_kind,
                    angle=np.pi / 5,
                )
                problem = MIS_MVC_Problem(
                    self.graph,
                    problem_kind=problem_kind,
                )
                self.assert_feasible_support(initial, problem)

    def test_phase_correction_aligns_nonzero_amplitudes(self):
        for problem_kind in ("mis", "mvc"):
            with self.subTest(problem_kind=problem_kind):
                initial = MIS_MVC_InitialState(
                    self.graph,
                    problem_kind=problem_kind,
                    angle=np.pi / 5,
                    phase_correct=True,
                )
                problem = MIS_MVC_Problem(
                    self.graph,
                    problem_kind=problem_kind,
                )
                amplitudes = self.assert_feasible_support(initial, problem)
                reference_phase = amplitudes[0] / abs(amplitudes[0])

                for amplitude in amplitudes:
                    aligned = amplitude / reference_phase
                    self.assertAlmostEqual(aligned.imag, 0.0)
                    self.assertGreaterEqual(aligned.real, -1e-10)


class TestLXMixer(unittest.TestCase):
    def setUp(self):
        self.graph = nx.path_graph(4)

    def assert_preserves_feasibility(self, mixer, problem):
        mixer.create_circuit()
        bound = mixer.circuit.assign_parameters(
            {parameter: 0.31 for parameter in mixer.circuit.parameters}
        )
        unitary = Operator(bound).data
        n = problem.N_qubits

        for column in range(2**n):
            source = q0_first(column, n)
            if not problem.isFeasible(source):
                continue
            for row in range(2**n):
                target = q0_first(row, n)
                if not problem.isFeasible(target):
                    self.assertAlmostEqual(abs(unitary[row, column]), 0.0)

    def test_rejects_invalid_problem_kind(self):
        with self.assertRaises(TypeError):
            VertexSubsetLXMixer(self.graph, problem_kind="invalid")

    def test_degree_order_and_parameter_order(self):
        mixer = VertexSubsetLXMixer(
            self.graph,
            problem_kind="mis",
            multi_angle=True,
        )

        self.assertEqual(mixer.vertex_order, (1, 2, 0, 3))
        self.assertEqual(mixer.parameter_nodes, (1, 2, 0, 3))
        self.assertEqual(mixer.get_num_parameters(), 4)
        self.assertEqual(
            [parameter.name for parameter in mixer.mixer_params],
            ["x_beta_0", "x_beta_1", "x_beta_2", "x_beta_3"],
        )

        mixer.create_circuit()
        targets = [
            mixer.circuit.find_bit(instruction.qubits[-1]).index
            for instruction in mixer.circuit.data
        ]
        self.assertEqual(targets, [1, 2, 0, 3])

    def test_shared_angle_mode_has_one_parameter(self):
        mixer = VertexSubsetLXMixer(
            self.graph,
            problem_kind="mvc",
            multi_angle=False,
        )
        mixer.create_circuit()

        self.assertEqual(mixer.get_num_parameters(), 1)
        self.assertEqual(len(mixer.circuit.parameters), 1)
        self.assertEqual(mixer.mixer_params[0].name, "x_beta")

    def test_rejects_incompatible_qubit_count(self):
        mixer = VertexSubsetLXMixer(self.graph, problem_kind="mis")
        mixer.setNumQubits(5)

        with self.assertRaises(ValueError):
            mixer.create_circuit()

    def test_mis_and_mvc_mixers_preserve_feasibility(self):
        for problem_kind in ("mis", "mvc"):
            with self.subTest(problem_kind=problem_kind):
                mixer = VertexSubsetLXMixer(
                    self.graph,
                    problem_kind=problem_kind,
                    multi_angle=True,
                )
                problem = MIS_MVC_Problem(
                    self.graph,
                    problem_kind=problem_kind,
                )
                self.assert_preserves_feasibility(mixer, problem)


if __name__ == "__main__":
    unittest.main()
