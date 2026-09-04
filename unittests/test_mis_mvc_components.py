"""Unit tests for the uploaded unified MIS/MVC components."""

import unittest

import networkx as nx
import numpy as np
from qiskit.quantum_info import Operator, Statevector

from qaoa.initialstates import MIS_MVC_InitialState, MIS_MVC_ParameterizedInitialState
from qaoa.mixers import VertexSubsetLXMixer, VertexSubsetOrbitLXMixer
from qaoa.problems import MIS_MVC_MultiAngle, MIS_MVC_Orbit, MIS_MVC_Problem
from qaoa.utils.vertex_subset import canonical_graph, degree_buckets, mis_mvc_warm_start_angle


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

    def test_multiangle_problem_reduces_to_shared_problem(self):
        multiangle = MIS_MVC_MultiAngle(self.graph, problem_kind="mis")
        multiangle.create_circuit()
        shared = MIS_MVC_Problem(self.graph, problem_kind="mis")
        shared.create_circuit()

        t = 0.29
        multiangle_bound = multiangle.circuit.assign_parameters(
            {parameter: t for parameter in multiangle.circuit.parameters}
        )
        shared_gamma = next(iter(shared.circuit.parameters))
        shared_bound = shared.circuit.assign_parameters({shared_gamma: t})

        np.testing.assert_allclose(
            Operator(multiangle_bound).data,
            Operator(shared_bound).data,
            atol=1e-10,
        )


class TestOrbitProblem(unittest.TestCase):
    def setUp(self):
        self.graph = nx.path_graph(4)

    def test_orbit_computation_and_parameter_count(self):
        for problem_kind in ("mis", "mvc"):
            with self.subTest(problem_kind=problem_kind):
                problem = MIS_MVC_Orbit(self.graph, problem_kind=problem_kind)
                self.assertEqual(problem.get_num_parameters(), 2)
                self.assertEqual(
                    {frozenset(orbit) for orbit in problem.orbits},
                    {frozenset((0, 3)), frozenset((1, 2))},
                )

    def test_phase_circuit_matches_energy(self):
        for problem_kind in ("mis", "mvc"):
            with self.subTest(problem_kind=problem_kind):
                orbit_problem = MIS_MVC_Orbit(self.graph, problem_kind=problem_kind)
                orbit_problem.create_circuit()
                base_problem = MIS_MVC_Problem(self.graph, problem_kind=problem_kind)
                base_problem.create_circuit()

                t = 0.37
                orbit_bound = orbit_problem.circuit.assign_parameters(
                    {parameter: t for parameter in orbit_problem.circuit.parameters}
                )
                base_gamma = next(iter(base_problem.circuit.parameters))
                base_bound = base_problem.circuit.assign_parameters({base_gamma: t})

                np.testing.assert_allclose(
                    Operator(orbit_bound).data,
                    Operator(base_bound).data,
                    atol=1e-10,
                )

    def test_symmetry_related_vertices_share_gamma_parameter(self):
        problem = MIS_MVC_Orbit(self.graph, problem_kind="mis")
        problem.create_circuit()

        params_by_target = {
            problem.circuit.find_bit(instruction.qubits[0]).index: str(
                instruction.operation.params[0]
            )
            for instruction in problem.circuit.data
        }
        self.assertEqual(params_by_target[0], params_by_target[3])
        self.assertEqual(params_by_target[1], params_by_target[2])
        self.assertNotEqual(params_by_target[0], params_by_target[1])

    def test_single_orbit_reduction(self):
        graph = nx.complete_graph(4)
        problem = MIS_MVC_Orbit(graph, problem_kind="mvc")
        problem.create_circuit()

        self.assertEqual(problem.get_num_parameters(), 1)
        self.assertEqual(len(problem.circuit.parameters), 1)
        self.assertEqual(problem.parameter_nodes, (0,))


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


class TestOrbitLXMixer(TestLXMixer):
    def test_orbit_computation_and_parameter_count(self):
        for problem_kind in ("mis", "mvc"):
            with self.subTest(problem_kind=problem_kind):
                mixer = VertexSubsetOrbitLXMixer(self.graph, problem_kind=problem_kind)
                self.assertEqual(mixer.vertex_order, (1, 2, 0, 3))
                self.assertEqual(mixer.get_num_parameters(), 2)
                self.assertEqual(
                    {frozenset(orbit) for orbit in mixer.orbits},
                    {frozenset((0, 3)), frozenset((1, 2))},
                )
                self.assertEqual(mixer.parameter_nodes, (0, 1))
                self.assertEqual(
                    [parameter.name for parameter in mixer.mixer_params],
                    ["x_beta_orbit_0", "x_beta_orbit_1"],
                )

    def test_orbit_mixers_preserve_feasibility(self):
        for problem_kind in ("mis", "mvc"):
            with self.subTest(problem_kind=problem_kind):
                mixer = VertexSubsetOrbitLXMixer(self.graph, problem_kind=problem_kind)
                problem = MIS_MVC_Problem(self.graph, problem_kind=problem_kind)
                self.assert_preserves_feasibility(mixer, problem)

    def test_symmetry_related_vertices_share_beta_parameter(self):
        mixer = VertexSubsetOrbitLXMixer(self.graph, problem_kind="mis")
        mixer.create_circuit()

        params_by_target = {
            mixer.circuit.find_bit(instruction.qubits[-1]).index: str(
                instruction.operation.base_gate.params[0]
            )
            for instruction in mixer.circuit.data
        }
        self.assertEqual(params_by_target[0], params_by_target[3])
        self.assertEqual(params_by_target[1], params_by_target[2])
        self.assertNotEqual(params_by_target[0], params_by_target[1])

    def test_single_orbit_reduction(self):
        graph = nx.complete_graph(4)
        mixer = VertexSubsetOrbitLXMixer(graph, problem_kind="mvc")
        mixer.create_circuit()

        self.assertEqual(mixer.get_num_parameters(), 1)
        self.assertEqual(len(mixer.circuit.parameters), 1)
        self.assertEqual(mixer.parameter_nodes, (0,))


class TestDegreeBuckets(unittest.TestCase):
    def test_path_graph_buckets(self):
        # path_graph(4): nodes 0,3 have degree 1; nodes 1,2 have degree 2
        canonical, _ = canonical_graph(nx.path_graph(4))
        buckets = degree_buckets(canonical)
        self.assertEqual(buckets, [[0, 3], [1, 2]])

    def test_complete_graph_single_bucket(self):
        canonical, _ = canonical_graph(nx.complete_graph(4))
        buckets = degree_buckets(canonical)
        self.assertEqual(len(buckets), 1)
        self.assertEqual(sorted(buckets[0]), [0, 1, 2, 3])


class TestMisVcWarmStartAngle(unittest.TestCase):
    def test_uniform_returns_scalar(self):
        angle = mis_mvc_warm_start_angle(nx.path_graph(4), grouping="uniform")
        self.assertIsInstance(angle, float)
        self.assertGreater(angle, 0)
        self.assertLess(angle, np.pi / 2)

    def test_degree_returns_list(self):
        angles = mis_mvc_warm_start_angle(nx.path_graph(4), grouping="degree")
        self.assertEqual(len(angles), 2)  # two distinct degrees in path_graph(4)

    def test_per_vertex_returns_list_of_length_n(self):
        G = nx.path_graph(4)
        angles = mis_mvc_warm_start_angle(G, grouping="per_vertex")
        self.assertEqual(len(angles), G.number_of_nodes())

    def test_isolated_vertex_defaults_to_pi_over_4(self):
        G = nx.Graph()
        G.add_node(0)
        angle = mis_mvc_warm_start_angle(G, grouping="uniform")
        self.assertAlmostEqual(angle, np.pi / 4)

    def test_invalid_grouping_raises(self):
        with self.assertRaises(ValueError):
            mis_mvc_warm_start_angle(nx.path_graph(4), grouping="bad")


class TestParameterizedInitialState(unittest.TestCase):
    def setUp(self):
        self.graph = nx.path_graph(4)

    def _bound_statevector(self, initial_state, angle_val):
        """Bind all parameters to *angle_val* and return the Statevector."""
        initial_state.create_circuit()
        bound = initial_state.circuit.assign_parameters(
            {p: angle_val for p in initial_state.circuit.parameters}
        )
        return Statevector.from_instruction(bound)

    def test_rejects_invalid_problem_kind(self):
        with self.assertRaises(TypeError):
            MIS_MVC_ParameterizedInitialState(self.graph, problem_kind="bad")

    def test_rejects_invalid_grouping(self):
        with self.assertRaises(ValueError):
            MIS_MVC_ParameterizedInitialState(
                self.graph, problem_kind="mis", grouping="bad"
            )

    def test_uniform_has_one_parameter(self):
        init = MIS_MVC_ParameterizedInitialState(
            self.graph, problem_kind="mis", grouping="uniform"
        )
        self.assertEqual(init.get_num_parameters(), 1)
        init.create_circuit()
        self.assertEqual(len(init.circuit.parameters), 1)

    def test_degree_parameter_count(self):
        # path_graph(4) has 2 distinct degrees
        init = MIS_MVC_ParameterizedInitialState(
            self.graph, problem_kind="mis", grouping="degree"
        )
        self.assertEqual(init.get_num_parameters(), 2)
        init.create_circuit()
        self.assertEqual(len(init.circuit.parameters), 2)

    def test_per_vertex_parameter_count(self):
        init = MIS_MVC_ParameterizedInitialState(
            self.graph, problem_kind="mis", grouping="per_vertex"
        )
        self.assertEqual(init.get_num_parameters(), self.graph.number_of_nodes())

    def test_rejects_incompatible_qubit_count(self):
        init = MIS_MVC_ParameterizedInitialState(self.graph, problem_kind="mis")
        with self.assertRaises(ValueError):
            init.setNumQubits(5)

    def test_mis_feasible_support_uniform(self):
        problem = MIS_MVC_Problem(self.graph, problem_kind="mis")
        init = MIS_MVC_ParameterizedInitialState(
            self.graph, problem_kind="mis", grouping="uniform"
        )
        sv = self._bound_statevector(init, np.pi / 5)
        for idx, amp in enumerate(sv.data):
            bitstring = format(idx, f"0{problem.N_qubits}b")[::-1]
            if not problem.isFeasible(bitstring):
                self.assertAlmostEqual(abs(amp), 0.0)

    def test_mvc_feasible_support_degree(self):
        problem = MIS_MVC_Problem(self.graph, problem_kind="mvc")
        init = MIS_MVC_ParameterizedInitialState(
            self.graph, problem_kind="mvc", grouping="degree"
        )
        sv = self._bound_statevector(init, np.pi / 6)
        for idx, amp in enumerate(sv.data):
            bitstring = format(idx, f"0{problem.N_qubits}b")[::-1]
            if not problem.isFeasible(bitstring):
                self.assertAlmostEqual(abs(amp), 0.0)

    def test_uniform_matches_fixed_angle_initialstate(self):
        """When bound to angle θ, parameterized uniform == fixed-angle variant."""
        theta = np.pi / 5
        fixed = MIS_MVC_InitialState(
            self.graph, problem_kind="mis", angle=theta, phase_correct=True
        )
        fixed.create_circuit()
        sv_fixed = Statevector.from_instruction(fixed.circuit)

        param_init = MIS_MVC_ParameterizedInitialState(
            self.graph, problem_kind="mis", grouping="uniform", phase_correct=True
        )
        sv_param = self._bound_statevector(param_init, theta)

        # Both circuits are identical in structure, so amplitudes (including
        # phases) should match exactly, not just in absolute value.
        np.testing.assert_allclose(sv_param.data, sv_fixed.data, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
