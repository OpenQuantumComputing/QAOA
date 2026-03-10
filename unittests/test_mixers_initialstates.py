"""
Unit tests for mixers and initial states not covered by existing tests.

Covers:
- Mixers: Grover, XMultiAngle, XOrbit, MaxKCutGrover, MaxKCutLX, XYTensor
- Initial states: Dicke, StateVector, LessThanK, MaxKCutFeasible, PlusParameterized, Tensor
"""

import unittest

import numpy as np
import networkx as nx

from qiskit import QuantumCircuit


# ---------------------------------------------------------------------------
# Mixer tests
# ---------------------------------------------------------------------------


class TestGroverMixer(unittest.TestCase):
    """Tests for the Grover mixer."""

    def test_grover_mixer_with_plus_subcircuit(self):
        from qaoa.mixers import Grover
        from qaoa.initialstates import Plus

        mixer = Grover(subcircuit=Plus())
        mixer.setNumQubits(3)
        mixer.create_circuit()
        self.assertIsInstance(mixer.circuit, QuantumCircuit)

    def test_grover_mixer_1_qubit(self):
        from qaoa.mixers import Grover
        from qaoa.initialstates import Plus

        mixer = Grover(subcircuit=Plus())
        mixer.setNumQubits(1)
        mixer.create_circuit()
        self.assertIsInstance(mixer.circuit, QuantumCircuit)

    def test_grover_mixer_circuit_has_params(self):
        from qaoa.mixers import Grover
        from qaoa.initialstates import Plus

        mixer = Grover(subcircuit=Plus())
        mixer.setNumQubits(2)
        mixer.create_circuit()
        self.assertGreater(len(mixer.circuit.parameters), 0)


class TestXMultiAngleMixer(unittest.TestCase):
    """Tests for the XMultiAngle mixer."""

    def test_xmultiangle_num_parameters(self):
        from qaoa.mixers import XMultiAngle

        mixer = XMultiAngle()
        mixer.setNumQubits(4)
        self.assertEqual(mixer.get_num_parameters(), 4)

    def test_xmultiangle_circuit_correct_params(self):
        from qaoa.mixers import XMultiAngle

        mixer = XMultiAngle()
        mixer.setNumQubits(3)
        mixer.create_circuit()
        self.assertIsInstance(mixer.circuit, QuantumCircuit)
        self.assertEqual(len(mixer.circuit.parameters), 3)

    def test_xmultiangle_different_params_per_qubit(self):
        from qaoa.mixers import XMultiAngle

        mixer = XMultiAngle()
        mixer.setNumQubits(2)
        mixer.create_circuit()
        param_names = [p.name for p in mixer.circuit.parameters]
        # All parameter names should be distinct
        self.assertEqual(len(param_names), len(set(param_names)))


class TestXOrbitMixer(unittest.TestCase):
    """Tests for the XOrbit mixer."""

    def test_xorbit_symmetric_graph(self):
        """On K3, all nodes are in the same orbit → 1 parameter."""
        from qaoa.mixers import XOrbit

        G = nx.complete_graph(3)
        mixer = XOrbit(G)
        mixer.setNumQubits(3)
        mixer.create_circuit()
        self.assertIsInstance(mixer.circuit, QuantumCircuit)
        # K3 is vertex-transitive → all qubits share 1 beta
        self.assertEqual(mixer.get_num_parameters(), 1)

    def test_xorbit_asymmetric_graph(self):
        """On a path graph, the two endpoints form one orbit, the middle another."""
        from qaoa.mixers import XOrbit

        G = nx.path_graph(3)
        mixer = XOrbit(G)
        mixer.setNumQubits(3)
        mixer.create_circuit()
        self.assertIsInstance(mixer.circuit, QuantumCircuit)
        # Path graph P3: nodes 0 and 2 are symmetric, node 1 is distinct → 2 orbits
        self.assertEqual(mixer.get_num_parameters(), 2)

    def test_xorbit_circuit_has_parameters(self):
        from qaoa.mixers import XOrbit

        G = nx.path_graph(4)
        mixer = XOrbit(G)
        mixer.setNumQubits(4)
        mixer.create_circuit()
        self.assertGreater(len(mixer.circuit.parameters), 0)


class TestMaxKCutGroverMixer(unittest.TestCase):
    """Tests for the MaxKCutGrover mixer."""

    def test_maxkcut_grover_k3_binary(self):
        from qaoa.mixers import MaxKCutGrover

        mixer = MaxKCutGrover(
            k_cuts=3, problem_encoding="binary",
            color_encoding="LessThanK", tensorized=False
        )
        mixer.setNumQubits(4)  # 2 nodes × 2 bits
        mixer.create_circuit()
        self.assertIsInstance(mixer.circuit, QuantumCircuit)

    def test_maxkcut_grover_invalid_k(self):
        from qaoa.mixers import MaxKCutGrover

        with self.assertRaises(ValueError):
            MaxKCutGrover(
                k_cuts=9, problem_encoding="binary",
                color_encoding="LessThanK", tensorized=False
            )

    def test_maxkcut_grover_invalid_encoding(self):
        from qaoa.mixers import MaxKCutGrover

        with self.assertRaises(ValueError):
            MaxKCutGrover(
                k_cuts=3, problem_encoding="invalid",
                color_encoding="LessThanK", tensorized=False
            )


class TestMaxKCutLXMixer(unittest.TestCase):
    """Tests for the MaxKCutLX mixer."""

    def test_maxkcut_lx_k3(self):
        from qaoa.mixers import MaxKCutLX

        mixer = MaxKCutLX(k_cuts=3, color_encoding="LessThanK")
        mixer.setNumQubits(4)  # 2 nodes × 2 bits
        mixer.create_circuit()
        self.assertIsInstance(mixer.circuit, QuantumCircuit)

    def test_maxkcut_lx_k3_ring(self):
        from qaoa.mixers import MaxKCutLX

        mixer = MaxKCutLX(k_cuts=3, color_encoding="LessThanK", topology="ring")
        mixer.setNumQubits(4)
        mixer.create_circuit()
        self.assertIsInstance(mixer.circuit, QuantumCircuit)

    def test_maxkcut_lx_power_of_two_raises(self):
        from qaoa.mixers import MaxKCutLX

        with self.assertRaises(ValueError):
            MaxKCutLX(k_cuts=4, color_encoding="LessThanK")

    def test_maxkcut_lx_empty_encoding_raises(self):
        from qaoa.mixers import MaxKCutLX

        with self.assertRaises(ValueError):
            MaxKCutLX(k_cuts=3, color_encoding="")


class TestXYMixerTopologies(unittest.TestCase):
    """Additional XY mixer tests for ring/chain topologies."""

    def test_xy_ring_topology(self):
        from qaoa.mixers import XY

        mixer = XY(case="ring")
        mixer.setNumQubits(4)
        mixer.create_circuit()
        self.assertIsInstance(mixer.circuit, QuantumCircuit)
        # Ring: 4 pairs for 4 qubits
        self.assertEqual(len(mixer.topology), 4)

    def test_xy_chain_topology(self):
        from qaoa.mixers import XY

        mixer = XY(case="chain")
        mixer.setNumQubits(4)
        mixer.create_circuit()
        self.assertIsInstance(mixer.circuit, QuantumCircuit)
        # Chain: 3 pairs for 4 qubits
        self.assertEqual(len(mixer.topology), 3)

    def test_xy_explicit_topology(self):
        from qaoa.mixers import XY

        mixer = XY(topology=[[0, 1], [2, 3]])
        mixer.setNumQubits(4)
        mixer.create_circuit()
        self.assertIsInstance(mixer.circuit, QuantumCircuit)

    def test_xy_generate_pairs_ring(self):
        from qaoa.mixers import XY

        pairs = XY.generate_pairs(4, case="ring")
        self.assertEqual(len(pairs), 4)
        self.assertIn([3, 0], pairs)

    def test_xy_generate_pairs_chain(self):
        from qaoa.mixers import XY

        pairs = XY.generate_pairs(3, case="chain")
        self.assertEqual(len(pairs), 2)
        self.assertNotIn([2, 0], pairs)


# ---------------------------------------------------------------------------
# Initial state tests
# ---------------------------------------------------------------------------


class TestDickeInitialState(unittest.TestCase):
    """Tests for the Dicke initial state."""

    def test_dicke_k1_n3(self):
        from qaoa.initialstates import Dicke

        init = Dicke(k=1)
        init.setNumQubits(3)
        init.create_circuit()
        self.assertIsInstance(init.circuit, QuantumCircuit)
        self.assertEqual(init.circuit.num_qubits, 3)

    def test_dicke_k2_n4(self):
        from qaoa.initialstates import Dicke

        init = Dicke(k=2)
        init.setNumQubits(4)
        init.create_circuit()
        self.assertIsInstance(init.circuit, QuantumCircuit)

    def test_dicke_k3_n5(self):
        from qaoa.initialstates import Dicke

        init = Dicke(k=3)
        init.setNumQubits(5)
        init.create_circuit()
        self.assertIsInstance(init.circuit, QuantumCircuit)


class TestStateVectorInitialState(unittest.TestCase):
    """Tests for the StateVector initial state."""

    def test_statevector_uniform_superposition(self):
        from qaoa.initialstates import StateVector

        # Uniform superposition of 2-qubit system
        sv = np.ones(4) / 2.0
        init = StateVector(statevector=sv)
        init.setNumQubits(2)
        init.create_circuit()
        self.assertIsInstance(init.circuit, QuantumCircuit)
        self.assertEqual(init.circuit.num_qubits, 2)

    def test_statevector_computational_basis(self):
        from qaoa.initialstates import StateVector

        # |01⟩ state
        sv = np.array([0.0, 1.0, 0.0, 0.0])
        init = StateVector(statevector=sv)
        init.setNumQubits(2)
        init.create_circuit()
        self.assertIsInstance(init.circuit, QuantumCircuit)


class TestLessThanKInitialState(unittest.TestCase):
    """Tests for the LessThanK initial state."""

    def test_lessthank_k2(self):
        from qaoa.initialstates import LessThanK

        init = LessThanK(k=2)
        init.create_circuit()
        self.assertIsInstance(init.circuit, QuantumCircuit)

    def test_lessthank_k3(self):
        from qaoa.initialstates import LessThanK

        init = LessThanK(k=3)
        init.create_circuit()
        self.assertIsInstance(init.circuit, QuantumCircuit)

    def test_lessthank_k5(self):
        from qaoa.initialstates import LessThanK

        init = LessThanK(k=5)
        init.create_circuit()
        self.assertIsInstance(init.circuit, QuantumCircuit)

    def test_lessthank_k6(self):
        from qaoa.initialstates import LessThanK

        init = LessThanK(k=6)
        init.create_circuit()
        self.assertIsInstance(init.circuit, QuantumCircuit)

    def test_lessthank_k7(self):
        from qaoa.initialstates import LessThanK

        init = LessThanK(k=7)
        init.create_circuit()
        self.assertIsInstance(init.circuit, QuantumCircuit)

    def test_lessthank_invalid_k_raises(self):
        from qaoa.initialstates import LessThanK

        with self.assertRaises(ValueError):
            LessThanK(k=9)


class TestMaxKCutFeasibleInitialState(unittest.TestCase):
    """Tests for the MaxKCutFeasible initial state."""

    def test_maxkcut_feasible_k3_binary(self):
        from qaoa.initialstates import MaxKCutFeasible

        init = MaxKCutFeasible(k_cuts=3, problem_encoding="binary",
                                color_encoding="LessThanK")
        init.setNumQubits(4)  # 2 nodes × 2 bits
        init.create_circuit()
        self.assertIsInstance(init.circuit, QuantumCircuit)

    def test_maxkcut_feasible_invalid_encoding_raises(self):
        from qaoa.initialstates import MaxKCutFeasible

        with self.assertRaises(ValueError):
            MaxKCutFeasible(k_cuts=3, problem_encoding="invalid")


class TestPlusParameterizedInitialState(unittest.TestCase):
    """Tests for the PlusParameterized initial state."""

    def test_plus_parameterized_default(self):
        from qaoa.initialstates import PlusParameterized

        init = PlusParameterized()
        init.setNumQubits(3)
        init.create_circuit()
        self.assertIsInstance(init.circuit, QuantumCircuit)
        # Default: N_qubits phase parameters
        self.assertEqual(len(init.circuit.parameters), 3)

    def test_plus_parameterized_custom_num_phases(self):
        from qaoa.initialstates import PlusParameterized

        init = PlusParameterized(num_phases=2)
        init.setNumQubits(4)
        init.create_circuit()
        self.assertIsInstance(init.circuit, QuantumCircuit)
        # 2 phase parameters
        self.assertEqual(len(init.circuit.parameters), 2)

    def test_plus_parameterized_num_parameters(self):
        from qaoa.initialstates import PlusParameterized

        init = PlusParameterized(num_phases=3)
        init.setNumQubits(5)
        self.assertEqual(init.get_num_parameters(), 3)


class TestTensorInitialState(unittest.TestCase):
    """Tests for the Tensor initial state."""

    def test_tensor_plus(self):
        from qaoa.initialstates import Tensor, Plus

        # Create a 4-qubit tensor product of 2-qubit Plus states
        plus = Plus()
        plus.setNumQubits(2)
        init = Tensor(subcircuit=plus, num=2)
        init.create_circuit()
        self.assertIsInstance(init.circuit, QuantumCircuit)
        self.assertEqual(init.circuit.num_qubits, 4)


if __name__ == "__main__":
    unittest.main()
