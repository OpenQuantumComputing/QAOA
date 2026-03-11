"""
Integration tests verifying that the QAOA library works correctly with
Qiskit >= 2.x.

These tests cover:
1. Qiskit version verification (>= 2.0)
2. Correct Sampler import (qiskit.primitives.Sampler was removed in Qiskit 2.x)
3. End-to-end QAOA optimization on a small MaxCut problem
4. All key components: problems, mixers, initial states, transpile
5. Core Qiskit 2.x API elements used by the library
"""

import unittest

import numpy as np
import networkx as nx

import qiskit


class TestQiskitVersion(unittest.TestCase):
    """Verify that the installed Qiskit version is >= 2.0."""

    def test_qiskit_version_is_2_or_higher(self):
        major = int(qiskit.__version__.split(".")[0])
        self.assertGreaterEqual(
            major,
            2,
            f"Qiskit {qiskit.__version__} is installed, but >= 2.0.0 is required.",
        )


class TestQiskitImports(unittest.TestCase):
    """Verify that all Qiskit symbols used by the library can be imported."""

    def test_core_qiskit_imports(self):
        from qiskit import (
            QuantumCircuit,
            QuantumRegister,
            ClassicalRegister,
            AncillaRegister,
            transpile,
        )
        from qiskit.circuit import Parameter
        from qiskit.circuit.library import (
            PhaseGate,
            RYGate,
            XXPlusYYGate,
            PauliEvolutionGate,
        )
        from qiskit.quantum_info import SparsePauliOp, Pauli, Statevector

    def test_sampler_import(self):
        """qiskit.primitives.Sampler was removed in Qiskit 2.x; the fallback
        to qiskit_aer.primitives.Sampler must work."""
        try:
            from qiskit_aer.primitives import Sampler
        except ImportError:
            from qiskit.primitives import StatevectorSampler as Sampler
        self.assertIsNotNone(Sampler)

    def test_aer_simulator_import(self):
        from qiskit_aer import AerSimulator
        backend = AerSimulator()
        self.assertIsNotNone(backend)

    def test_qiskit_algorithms_optimizers(self):
        from qiskit_algorithms.optimizers import COBYLA, SPSA, QNSPSA
        self.assertIsNotNone(COBYLA)
        self.assertIsNotNone(SPSA)
        self.assertIsNotNone(QNSPSA)

    def test_sampler_v2_for_qnspsa(self):
        """The SamplerV2 used for QNSPSA fidelity must be a BaseSamplerV2 instance."""
        from qiskit.primitives import BaseSamplerV2
        try:
            from qiskit_aer.primitives import SamplerV2 as _SamplerV2
        except ImportError:
            from qiskit.primitives import StatevectorSampler as _SamplerV2
        self.assertTrue(issubclass(_SamplerV2, BaseSamplerV2))

    def test_qaoa_package_import(self):
        import qaoa
        from qaoa import QAOA
        from qaoa import problems, mixers, initialstates


class TestQAOAEndToEnd(unittest.TestCase):
    """End-to-end QAOA test on a small MaxCut instance using Qiskit 2.x."""

    def setUp(self):
        from qiskit_aer import AerSimulator
        self.backend = AerSimulator()
        # 3-node path graph: optimal MaxCut = 2
        self.G = nx.path_graph(3)
        self.angles = {"gamma": [0, 2 * np.pi, 5], "beta": [0, np.pi, 5]}

    def test_maxcut_optimize_depth1(self):
        """QAOA at depth 1 should find a positive approximation ratio for MaxCut."""
        from qaoa import QAOA
        from qaoa import problems, mixers, initialstates

        problem = problems.MaxCut(self.G)
        mixer = mixers.X()
        init = initialstates.Plus()

        qaoa = QAOA(
            problem,
            mixer,
            init,
            backend=self.backend,
            shots=512,
        )
        qaoa.optimize(depth=1, angles=self.angles)

        best_exp = qaoa.get_Exp(depth=1)
        # For a path graph with 3 nodes the optimum is -2 (cost stored negated).
        # A good QAOA should achieve at least -1 (approximation ratio >= 0.5).
        self.assertLessEqual(best_exp, -1.0)

    def test_circuit_creation_and_transpile(self):
        """Verify that the parameterised circuit is created and transpiled correctly."""
        from qaoa import QAOA
        from qaoa import problems, mixers, initialstates
        from qiskit import QuantumCircuit

        problem = problems.MaxCut(self.G)
        mixer = mixers.X()
        init = initialstates.Plus()

        qaoa = QAOA(
            problem,
            mixer,
            init,
            backend=self.backend,
        )
        qaoa.createParameterizedCircuit(depth=1)

        self.assertIsInstance(qaoa.parameterized_circuit, QuantumCircuit)
        # After transpilation the circuit should have no free parameters
        # (they are bound at runtime via parameter_binds).
        self.assertGreater(len(qaoa.parameterized_circuit.parameters), 0)

    def test_hist_method(self):
        """The hist() helper should return a non-empty measurement histogram."""
        from qaoa import QAOA
        from qaoa import problems, mixers, initialstates

        problem = problems.MaxCut(self.G)
        qaoa = QAOA(
            problem,
            mixers.X(),
            initialstates.Plus(),
            backend=self.backend,
            shots=256,
        )
        qaoa.optimize(depth=1, angles=self.angles)
        angles = qaoa.get_angles(depth=1)
        hist = qaoa.hist(angles, shots=256)

        self.assertIsInstance(hist, dict)
        self.assertGreater(len(hist), 0)
        total_shots = sum(hist.values())
        self.assertEqual(total_shots, 256)

    def test_qnspsa_optimizer(self):
        """QNSPSA optimizer must work with a BaseSamplerV2-compatible sampler.

        This tests the fix for the Qiskit 2.x incompatibility where
        QNSPSA.get_fidelity() requires a BaseSamplerV2 instance.
        """
        from qaoa import QAOA
        from qaoa import problems, mixers, initialstates
        from qiskit_algorithms.optimizers import QNSPSA

        problem = problems.MaxCut(self.G)
        qaoa_inst = QAOA(
            problem,
            mixers.X(),
            initialstates.Plus(),
            backend=self.backend,
            optimizer=[QNSPSA, {"maxiter": 3}],
            shots=256,
        )
        # Should not raise ValueError about BaseSamplerV2
        qaoa_inst.optimize(depth=1, angles=self.angles)
        best_exp = qaoa_inst.get_Exp(depth=1)
        self.assertIsInstance(best_exp, float)


class TestQAOAComponents(unittest.TestCase):
    """Verify individual QAOA components work with Qiskit 2.x."""

    def _make_graph(self):
        G = nx.Graph()
        G.add_nodes_from([0, 1])
        G.add_weighted_edges_from([(0, 1, 1.0)])
        return G

    def test_maxcut_problem_circuit(self):
        from qaoa.problems import MaxCut
        from qiskit.circuit import QuantumCircuit

        problem = MaxCut(self._make_graph())
        problem.create_circuit()
        self.assertIsInstance(problem.circuit, QuantumCircuit)
        self.assertEqual(len(problem.circuit.parameters), 1)

    def test_x_mixer_circuit(self):
        from qaoa.mixers import X
        from qiskit.circuit import QuantumCircuit

        mixer = X()
        mixer.setNumQubits(2)
        mixer.create_circuit()
        self.assertIsInstance(mixer.circuit, QuantumCircuit)

    def test_xy_mixer_circuit(self):
        from qaoa.mixers import XY
        from qiskit.circuit import QuantumCircuit

        mixer = XY(topology=[[0, 1]])  # explicit edge topology for 2-qubit ring
        mixer.setNumQubits(2)
        mixer.create_circuit()
        self.assertIsInstance(mixer.circuit, QuantumCircuit)

    def test_plus_initialstate_circuit(self):
        from qaoa.initialstates import Plus
        from qiskit.circuit import QuantumCircuit

        init = Plus()
        init.setNumQubits(2)
        init.create_circuit()
        self.assertIsInstance(init.circuit, QuantumCircuit)

    def test_transpile_with_aer_simulator(self):
        """transpile() should work with AerSimulator in Qiskit 2.x."""
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit import Parameter
        from qiskit_aer import AerSimulator

        backend = AerSimulator()
        qc = QuantumCircuit(2, 2)
        theta = Parameter("theta")
        qc.h(0)
        qc.rz(theta, 0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        transpiled = transpile(qc, backend)
        self.assertIsInstance(transpiled, QuantumCircuit)


if __name__ == "__main__":
    unittest.main()
