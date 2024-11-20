import unittest
import sys
import networkx as nx
import numpy as np

from qiskit_aer import Aer
from qiskit.quantum_info import Statevector

sys.path.append("../")

from qaoa.initialstates import MaxKCutFeasible


class TestMaxKCutFeasibleInitialstate(unittest.TestCase):
    def __init__(self, methodname):
        super().__init__(methodname)

        V = np.arange(0, 1, 1)
        E = []

        self.G = nx.Graph()
        self.G.add_nodes_from(V)
        self.G.add_weighted_edges_from(E)

    def test_feasible_initialstate_binary(self):
        """
        Test that MaxKCutFeasible (case: binary) prepares the correct initialstate
        for all k in [3, 5, 6, 7]
        """
        coen = ["LessThanK", "Dicke1_2"]
        for color_encoding in coen:
            for k in [3, 5, 6, 7]:
                if (color_encoding == "Dicke1_2") and (k != 6):
                    continue

                k_bits = int(np.ceil(np.log2(k)))
                initialstate = MaxKCutFeasible(
                    k, "binary", color_encoding=color_encoding
                )
                initialstate.setNumQubits(k_bits)
                initialstate.create_circuit()
                circuit = initialstate.circuit

                statevector = Statevector(circuit)
                sample_counts = statevector.sample_counts(shots=100000)
                for string in sample_counts:
                    string = string[::-1]
                    self.assertTrue(string not in initialstate.infeasible)

    def test_feasible_initialstate_onehot(self):
        """
        Test that MaxKCutFeasible (case: onehot) prepares the correct initialstate
        for all 2 <= k <= 8.
        """
        for k in range(2, 9):
            initialstate = MaxKCutFeasible(k, "onehot")
            initialstate.setNumQubits(k)
            initialstate.create_circuit()
            circuit = initialstate.circuit

            computed = Statevector(circuit)
            expected = np.zeros(2**k)
            for i in range(k):
                expected[2**i] = 1 / np.sqrt(k)
            equiv = np.allclose(computed, expected)
            msg = f"One-Hot: k = {k}. Expected: {expected}, computed: {computed}."
            self.assertTrue(equiv, msg)


if __name__ == "__main__":
    unittest.main()
