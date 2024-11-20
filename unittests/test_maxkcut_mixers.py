import unittest
import sys
import networkx as nx
import numpy as np

from qiskit.quantum_info import Statevector
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister

sys.path.append("../")

from qaoa.mixers import MaxKCutGrover, MaxKCutLX
from qaoa.initialstates import MaxKCutFeasible


class TestFeasibleOutputsFromMixers(unittest.TestCase):
    def __init__(self, methodname):
        super().__init__(methodname)

        V = np.arange(0, 1, 1)
        E = []

        self.G = nx.Graph()
        self.G.add_nodes_from(V)
        self.G.add_weighted_edges_from(E)

    def test_LXmixer_binary(self):
        for mixertype in ["LX", "Grover"]:
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

                    if mixertype == "LX":
                        mixer = MaxKCutLX(k, color_encoding=color_encoding)
                    else:
                        mixer = MaxKCutGrover(
                            k,
                            color_encoding=color_encoding,
                            problem_encoding="binary",
                            tensorized=False,
                        )
                    mixer.setNumQubits(k_bits)
                    mixer.create_circuit()

                    circuit = initialstate.circuit
                    circuit.compose(mixer.circuit, inplace=True)

                    circuit = circuit.assign_parameters(
                        {circuit.parameters[0]: 0.5912847},
                        inplace=False,
                    )

                    statevector = Statevector(circuit)
                    sample_counts = statevector.sample_counts(shots=100000)
                    for string in sample_counts:
                        string = string[::-1]
                        self.assertTrue(string not in initialstate.infeasible)


if __name__ == "__main__":
    unittest.main()
