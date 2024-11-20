import numpy as np
import networkx as nx
import unittest

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.quantum_info import Statevector

import sys

sys.path.append("../")
from qaoa.problems import MaxKCutBinaryPowerOfTwo, MaxKCutBinaryFullH


class TestMaxKCutBinaryProblem(unittest.TestCase):
    def __init__(self, methodname):
        super().__init__(methodname)

        V = np.arange(0, 2, 1)
        E = [(0, 1, 1.0)]

        self.G = nx.Graph()
        self.G.add_nodes_from(V)
        self.G.add_weighted_edges_from(E)

    def stateVectorToBitstring(self, sv):
        probabilities = np.abs(sv) ** 2
        # Check if the array has exactly one `1` and the rest are `0`s
        is_comp_basis_state = np.count_nonzero(probabilities) == 1 and np.all(
            np.logical_or(probabilities == 0, probabilities == 1)
        )
        self.assertTrue(is_comp_basis_state)

        # Find the index of the highest probability
        max_prob_index = np.argmax(probabilities)

        num_qubits = int(np.log2(len(sv)))

        # Convert index to bitstring (assuming qubits in little-endian order)
        bitstring = format(max_prob_index, f"0{num_qubits}b")
        return bitstring

    def test_MaxKCutBinaryPowerTwo(self):
        for k in [2, 4, 8]:
            for method in ["PauliBasis", "Diffusion"]:
                problem = MaxKCutBinaryPowerOfTwo(
                    self.G,
                    k,
                    method=method,
                )
                problem.create_circuit()
                circuit = problem.circuit

                theta = np.pi
                circuit.assign_parameters([theta], inplace=True)
                # for k = 2 RZ is applied instead of a phase gate
                # they are equal up to a global phase, which we retract
                if method == "PauliBasis":
                    circuit.global_phase = -theta / 2

                num_qubits = problem.N_qubits
                for i in range(2 ** int(num_qubits / 2)):
                    for j in range(2 ** int(num_qubits / 2)):
                        q = QuantumRegister(len(circuit.qubits))
                        circuit_with_IX = QuantumCircuit(q)
                        binary_str_i = format(
                            i, f"0{int(num_qubits/2)}b"
                        )  # Create a binary string with leading zeros
                        binary_str_j = format(
                            j, f"0{int(num_qubits/2)}b"
                        )  # Create a binary string with leading zeros
                        for ind, bit in enumerate(binary_str_i):
                            if bit == "1":
                                circuit_with_IX.x(
                                    ind
                                )  # Apply an X-gate to the corresponding qubit
                        for ind, bit in enumerate(binary_str_j):
                            if bit == "1":
                                circuit_with_IX.x(
                                    int(num_qubits / 2) + ind
                                )  # Apply an X-gate to the corresponding qubit

                        circuit_with_IX_and_Hamiltonian = circuit_with_IX.compose(
                            circuit, inplace=False
                        )

                        sv_IX = Statevector(circuit_with_IX).data
                        sv_IX_Hamiltonian = Statevector(
                            circuit_with_IX_and_Hamiltonian
                        ).data

                        inner_product = np.vdot(sv_IX, sv_IX_Hamiltonian)

                        bitstring = self.stateVectorToBitstring(sv_IX)

                        # qiskit binary strings use little endian encoding, but our cost function expects big endian encoding. Therefore, we reverse the order
                        bitstring = bitstring[::-1]

                        # there should be a phase difference, if the nodes have the same color or not
                        int_i = int(binary_str_i, 2)
                        int_j = int(binary_str_j, 2)

                        if binary_str_i == binary_str_j:
                            self.assertTrue(np.isclose(inner_product, -1))
                            self.assertTrue(np.isclose(problem.cost(bitstring), 0))
                        else:
                            self.assertTrue(np.isclose(inner_product, 1))
                            self.assertTrue(np.isclose(problem.cost(bitstring), 1))

    def test_MaxKCutBinaryPowerTwo_PauliBasisequalDiffusion(self):
        # This tests if the Hamiltonians of "method=PauliBasis" is equal to "method=Diffusion"

        theta = -1.92748

        for k in [2, 4, 8]:
            statevector = {}
            for method in ["PauliBasis", "Diffusion"]:
                problem = MaxKCutBinaryPowerOfTwo(
                    self.G,
                    k,
                    method=method,
                )
                problem.create_circuit()
                circuit = problem.circuit

                circuit.assign_parameters([theta], inplace=True)
                # for k = 2 RZ is applied instead of a phase gate
                # they are equal up to a global phase, which we retract
                if method == "PauliBasis":
                    circuit.global_phase = -theta / 2

                q = QuantumRegister(len(circuit.qubits))
                circ = QuantumCircuit(q)

                circ.h(q[: problem.N_qubits])

                circuit = circ.compose(circuit, inplace=False)

                statevector[method] = Statevector(circuit).data
            self.assertTrue(
                np.allclose(
                    statevector["PauliBasis"], statevector["Diffusion"], atol=1e-8
                )
            )

    def test_MaxKCutBinaryFullH(self):
        for k in [3, 5, 6, 7]:
            for method in ["PauliBasis", "Diffusion", "PowerOfTwo"]:
                if k in [3, 7]:
                    colors = [
                        "LessThanK",
                    ]
                else:
                    colors = ["LessThanK", "max_balanced"]
                for color_encoding in colors:
                    problem = MaxKCutBinaryFullH(
                        self.G,
                        k,
                        color_encoding=color_encoding,
                        method=method,
                    )
                    problem.create_circuit()
                    circuit = problem.circuit

                    theta = np.pi
                    circuit.assign_parameters([theta], inplace=True)
                    # for k = 2 RZ is applied instead of a phase gate
                    # they are equal up to a global phase, which we retract
                    if method == "PauliBasis":
                        circuit.global_phase = -theta / 2

                    num_qubits = problem.N_qubits
                    for i in range(2 ** int(num_qubits / 2)):
                        for j in range(2 ** int(num_qubits / 2)):
                            q = QuantumRegister(len(circuit.qubits))
                            circuit_with_IX = QuantumCircuit(q)
                            binary_str_i = format(
                                i, f"0{int(num_qubits/2)}b"
                            )  # Create a binary string with leading zeros
                            binary_str_j = format(
                                j, f"0{int(num_qubits/2)}b"
                            )  # Create a binary string with leading zeros
                            for ind, bit in enumerate(binary_str_i):
                                if bit == "1":
                                    circuit_with_IX.x(
                                        ind
                                    )  # Apply an X-gate to the corresponding qubit
                            for ind, bit in enumerate(binary_str_j):
                                if bit == "1":
                                    circuit_with_IX.x(
                                        int(num_qubits / 2) + ind
                                    )  # Apply an X-gate to the corresponding qubit

                            circuit_with_IX_and_Hamiltonian = circuit_with_IX.compose(
                                circuit, inplace=False
                            )

                            sv_IX = Statevector(circuit_with_IX).data
                            sv_IX_Hamiltonian = Statevector(
                                circuit_with_IX_and_Hamiltonian
                            ).data

                            inner_product = np.vdot(sv_IX, sv_IX_Hamiltonian)

                            bitstring = self.stateVectorToBitstring(sv_IX)

                            # remove ancilla bits
                            if method == "PowerOfTwo":
                                bitstring = bitstring[2:]

                            # qiskit binary strings use little endian encoding, but our cost function expects big endian encoding. Therefore, we reverse the order
                            bitstring = bitstring[::-1]

                            # there should be a phase difference, if the nodes have the same color or not
                            int_i = int(binary_str_i, 2)
                            int_j = int(binary_str_j, 2)
                            if color_encoding == "LessThanK":
                                samecolor = (int_i >= k - 1) and (int_j >= k - 1)
                            elif color_encoding == "max_balanced":
                                if k == 5:
                                    # ((0,1), (2), (3), (4, 5), (6,7))
                                    samecolor = (
                                        ((int_i == 0) and (int_j == 1))
                                        or ((int_i == 1) and (int_j == 0))
                                        or ((int_i == 4) and (int_j == 5))
                                        or ((int_i == 5) and (int_j == 4))
                                        or ((int_i == 6) and (int_j == 7))
                                        or ((int_i == 7) and (int_j == 6))
                                    )
                                elif k == 6:
                                    # ((0,1), (2), (3), (4, 5), (6), (7))
                                    samecolor = (
                                        ((int_i == 0) and (int_j == 1))
                                        or ((int_i == 1) and (int_j == 0))
                                        or ((int_i == 4) and (int_j == 5))
                                        or ((int_i == 5) and (int_j == 4))
                                    )

                            if (binary_str_i == binary_str_j) or samecolor:
                                self.assertTrue(np.isclose(inner_product, -1))
                                self.assertTrue(np.isclose(problem.cost(bitstring), 0))
                            else:
                                self.assertTrue(np.isclose(inner_product, 1))
                                self.assertTrue(np.isclose(problem.cost(bitstring), 1))

    def test_MaxKCutBinaryPowerTwo_PauliBasisequalDiffusion(self):
        # This tests if the Hamiltonians of "method=PauliBasis" is equal to "method=Diffusion" is equal to "method="PowerOfTwo"

        theta = np.pi  # -1.92748

        for k in [3, 5, 6, 7]:
            if k in [3, 7]:
                colors = [
                    "LessThanK",
                ]
            else:
                colors = ["LessThanK", "max_balanced"]
            for color_encoding in colors:
                statevector = {}
                for method in ["PauliBasis", "Diffusion", "PowerOfTwo"]:
                    problem = MaxKCutBinaryFullH(
                        self.G,
                        k,
                        color_encoding=color_encoding,
                        method=method,
                    )

                    problem.create_circuit()
                    circuit = problem.circuit

                    circuit.assign_parameters([theta], inplace=True)
                    # for k = 2 RZ is applied instead of a phase gate
                    # they are equal up to a global phase, which we retract
                    if method == "PauliBasis":
                        circuit.global_phase = -theta / 2

                    q = QuantumRegister(len(circuit.qubits))
                    circ = QuantumCircuit(q)

                    circ.h(q[: problem.N_qubits])

                    circuit = circ.compose(circuit, inplace=False)

                    if method == "PowerOfTwo":
                        sv = Statevector(circuit).data
                        # Reshape the state vector to separate the last 2 qubits
                        reshaped_state = sv.reshape([2**2, 2**problem.N_qubits])

                        # Sum over the amplitudes of the last qubit to trace it out
                        statevector[method] = np.sum(reshaped_state, axis=0)
                    else:
                        statevector[method] = Statevector(circuit).data

                self.assertTrue(
                    np.allclose(
                        statevector["PauliBasis"], statevector["Diffusion"], atol=1e-8
                    )
                )
                self.assertTrue(
                    np.allclose(
                        statevector["PauliBasis"], statevector["PowerOfTwo"], atol=1e-8
                    )
                )
                self.assertTrue(
                    np.allclose(
                        statevector["PowerOfTwo"], statevector["Diffusion"], atol=1e-8
                    )
                )


if __name__ == "__main__":
    unittest.main()
