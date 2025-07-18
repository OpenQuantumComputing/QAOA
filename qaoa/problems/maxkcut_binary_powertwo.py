import networkx as nx
import numpy as np
import itertools
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import PhaseGate

from qiskit.circuit.library import PauliEvolutionGate

from qiskit.quantum_info import SparsePauliOp, Pauli

from .graph_problem import GraphProblem


class MaxKCutBinaryPowerOfTwo(GraphProblem):
    """
    Max k-CUT binary power of two graph problem.

    Subclass of the `GraphProblem` class. This class implements the Max k-Cut problem for graphs where the number of colors (k) is a power of two, using a binary encoding for node colors. It provides methods for constructing color mappings, generating quantum circuits for edges, and building the corresponding cost Hamiltonian in the Pauli basis.

    Attributes:
        G (nx.Graph): The input graph on which the Max k-Cut problem is defined.
        k_cuts (int): The number of partitions (colors) to cut the graph into (must be a power of two).
        method (str): The method used for circuit construction ("PauliBasis" or "Diffusion").
        fix_one_node (bool): If True, fixes the last node to a specific color, reducing the number of variables.

    Methods:
        is_power_of_two(k): Checks if the given integer k is a power of two.
        validate_parameters(k, method): Validates the input parameters for k and method.
        construct_colors(): Constructs the mapping from binary strings to color classes based on k.
        create_edge_circuit(theta): Creates the parameterized quantum circuit for an edge, according to the chosen method.
        create_edge_circuit_fixed_node(theta): Creates the parameterized quantum circuit for an edge when one node is fixed.
        getPauliOperator(k_cuts, color_encoding): Returns the Pauli operators for the cost Hamiltonian for the given k and encoding.

    """
    def __init__(
        self,
        G: nx.Graph,
        k_cuts: int,
        method: str = "Diffusion",
        fix_one_node: bool = False,  # this fixes the last node to color 1, i.e., one qubit gets removed
    ) -> None:
        """
        Args:
            G (nx.Graph): The input graph on which the Max k-Cut problem is defined.
            k_cuts (int): The number of partitions (colors) to cut the graph into (must be a power of two).
            method (str): The method used for circuit construction ("PauliBasis" or "Diffusion").
            fix_one_node (bool): If True, fixes the last node to a specific color, reducing the number of variables.

        Raises:
            ValueError: If k_cuts is not a power of two, is less than 2, greater than 8, or if method is not valid.
        """
        MaxKCutBinaryPowerOfTwo.validate_parameters(k_cuts, method)

        self.k_cuts = k_cuts
        self.method = method

        N_qubits_per_node = int(np.ceil(np.log2(self.k_cuts)))
        super().__init__(G, N_qubits_per_node, fix_one_node)

        if self.method == "PauliBasis":
            self.op, self.ophalf = self.getPauliOperator(self.k_cuts, "all")

        self.construct_colors()

    @staticmethod
    def is_power_of_two(k) -> bool:
        """
        Checks if the given integer k is a power of two.

        Args:
            k (int): The integer to check.

        Returns:
            bool: True if k is a power of two, False otherwise.
        """
        if k > 0 and (k & (k - 1)) == 0:
            return True
        return False

    @staticmethod
    def validate_parameters(k, method) -> None:
        """
        Validates the input parameters for k and method.

        Args:
            k (int): Number of partitions (colors).
            method (str): Circuit construction method ("PauliBasis" or "Diffusion").

        Raises:
            ValueError: If k is not a power of two.
            ValueError: If k is less than 2 or greater than 8.
            ValueError: If method is not valid.
        """
        ### 1) k_cuts must be a power of 2
        if not MaxKCutBinaryPowerOfTwo.is_power_of_two(k):
            raise ValueError("k_cuts must be a power of two")

        ### 2) k_cuts needs to be between 2 and 8
        if (k < 2) or (k > 8):
            raise ValueError(
                "k_cuts must be 2 or more, and is not implemented for k_cuts > 8"
            )

        ### 3) method
        valid_methods = ["PauliBasis", "Diffusion"]
        if method not in valid_methods:
            raise ValueError("method must be in " + str(valid_methods))

    def construct_colors(self):
        """
        Constructs the mapping from binary strings to color classes based on k.

        Raises:
            ValueError: If k_cuts is not supported.
        """
        if self.k_cuts == 2:
            self.colors = {"color1": ["0"], "color2": ["1"]}
        elif self.k_cuts == 4:
            self.colors = {
                "color1": ["00"],
                "color2": ["01"],
                "color3": ["10"],
                "color4": ["11"],
            }
        elif self.k_cuts == 8:
            self.colors = {
                "color1": ["000"],
                "color2": ["001"],
                "color3": ["010"],
                "color4": ["011"],
                "color5": ["100"],
                "color6": ["101"],
                "color7": ["110"],
                "color8": ["111"],
            }
        # Create a dictionary to map each index to its corresponding set
        self.bitstring_to_color = {}
        for key, indices in self.colors.items():
            for index in indices:
                self.bitstring_to_color[index] = key

    def create_edge_circuit(self, theta):
        """
        Creates the parameterized quantum circuit for an edge, according to the chosen method.

        Args:
            theta (float): The phase parameter.

        Returns:
            qc (QuantumCircuit): The constructed quantum circuit for the edge.
        """
        qc = QuantumCircuit(2 * self.N_qubits_per_node)
        if self.method == "PauliBasis":
            qc.append(PauliEvolutionGate(self.op, time=theta), qc.qubits)
        else:
            for k in range(self.N_qubits_per_node):
                qc.cx(k, self.N_qubits_per_node + k)
                qc.x(self.N_qubits_per_node + k)
            # C^{n-1}Phase
            if self.N_qubits_per_node == 1:
                phase_gate = PhaseGate(-theta)
            else:
                phase_gate = PhaseGate(-theta).control(self.N_qubits_per_node - 1)
            qc.append(
                phase_gate,
                [
                    self.N_qubits_per_node - 1 + ind
                    for ind in range(1, self.N_qubits_per_node + 1)
                ],
            )
            for k in reversed(range(self.N_qubits_per_node)):
                qc.x(self.N_qubits_per_node + k)
                qc.cx(k, self.N_qubits_per_node + k)
        return qc

    def create_edge_circuit_fixed_node(self, theta):
        """
        Creates the parameterized quantum circuit for an edge when one node is fixed.

        Args:
            theta (float): The phase parameter.

        Returns:
            qc (QuantumCircuit): The constructed quantum circuit for the edge with a fixed node.
        """
        qc = QuantumCircuit(self.N_qubits_per_node)
        if self.method == "PauliBasis":
            qc.append(PauliEvolutionGate(self.ophalf, time=-theta), qc.qubits)
        else:
            qc.x(qc.qubits)
            # C^{n-1}Phase
            if self.N_qubits_per_node == 1:
                phase_gate = PhaseGate(-theta)
            else:
                phase_gate = PhaseGate(-theta).control(self.N_qubits_per_node - 1)
            qc.append(phase_gate, qc.qubits)
            qc.x(qc.qubits)
        return qc

    def getPauliOperator(self, k_cuts, color_encoding):
        """
        Returns the Pauli operators for the cost Hamiltonian for the given k and encoding.

        Args:
            k_cuts (int): Number of partitions (colors).
            color_encoding (str): The encoding scheme for colors.

        Returns:
            op (SparsePauliOp): The full Pauli operator for the cost Hamiltonian.
            ophalf (SparsePauliOp): The half Pauli operator for the fixed-node case.
        """
        # flip Pauli strings, because of qiskit's little endian encoding
        if k_cuts == 2:
            P = [
                [2 / (2**1), Pauli("ZZ")],
            ]
            Phalf = [
                [2 / (2**1), Pauli("Z")],
            ]
        elif k_cuts == 4:
            P = [
                [-8 / (2**4), Pauli("IIII"[::-1])],
                [+8 / (2**4), Pauli("IZIZ"[::-1])],
                [+8 / (2**4), Pauli("ZIZI"[::-1])],
                [+8 / (2**4), Pauli("ZZZZ"[::-1])],
            ]
            Phalf = [
                [-8 / (2**4), Pauli("II"[::-1])],
                [+8 / (2**4), Pauli("IZ"[::-1])],
                [+8 / (2**4), Pauli("ZI"[::-1])],
                [+8 / (2**4), Pauli("ZZ"[::-1])],
            ]
        else:
            P = [
                [-48 / (2**6), Pauli("IIIIII"[::-1])],
                [+16 / (2**6), Pauli("IIZIIZ"[::-1])],
                [+16 / (2**6), Pauli("IZIIZI"[::-1])],
                [+16 / (2**6), Pauli("IZZIZZ"[::-1])],
                [+16 / (2**6), Pauli("ZIIZII"[::-1])],
                [+16 / (2**6), Pauli("ZIZZIZ"[::-1])],
                [+16 / (2**6), Pauli("ZZIZZI"[::-1])],
                [+16 / (2**6), Pauli("ZZZZZZ"[::-1])],
            ]
            Phalf = [
                [-48 / (2**6), Pauli("III"[::-1])],
                [+16 / (2**6), Pauli("IIZ"[::-1])],
                [+16 / (2**6), Pauli("IZI"[::-1])],
                [+16 / (2**6), Pauli("IZZ"[::-1])],
                [+16 / (2**6), Pauli("ZII"[::-1])],
                [+16 / (2**6), Pauli("ZIZ"[::-1])],
                [+16 / (2**6), Pauli("ZZI"[::-1])],
                [+16 / (2**6), Pauli("ZZZ"[::-1])],
            ]

        # devide coefficients by 2, since:
        # "The evolution gates are related to the Pauli rotation gates by a factor of 2"
        op = SparsePauliOp([item[1] for item in P], coeffs=[item[0] / 2 for item in P])
        ophalf = SparsePauliOp(
            [item[1] for item in Phalf], coeffs=[item[0] / 2 for item in Phalf]
        )
        return op, ophalf
