from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit import Parameter
from abc import abstractmethod

from .base_problem import Problem
from qaoa.util import *


class GraphProblem(Problem):
    def __init__(
        self,
        G,
        N_qubits_per_node=1,
        fix_one_node: bool = False,  # this fixes the last node to color 1, i.e., one qubit gets removed
    ) -> None:
        super().__init__()

        # fixes the last node to "color1"
        self.fix_one_node = fix_one_node

        # ensure graph has labels 0, 1, ..., num_V-1
        self.graph_handler = GraphHandler(G)
        self.num_V = self.graph_handler.G.number_of_nodes()

        self.N_qubits_per_node = N_qubits_per_node
        self.N_qubits = (self.num_V - self.fix_one_node) * self.N_qubits_per_node

        self.beta_param = Parameter("gamma")

    @abstractmethod
    def create_edge_circuit(self, theta):
        """
        Abstract method to create circuit for an edge
        """
        pass

    @abstractmethod
    def create_edge_circuit_fixed_node(self, theta):
        """
        Abstract method to create circuit for an edge where one node is fixed
        """
        pass

    def create_circuit(self):
        """
        Adds a parameterized circuit for the cost part to the member variable self.parameteried_circuit
        and a parameter to the parameter list self.gamma_params
        """
        q = QuantumRegister(self.N_qubits)
        a = AncillaRegister(self.N_ancilla_qubits)
        self.circuit = QuantumCircuit(q, a)

        for _, edges in self.graph_handler.parallel_edges.items():
            for edge in edges:
                i, j = edge
                I = i * self.N_qubits_per_node
                J = j * self.N_qubits_per_node

                theta_ij = self.beta_param * self.graph_handler.G[i][j].get("weight", 1)

                if self.num_V - self.fix_one_node not in [i, j]:
                    qubits_to_map = list(range(I, I + self.N_qubits_per_node)) + list(
                        range(J, J + self.N_qubits_per_node)
                    )
                    ancilla_to_map = (
                        list(range(0, self.N_ancilla_qubits))
                        if self.N_ancilla_qubits > 0
                        else []
                    )
                    IJcirc = self.create_edge_circuit(theta_ij)
                    self.circuit.append(
                        IJcirc,
                        q[qubits_to_map]
                        + (a[ancilla_to_map] if ancilla_to_map else []),
                    )
                else:
                    # if self.fix_one_node is False, this branch does not exist
                    minIJ = min(I, J)
                    minIJcirc = self.create_edge_circuit_fixed_node(theta_ij)
                    self.circuit.append(
                        minIJcirc, q[list(range(minIJ, minIJ + self.N_qubits_per_node))]
                    )

    # the code below might go into BaseMaxKCut(GraphProblem)

    @staticmethod
    def is_power_of_two(k) -> bool:
        """
        Return True if k is a power of two, False otherwise.
        """
        if k > 0 and (k & (k - 1)) == 0:
            return True
        return False

    def same_color(self, str1: str, str2: str) -> bool:
        """Check if two strings map to the same color."""
        return self.bitstring_to_color.get(str1) == self.bitstring_to_color.get(str2)

    def slice_string(self, string: str) -> list:
        """
        Convert a binary string to a list of labels for each node.

        Args:
            string (str): Binary string.

        Returns:
            list: List of labels for each node.
        """
        k = self.N_qubits_per_node
        labels = [
            string[v * k : (v + 1) * k] for v in range(self.num_V - self.fix_one_node)
        ]
        # Add fixed node label if applicable
        if self.fix_one_node:
            labels.append(self.colors["color1"][0])
        return labels

    def cost(self, string: str) -> float | int:
        """
        Compute the cost for a given solution.

        Args:
            string (str): Binary string.

        Returns:
            float | int: The cost of the given solution.
        """
        if len(string) != self.N_qubits:
            raise ValueError(
                f"Expected a string of length {self.N_qubits}, "
                f"but received length {len(string)}."
            )

        labels = self.slice_string(string)
        return sum(
            self.graph_handler.G[edge[0]][edge[1]].get("weight", 1)
            for edge in self.graph_handler.G.edges()
            if not self.same_color(labels[edge[0]], labels[edge[1]])
        )
