from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
import networkx as nx

from .base_problem import Problem


class MaxKCutOneHot(Problem):
    """
    Max k-CUT problem using one-hot encoding.

    Subclass of the `Problem` class. This class formulates the Max k-Cut problem for a given graph using a one-hot encoding for node colors.
    It provides methods to convert bitstrings to color labels, compute the cut value and construct the corresponding quantum circuit.

    Attributes:
        G (nx.Graph): The input graph on which the Max k-Cut problem is defined.
        k_cuts (int): The number of partitions (colors) to cut the graph into.
        num_V (int): The number of nodes in the graph.
        N_qubits (int): The total number of qubits (nodes × colors).

    Methods:
        binstringToLabels(string): Converts a binary string in one-hot encoding to a string of color labels for each node.
        cost(string): Computes the Max k-Cut cost for a given binary string representing a coloring.
        create_circuit(): Creates the parameterized quantum circuit corresponding to the Max k-Cut cost function using one-hot encoding.
    """
    def __init__(self, G: nx.Graph, k_cuts: int) -> None:
        """
        Args:
            G (nx.Graph): The input graph on which the Max k-Cut problem is defined.
            k_cuts (int): The number of partitions (colors) to cut the graph into.

        Raises: 
            ValueError: If k_cuts is less than 2 or greater than 8.
        """
        super().__init__()
        if (k_cuts < 2) or (k_cuts > 8):
            raise ValueError(
                "k_cuts must be 2 or more, and is not implemented for k_cuts > 8"
            )
        self.G = G
        self.num_V = self.G.number_of_nodes()
        self.k_cuts = k_cuts
        self.N_qubits = self.num_V * self.k_cuts

    def binstringToLabels(self, string: str) -> str:
        """
        Converts a binary string in one-hot encoding to a string of color labels for each node.

        Args: 
            string (str): The binary string representing the one-hot encoding of node colors.

        Raises:
            ValueError: If a segment of the string does not represent a valid one-hot encoding.

        Returns:
            labels (str): String of color labels for each node.
        """
        k = self.k_cuts
        labels = ""
        for v in range(self.num_V):
            segment = string[v * k : (v + 1) * k]
            rev = segment[::-1]
            idx = rev.find("1")
            if idx == -1:
                raise ValueError(
                    f"Segment {segment} from {string} is not a valid encoding"
                )
            labels += str(idx)
        return labels

    def cost(self, string: str) -> float | int:
        """
        Computes the Max k-Cut cost for a given binary string representing a coloring.

        Args:
            string (str): The binary string representing the one-hot encoding of node colors.

        Returns:
            C (float or int): The total cut value for the given coloring.
        """
        labels = self.binstringToLabels(string)
        C = 0
        for edge in self.G.edges():
            i = edge[0]
            j = edge[1]
            li = min(self.k_cuts - 1, int(labels[int(i)]))
            lj = min(self.k_cuts - 1, int(labels[int(j)]))
            if li != lj:
                w = self.G[edge[0]][edge[1]]["weight"]
                C += w
        return C

    def create_circuit(self) -> None:
        """
        Creates the parameterized quantum circuit corresponding to the Max k-Cut cost function using one-hot encoding.

        """
        q = QuantumRegister(self.N_qubits)
        c = ClassicalRegister(self.N_qubits)
        self.circuit = QuantumCircuit(q, c)

        cost_param = Parameter("x_gamma")

        # the objective Hamiltonian
        for edge in self.G.edges():
            i = int(edge[0])
            j = int(edge[1])
            w = self.G[edge[0]][edge[1]]["weight"]
            wg = w * cost_param
            I = self.k_cuts * i
            J = self.k_cuts * j
            for k in range(self.k_cuts):
                self.circuit.cx(q[I + k], q[J + k])
                self.circuit.rz(wg, q[J + k])
                self.circuit.cx(q[I + k], q[J + k])
            self.circuit.barrier()
