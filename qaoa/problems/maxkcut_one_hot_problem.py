from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
import networkx as nx

from .base_problem import Problem


class MaxKCutOneHot(Problem):
    """
    Max k-CUT one hot problem.

    Subclass of the `Problem` class, and it is...

    Attributes:
        G (nx.Graph):
        k_cuts (int):

    Methods:
        binstringToLabels(string): 
        cost(string):
        create_circuit():
        
    """
    def __init__(self, G: nx.Graph, k_cuts: int) -> None:
        """
        ...
        Args:
            G (nx.Graph):
            k_cuts (int):

        Raise: 
            ValueError:
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
        ...

        Args: 
            string (str):
        
        Raise:
            ValueError:
        
        Return:
            labels (...):
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
        Args:
            string (str):

        Return:
            C (...):
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
        ...
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
