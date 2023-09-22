from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

from .base_problem import Problem


class MaxCut(Problem):
    def __init__(self, G) -> None:
        self.G = G
        self.N_qubits = self.G.number_of_nodes()

    def cost(self, string):
        C = 0
        for edge in self.G.edges():
            i = int(edge[0])
            j = int(edge[1])
            if string[i] != string[j]:
                w = self.G[edge[0]][edge[1]]["weight"]
                C += w
        return C

    def create_circuit(self):
        """
        Adds a parameterized circuit for the cost part to the member variable self.parameteried_circuit
        and a parameter to the parameter list self.gamma_params
        """
        q = QuantumRegister(self.N_qubits)
        self.circuit = QuantumCircuit(q)
        cost_param = Parameter("x_gamma")

        ### cost Hamiltonian
        for edge in self.G.edges():
            i = int(edge[0])
            j = int(edge[1])
            w = self.G[edge[0]][edge[1]]["weight"]
            wg = w * cost_param
            self.circuit.cx(q[i], q[j])
            self.circuit.rz(wg, q[j])
            self.circuit.cx(q[i], q[j])
