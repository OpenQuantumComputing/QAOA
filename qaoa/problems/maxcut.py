from qaoa.problems import Problem

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter


class MaxCut(Problem):
    def __init__(self, parent) -> None:
        super().__init__(parent=parent)

        self.G = self.params["G"]
        self.N_qubits = self.G.number_of_nodes()
        self.params["N_qubits"] = self.N_qubits

    def cost(self, string):
        C = 0
        for edge in self.G.edges():
            i = int(edge[0])
            j = int(edge[1])
            if string[i] != string[j]:
                w = self.G[edge[0]][edge[1]]["weight"]
                C += w
        return C

    def create_phase(self):
        """
        Adds a parameterized circuit for the cost part to the member variable self.parameteried_circuit
        and a parameter to the parameter list self.gamma_params
        """
        q = QuantumRegister(self.N_qubits)
        self.phase_circuit = QuantumCircuit(q)
        cost_param = Parameter("x_gamma")
        usebarrier = self.params.get("usebarrier", False)

        if usebarrier:
            self.phase_circuit.barrier()

        ### cost Hamiltonian
        for edge in self.G.edges():
            i = int(edge[0])
            j = int(edge[1])
            w = self.G[edge[0]][edge[1]]["weight"]
            wg = w * cost_param
            self.phase_circuit.cx(q[i], q[j])
            self.phase_circuit.rz(wg, q[j])
            self.phase_circuit.cx(q[i], q[j])
            if usebarrier:
                self.phase_circuit.barrier()
