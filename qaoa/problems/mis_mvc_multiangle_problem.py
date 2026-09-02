"""Multi-angle phase separators for MIS and MVC."""

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

from .base_problem import ObjectiveSense
from .mis_mvc_problem import MIS_MVC_Problem


class MIS_MVC_MultiAngle(MIS_MVC_Problem):
    """MIS/MVC problem with one gamma parameter per vertex."""

    def get_num_parameters(self):
        return self.N_qubits

    def create_circuit(self):
        width = max(1, len(str(self.N_qubits - 1)))
        gamma_params = [
            Parameter(f"gamma_{node:0{width}d}") for node in range(self.N_qubits)
        ]

        q = QuantumRegister(self.N_qubits)
        self.circuit = QuantumCircuit(q)

        direction = -1.0 if self.objective_sense is ObjectiveSense.MINIMIZE else 1.0
        for node, gamma in enumerate(gamma_params):
            self.circuit.rz(direction * self.weights[node] * gamma, q[node])
