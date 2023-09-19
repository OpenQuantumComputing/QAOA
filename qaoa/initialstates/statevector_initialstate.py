from qaoa.initialstates import InitialState

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter


class StateVector(InitialState):
    def __init__(self, statevector) -> None:
        super().__init__()
        self.statevector = statevector

    def create_circuit(self):
        q = QuantumRegister(self.N_qubits)
        self.circuit = QuantumCircuit(q)
        self.circuit.initialize(self.statevector, q)
