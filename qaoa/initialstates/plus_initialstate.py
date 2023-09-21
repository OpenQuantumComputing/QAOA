from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

from .base_initialstate import InitialState


class Plus(InitialState):
    def __init__(self) -> None:
        super().__init__()

    def create_circuit(self):
        q = QuantumRegister(self.N_qubits)
        self.circuit = QuantumCircuit(q)
        self.circuit.h(q)
