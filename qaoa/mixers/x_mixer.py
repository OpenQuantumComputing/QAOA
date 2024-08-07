from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

from .base_mixer import Mixer


class X(Mixer):
    def __init__(self) -> None:
        self.mixer_param = Parameter("x_beta")

    def create_circuit(self):
        q = QuantumRegister(self.N_qubits)

        self.circuit = QuantumCircuit(q)
        self.circuit.rx(-2 * self.mixer_param, range(self.N_qubits))
