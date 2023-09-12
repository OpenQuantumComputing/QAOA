import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

from .BaseMixer import Mixer

class X(Mixer):
    def __init__(self, parent) -> None:
        super().__init__(parent)

    def set_initial_state(self, circuit, qubit_register):
        circuit.h(qubit_register)

    def create_mixer(self):
        q = QuantumRegister(self.params["N_qubits"])
        mixer_param = Parameter("x_beta")

        self.mixer_circuit = QuantumCircuit(q)
        self.mixer_circuit.rx(-2 * mixer_param, range(self.params["N_qubits"]))

        usebarrier = self.params.get("usebarrier", False)
        if usebarrier:
            self.mixer_circuit.barrier()
