from qiskit.circuit import Parameter
from qiskit.circuit.library import PhaseGate

from qaoa.mixers.base_mixer import Mixer
from qaoa.initialstates.base_initialstate import InitialState


class Grover(Mixer):
    def __init__(self, subcircuit: InitialState) -> None:
        """
        Args:
            subcircuit (InitialState): the circuit that is to be tensorised
        """
        self.subcircuit = subcircuit
        self.mixer_param = Parameter("x_beta")

    def create_circuit(self):
        self.subcircuit.create_circuit()
        US = self.subcircuit.circuit

        self.circuit = US.inverse()
        self.circuit.barrier()

        if self.subcircuit.N_qubits == 2:
            self.circuit.x(range(self.subcircuit.N_qubits))
            self.circuit.crz(-self.mixer_param, -2, -1)
            self.circuit.x(range(self.subcircuit.N_qubits))
        else:
            self.circuit.x(range(self.subcircuit.N_qubits))
            phase_gate = PhaseGate(-self.mixer_param).control(self.subcircuit.N_qubits - 1)
            self.circuit.append(phase_gate, self.circuit.qubits)
            self.circuit.x(range(self.subcircuit.N_qubits))
        self.circuit.barrier()

        self.circuit.compose(US, range(self.subcircuit.N_qubits), inplace=True)
        self.circuit.barrier()
