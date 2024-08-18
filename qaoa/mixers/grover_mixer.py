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
        # given feasibel states f \in F,
        # Let US the circuit that prepares US = 1/|F| \sum_{f\inF} |f>
        # The Grover mixer has the form US^\dagger X^n C^{n-1}Phase X^n US,

        self.subcircuit.create_circuit()
        US = self.subcircuit.circuit

        # US^\dagger
        self.circuit = US.inverse()
        # X^n
        self.circuit.x(range(self.subcircuit.N_qubits))
        # C^{n-1}Phase
        phase_gate = PhaseGate(-self.mixer_param).control(self.subcircuit.N_qubits - 1)
        self.circuit.append(phase_gate, self.circuit.qubits)
        # X^n
        self.circuit.x(range(self.subcircuit.N_qubits))
        # US
        self.circuit.compose(US, range(self.subcircuit.N_qubits), inplace=True)
