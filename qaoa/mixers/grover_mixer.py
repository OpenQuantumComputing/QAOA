from qiskit.circuit import Parameter
from qiskit.circuit.library import PhaseGate

from .base_mixer import Mixer
from qaoa.initialstates.base_initialstate import InitialState


class Grover(Mixer):
    """
    Grover mixer.

    Subclass of the `Mixer` subclass that implements the Grover mixing operation.

    Attributes:
        subcircuit (InitialState): The initial state circuit.
        circuit (QuantumCircuit): The quantum circuit representing the mixer.
        mixer_param (Parameter): The parameter for the mixer.
        N_qubits (int): The number of qubits in the mixer circuit.

    Methods:
        create_circuit(): Constructs the Grover mixer circuit using the subcircuit.
    """

    def __init__(self, subcircuit: InitialState) -> None:
        """
        Initializes the Grover mixer.

        Args:
            subcircuit (InitialState): The initial state circuit.
        """
        self.subcircuit = subcircuit
        self.mixer_param = Parameter("x_beta")

    def create_circuit(self):
        """
        Constructs the Grover mixer circuit using the subcircuit.

        Given feasible states f \in F,
        and let US be the circuit that prepares US = 1/|F| \sum_{f\inF} |f>.
        The Grover mixer has the form US^\dagger X^n C^{n-1}Phase X^n US.
        """

        self.subcircuit.setNumQubits(self.N_qubits)
        self.subcircuit.create_circuit()
        US = self.subcircuit.circuit

        # US^\dagger
        self.circuit = US.inverse()
        # X^n
        self.circuit.x(range(self.subcircuit.N_qubits))
        # C^{n-1}Phase
        if self.subcircuit.N_qubits == 1:
            phase_gate = PhaseGate(-self.mixer_param)
        else:
            phase_gate = PhaseGate(-self.mixer_param).control(
                self.subcircuit.N_qubits - 1
            )
        self.circuit.append(phase_gate, self.circuit.qubits)
        # X^n
        self.circuit.x(range(self.subcircuit.N_qubits))
        # US
        self.circuit.compose(US, range(self.subcircuit.N_qubits), inplace=True)
