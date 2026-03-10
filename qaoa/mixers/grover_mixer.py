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

    def __init__(self, subcircuit: InitialState, label: str | None = None) -> None:
        """
        Initializes the Grover mixer.

        Args:
            subcircuit (InitialState): The initial state circuit.  If the
                subcircuit already has ``N_qubits`` set the Grover mixer
                inherits it automatically so that ``setNumQubits`` does not
                need to be called separately.
            label (str | None): Optional annotation label.  Defaults to
                ``"Grover"``.
        """
        super().__init__(label=label)
        self.subcircuit = subcircuit
        self.mixer_param = Parameter("x_beta")
        # Inherit N_qubits from the subcircuit when it is already known.
        if hasattr(subcircuit, "N_qubits"):
            self.N_qubits = subcircuit.N_qubits

    def create_circuit(self):
        r"""
        Constructs the Grover mixer circuit using the subcircuit.

        Given feasible states f \in F,
        and let US be the circuit that prepares US = 1/|F| \sum_{f\inF} |f>.
        The Grover mixer has the form US^\dagger X^n C^{n-1}Phase X^n US.
        """

        # Only update the subcircuit's qubit count when it differs from our own,
        # preserving any N_qubits that was already set on the subcircuit.
        if not hasattr(self.subcircuit, "N_qubits") or self.subcircuit.N_qubits != self.N_qubits:
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
