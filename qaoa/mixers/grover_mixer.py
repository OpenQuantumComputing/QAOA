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

        The subcircuit (e.g. Dicke) is shown as a labelled box in circuit drawings,
        so the structure is immediately visible.
        """
        from qiskit import QuantumRegister, QuantumCircuit

        # Only update the subcircuit's qubit count when it differs from our own,
        # preserving any N_qubits that was already set on the subcircuit.
        if not hasattr(self.subcircuit, "N_qubits") or self.subcircuit.N_qubits != self.N_qubits:
            self.subcircuit.setNumQubits(self.N_qubits)
        self.subcircuit.create_circuit()
        US = self.subcircuit.circuit

        sub_label = getattr(self.subcircuit, "label", self.subcircuit.__class__.__name__)
        n = self.subcircuit.N_qubits

        qr = QuantumRegister(n, name="q")
        self.circuit = QuantumCircuit(qr)

        # US^\dagger — shown as a labelled box (e.g. "Dicke†")
        self.circuit.append(US.inverse().to_instruction(label=f"{sub_label}\u2020"), range(n))
        # X^n
        self.circuit.x(range(n))
        # C^{n-1}Phase
        if n == 1:
            phase_gate = PhaseGate(-self.mixer_param)
        else:
            phase_gate = PhaseGate(-self.mixer_param).control(n - 1)
        self.circuit.append(phase_gate, range(n))
        # X^n
        self.circuit.x(range(n))
        # US — shown as a labelled box (e.g. "Dicke")
        self.circuit.append(US.to_instruction(label=sub_label), range(n))
