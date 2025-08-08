from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

from .base_initialstate import InitialState


class Plus(InitialState):
    """
    Plus initial state.

    Subclass of `InitialState` class. Creates an the equal superposition of all computational basis states, |+>.

    Methods:
        create_circuit(): Generates the quantum circuit creating the plus initial state |+> from the |0> state.
    """
    def __init__(self) -> None:
        super().__init__()

    def create_circuit(self):
        """
        Creates a circuit of Hadamard-gates, which creates the |+> state from the |0> state.
        """
        q = QuantumRegister(self.N_qubits)
        self.circuit = QuantumCircuit(q)
        self.circuit.h(q)
