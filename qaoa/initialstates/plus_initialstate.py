from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

from .base_initialstate import InitialState


class Plus(InitialState):
    """
    Plus initial state.

    Subclass of `InitialState` class, and it creates an initial plus state

    Methods:
        create_circuit(): Creates a circuit that sets up plus states for the initial states
    """
    def __init__(self) -> None:
        super().__init__()

    def create_circuit(self):
        """
        Creates a circuit of Hadamard-gates, which creates an initial state that is a plus state
        """
        q = QuantumRegister(self.N_qubits)
        self.circuit = QuantumCircuit(q)
        self.circuit.h(q)
