from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

from .base_initialstate import InitialState


class StateVector(InitialState):
    """
    State vector initial state. 

    Subclass of the `InitialState` class, and it creates the initial statevector.

    Attributes:
        statevector (list): The statevector to initialize the circuit with.

    Methods:
        create_circuit(): Creates a circuit that creates the initial statevector.
    """
    def __init__(self, statevector) -> None:
        """
        Args:
            statevector (list): The statevector to initialize the circuit with.
        """
        super().__init__()
        self.statevector = statevector

    def create_circuit(self):
        """
        Creates a circuit that makes the initial statevector
        """
        q = QuantumRegister(self.N_qubits)
        self.circuit = QuantumCircuit(q)
        self.circuit.initialize(self.statevector, q)
