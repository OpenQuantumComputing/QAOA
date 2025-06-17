from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

from .base_mixer import Mixer


class X(Mixer):
    """
    X mixer.

    Subclass of the `Mixer` subclass that implements the X mixing operation.

    Attributes:
        mixer_param (Parameter): The parameter for the mixer.
        N_qubits (int): The number of qubits in the circuit.
        circuit (QuantumCircuit): The mixer's quantum circuit.

    Methods:
        create_circuit(): Constructs the X mixer circuit.
    """

    def __init__(self) -> None:
        """
        Initializes the X mixer.
        """
        self.mixer_param = Parameter("x_beta")

    def create_circuit(self):
        """
        Constructs the X mixer circuit.
        """
        q = QuantumRegister(self.N_qubits)

        self.circuit = QuantumCircuit(q)
        self.circuit.rx(-2 * self.mixer_param, range(self.N_qubits))
