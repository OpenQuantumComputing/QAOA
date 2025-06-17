import numpy as np
from copy import deepcopy

from qiskit import QuantumRegister, QuantumCircuit

from .base_initialstate import InitialState


class Tensor(InitialState):
    """
    Tensor initial state.

    Subclass of the `IntialState` class that creates a tensor out of a circuit

    Attributions:
        subcircuit (InitialState): the circuit that is to be tensorised
        num (int): number of qubits of the subpart 

    Methods:
        create_circuit(): 
    """
    def __init__(self, subcircuit: InitialState, num: int) -> None:
        """
        Args:
            subcircuit (InitialState): the circuit that is to be tensorised
            num (int): number of qubits of the subpart #subN_qubits
        """
        self.num = num
        self.subcircuit = subcircuit
        self.N_qubits = self.num * self.subcircuit.N_qubits

    def create_circuit(self) -> None:
        """
        Creates a circuit that tensorises a given subcircuit
        """
        self.subcircuit.create_circuit()
        self.circuit = self.subcircuit.circuit
        for v in range(self.num - 1):
            self.subcircuit.create_circuit()  # self.subcircuit.circuit.qregs)
            self.circuit.tensor(self.subcircuit.circuit, inplace=True)
