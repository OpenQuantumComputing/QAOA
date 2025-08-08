import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import XXPlusYYGate, PauliEvolutionGate

from qiskit.quantum_info import SparsePauliOp

from .base_initialstate import InitialState


class Dicke1_2(InitialState):
    """
    Dicke1_2 initial state.

    Subclass of the `InitialState` class, and it returns equal superposition Dicke 1 and Dicke 2 states. It is Hard Coded for the case of Hamming weight k = 6.

    Methods:
        create_circuit(): Generates the circuit that creates the superposition of Dicke 1 and Dicke 2 states.
    """

    def __init__(self) -> None:
        self.k = 6
        self.N_qubits = 3

    def create_circuit(self) -> None:
        """
        Generates the circuit that creates the superposition of Dicke 1 and Dicke 2 states.
        """
        q = QuantumRegister(self.N_qubits)
        circuit = QuantumCircuit(q)
        X = SparsePauliOp("X")
        Y = SparsePauliOp("Y")
        operator = Y ^ Y ^ Y
        circuit.x(0)
        # qc.ry(np.pi/2,2)
        circuit.append(PauliEvolutionGate(operator, time=np.pi / 4), q)
        circuit.append(XXPlusYYGate(np.arcsin(2 * np.sqrt(2) / 3), np.pi / 2), [0, 1])
        circuit.append(XXPlusYYGate(-np.pi / 2, np.pi / 2), [0, 2])
        circuit.x(1)
        circuit.cz(q[1], q[2])
        circuit.x(1)
        self.circuit = circuit
