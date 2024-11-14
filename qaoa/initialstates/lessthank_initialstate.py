import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import XXPlusYYGate

from .base_initialstate import InitialState  # type: ignore


class LessThanK(InitialState):
    def __init__(self, k: int) -> None:
        if not LessThanK.is_power_of_two_or_between_2_and_8(k):
            raise ValueError("k must be a power of two or between 2 and 8")
        self.k = k
        self.N_qubits = int(np.ceil(np.log2(self.k)))

    def create_circuit(self) -> None:
        if self.k == 3:
            self.circuit = self.k3()
        elif self.k == 5:
            self.circuit = self.k5()
        elif self.k == 6:
            self.circuit = self.k6()
        elif self.k == 7:
            self.circuit = self.k7()
        else:
            self.circuit = self.power_of_two()

    def is_power_of_two_or_between_2_and_8(k):
        # Check if k is between 2 and 8
        if 2 <= k <= 8:
            return True

        # Check if k is a power of two
        # A number is a power of two if it has exactly one bit set, i.e., k & (k - 1) == 0 and k > 0
        if k > 0 and (k & (k - 1)) == 0:
            return True

        return False

    def power_of_two(self) -> QuantumCircuit:
        q = QuantumRegister(self.N_qubits)
        circuit = QuantumCircuit(q)
        circuit.h(q)
        return circuit

    def k3(self) -> QuantumCircuit:
        q = QuantumRegister(self.N_qubits)
        circuit = QuantumCircuit(q)
        theta = np.arccos(1 / np.sqrt(3)) * 2
        phi = np.pi / 2
        beta = -np.pi / 2
        circuit.ry(theta, 1)
        gate = XXPlusYYGate(phi, beta)
        circuit.append(gate, [0, 1])
        return circuit

    def k5(self) -> QuantumCircuit:
        q = QuantumRegister(self.N_qubits)
        circuit = QuantumCircuit(q)
        theta = np.arcsin(1 / np.sqrt(5)) * 2
        circuit.ry(theta, 0)
        circuit.ch(0, [1, 2], ctrl_state=0)
        return circuit

    def k6(self) -> QuantumCircuit:
        q = QuantumRegister(self.N_qubits)
        circuit = QuantumCircuit(q)
        theta = np.pi / 2
        phi = np.arccos(1 / np.sqrt(3)) * 2
        gamma = np.pi / 2
        beta = -np.pi / 2
        circuit.ry(theta, 2)
        circuit.ry(phi, 1)
        gate = XXPlusYYGate(gamma, beta)
        circuit.append(gate, [0, 1])
        return circuit

    def k7(self) -> QuantumCircuit:
        q = QuantumRegister(self.N_qubits)
        circuit = QuantumCircuit(q)
        delta = np.arcsin(1 / np.sqrt(7)) * 2
        theta = np.pi / 2
        phi = np.arccos(1 / np.sqrt(3)) * 2
        gamma = np.pi / 2
        beta = -np.pi / 2
        circuit.ry(delta, 0)
        circuit.cx(0, 1)
        circuit.cry(theta, 0, 2, ctrl_state=0)
        circuit.cry(phi, 0, 1, ctrl_state=0)
        gate = XXPlusYYGate(gamma, beta)
        circuit.append(gate, [0, 1])
        return circuit
