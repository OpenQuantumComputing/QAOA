import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import XXPlusYYGate

from .base_initialstate import InitialState


class ColorState(InitialState):
    def __init__(self, k_cuts):
        super().__init__()
        self.k_cuts = k_cuts
        self.k_bits = int(np.ceil(np.log2(self.k_cuts)))

    def create_circuit(self):
        q = QuantumRegister(self.N_qubits * self.k_bits)
        c = ClassicalRegister(self.N_qubits * self.k_bits)
        self.circuit = QuantumCircuit(q, c)
        self.circuit = QuantumCircuit(q, c)
        if self.k_cuts == 3:
            self.k3()
        elif self.k_cuts == 5:
            self.k5()
        elif self.k_cuts == 6:
            self.k6()
        elif self.k_cuts == 7:
            self.k7()

    def k3(self):
        theta = np.arccos(1/np.sqrt(3))*2
        phi = np.pi/2
        beta = -np.pi/2
        for v in range(self.N_qubits):
            self.circuit.ry(theta, v*self.k_bits)
            gate = XXPlusYYGate(phi, beta)
            self.circuit.append(gate, [v*self.k_bits, v*self.k_bits + 1])

    def k5(self):
        theta = np.arcsin(1/np.sqrt(5))*2
        for v in range(self.N_qubits):
            self.circuit.ry(theta, v*self.k_bits + 2)
            self.circuit.ch(v*self.k_bits + 2, [v*self.k_bits, v*self.k_bits + 1], ctrl_state=0)
    
    def k6(self):
        theta = np.pi/2
        phi = np.arccos(1/np.sqrt(3))*2
        gamma = np.pi/2
        beta = -np.pi/2
        for v in range(self.N_qubits):
            self.circuit.ry(theta, v*self.k_bits)
            self.circuit.ry(phi, v*self.k_bits + 1)
            gate = XXPlusYYGate(gamma, beta)
            self.circuit.append(gate, [v*self.k_bits + 1, v*self.k_bits + 2])
    
    def k7(self):
        delta = np.arcsin(1/np.sqrt(7))*2
        theta = np.pi/2
        phi = np.arccos(1/np.sqrt(3))*2
        gamma = np.pi/2
        beta = -np.pi/2
        for v in range(self.N_qubits):
            self.circuit.ry(delta, v*self.k_bits + 2)
            self.circuit.cx(v*self.k_bits + 2, v*self.k_bits + 1)
            self.circuit.cry(theta, v*self.k_bits + 2, v*self.k_bits, ctrl_state=0)
            self.circuit.cry(phi, v*self.k_bits + 2, v*self.k_bits + 1, ctrl_state=0)
            gate = XXPlusYYGate(gamma, beta)
            self.circuit.append(gate, [v*self.k_bits + 1, v*self.k_bits + 2])
        