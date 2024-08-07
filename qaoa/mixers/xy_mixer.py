from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import XXPlusYYGate

from qaoa.mixers import Mixer

import math
import itertools

import numpy as np


class XY(Mixer):
    def __init__(self, topology=None) -> None:
        self.topology = topology
        self.mixer_param = Parameter("x_beta")

    def create_circuit(self):
        if not self.topology:
            print('No topology specified for the XY-mixer, assuming "ring" topology')
            self.topology = XY.generate_pairs(self.N_qubits)

        q = QuantumRegister(self.N_qubits)
        self.circuit = QuantumCircuit(q)

        for i, e in enumerate(self.topology):
            self.circuit.append(XXPlusYYGate(0.5 * self.mixer_param), e)

    @staticmethod
    def generate_pairs(n, case="ring"):
        # default ring, otherwise "chain"
        if n < 2:
            return []  # Not enough elements to form any pairs

        pairs = [[i, i + 1] for i in range(n - 1)]

        if case == "ring":
            pairs.append([n - 1, 0])

        return pairs
