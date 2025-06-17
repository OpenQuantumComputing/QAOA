from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import XXPlusYYGate

from .base_mixer import Mixer

import math
import itertools

import numpy as np


class XY(Mixer):
    """
    XY mixer.

    Subclass of the `Mixer` subclass that implements the XY mixing operation.

    Attributes:
        topology (list): The topology of the mixer, default is None.
        mixer_param (Parameter): The parameter for the XY mixer.
        N_qubits (int): The number of qubits in the mixer circuit.
        circuit (QuantumCircuit): The quantum circuit representing the XY mixer.

    Methods:
        create_circuit(): Constructs the XY mixer circuit using the specified topology.
        generate_pairs(n, case="ring"): Generates pairs of qubits based on the specified topology.
    """

    def __init__(self, topology=None) -> None:
        """
        Initializes the XY mixer.

        Args:
            topology (list, optional): The topology of the mixer. If None, defaults to "ring" topology.
        """
        self.topology = topology
        self.mixer_param = Parameter("x_beta")

    def create_circuit(self):
        """
        Constructs the XY mixer circuit using the specified topology.

        If no topology is specified, it defaults to a "ring" topology.
        """
        if not self.topology:
            print('No topology specified for the XY-mixer, assuming "ring" topology')
            self.topology = XY.generate_pairs(self.N_qubits)

        q = QuantumRegister(self.N_qubits)
        self.circuit = QuantumCircuit(q)

        for i, e in enumerate(self.topology):
            self.circuit.append(XXPlusYYGate(0.5 * self.mixer_param), e)

    @staticmethod
    def generate_pairs(n, case="ring"):
        """_summary_

        Args:
            n (int): The number of qubits.
            case (str, optional): Topology. Defaults to "ring".

        Returns:
            list: A list of pairs of qubit indices based on the specified topology.
        """
        # default ring, otherwise "chain"
        if n < 2:
            return []  # Not enough elements to form any pairs

        pairs = [[i, i + 1] for i in range(n - 1)]

        if case == "ring":
            pairs.append([n - 1, 0])

        return pairs
