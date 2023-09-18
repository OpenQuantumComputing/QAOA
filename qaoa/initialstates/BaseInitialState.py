import structlog

import math

import numpy as np


from abc import ABC, abstractmethod
from qiskit import QuantumCircuit, QuantumRegister

from qiskit.circuit import Parameter


class BaseInitialState(ABC):
    def __init__(self) -> None:
        self.circuit = None

    def setNumQubits(self, n):
        self.N_qubits = n

class InitialState(BaseInitialState):
    @abstractmethod
    def create_circuit(self):
        pass

