import structlog

LOG = structlog.get_logger(file=__name__)

import math

import numpy as np


from abc import ABC, abstractmethod
from qiskit import QuantumCircuit, QuantumRegister

from qiskit.circuit import Parameter


class BaseProblem(ABC):
    def __init__(self) -> None:
        self.circuit = None


class Problem(BaseProblem):
    @abstractmethod
    def cost(self, string):
        pass

    @abstractmethod
    def create_circuit(self):
        pass

    def isFeasible(self, string):
        return True
