from abc import ABC, abstractmethod

class BaseInitialState(ABC):
    def __init__(self) -> None:
        self.circuit = None

    def setNumQubits(self, n):
        self.N_qubits = n


class InitialState(BaseInitialState):
    @abstractmethod
    def create_circuit(self):
        pass
