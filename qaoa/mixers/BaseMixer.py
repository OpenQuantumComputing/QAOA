from abc import ABC, abstractmethod

class MixerBase(ABC):
    def __init__(self) -> None:
        self.circuit = None

    def setNumQubits(self, n):
        self.N_qubits = n


class Mixer(MixerBase):
    @abstractmethod
    def create_circuit(self):
        pass
