from abc import ABC, abstractmethod

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
