from qaoa.initialstates import InitialState
from qaoa.util import dicke_state

class DickeState(InitialState):
    def __init__(self, k) -> None:
        super().__init__()
        self.k = k

    def create_circuit(self):
        decompose = True
        if decompose:
            self.circuit = (
                dicke_state(self.N_qubits, self.k).decompose().decompose().decompose()
            )
        else:
            self.circuit = dicke_state(self.N_qubits, self.k)
