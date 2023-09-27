from qaoa.util import dicke_circuit

from .base_initialstate import InitialState


class DickeState(InitialState):
    def __init__(self, k) -> None:
        super().__init__()
        self.k = k

    def create_circuit(self):
        decompose = True
        if decompose:
            self.circuit = (
                dicke_circuit.DickeCircuit(self.N_qubits, self.k)
                .decompose()
                .decompose()
                .decompose()
            )
        else:
            self.circuit = dicke_circuit.DickeCircuit(self.N_qubits, self.k)
