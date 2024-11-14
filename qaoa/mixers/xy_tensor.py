from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import XXPlusYYGate

from .base_mixer import Mixer
from .xy_mixer import XY
from qaoa.initialstates.tensor_initialstate import Tensor


class XYTensor(Mixer):
    def __init__(self, k_cuts: int, topology=None) -> None:
        self.k_cuts = k_cuts
        self.topology = topology

    def create_circuit(self) -> None:
        self.num_V = self.N_qubits / self.k_cuts

        if not self.num_V.is_integer():
            raise ValueError(
                "Total qubits="
                + str(self.N_qubits)
                + " is not a multiple of "
                + str(self.k_cuts)
            )
        self.num_V = int(self.num_V)

        xy = XY(self.topology)
        xy.setNumQubits(self.k_cuts)

        self.tensor = Tensor(xy, self.num_V)

        self.tensor.create_circuit()
        self.circuit = self.tensor.circuit
