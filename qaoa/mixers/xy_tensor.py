from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import XXPlusYYGate

from .base_mixer import Mixer
from .xy_mixer import XY
from qaoa.initialstates.tensor_initialstate import Tensor


class XYTensor(Mixer):
    """
    XY tensor mixer for the Max k-Cut problem.

    Subclass of the `Mixer` class that implements the XY tensor mixing operation for the Max k-Cut problem.

    Attributes:
        k_cuts (int): The number of cuts in the Max k-Cut problem.
        topology (list): The topology of the mixer, default is None.
        num_V (int): The number of vertices in the Max k-Cut problem.

    Methods:
        create_circuit(): Constructs the XY tensor mixer circuit for the Max k-Cut problem.
    """

    def __init__(self, k_cuts: int, topology=None) -> None:
        """
        Initializes the XYTensor mixer for the Max k-Cut problem.

        Args:
            k_cuts (int): The number of cuts in the Max k-Cut problem.
            topology (list, optional): The topology of the mixer. If None, defaults to "ring" topology.
        """
        self.k_cuts = k_cuts
        self.topology = topology

    def create_circuit(self) -> None:
        """
        Constructs the XY tensor mixer circuit for the Max k-Cut problem.

        Raises:
            ValueError: If the total number of qubits is not a multiple of `k_cuts`.
        """
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
