from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

from .base_mixer import Mixer


class XMultiAngle(Mixer):
    """
    X mixer with an independent rotation angle per qubit (multi-angle QAOA).

    Each qubit receives its own RX rotation parameter, allowing the optimizer
    to tune them independently. This provides more expressibility than the
    standard X mixer at the cost of more parameters per layer.

    Attributes:
        N_qubits (int): The number of qubits in the circuit (set via setNumQubits).
        circuit (QuantumCircuit): The mixer's quantum circuit (created by create_circuit).

    Methods:
        get_num_parameters(): Returns the number of parameters per layer (N_qubits).
        create_circuit(): Constructs the multi-angle X mixer circuit.
    """

    def get_num_parameters(self):
        """
        Returns the number of parameters per layer.

        Returns:
            int: Number of parameters per layer, equal to the number of qubits.
        """
        return self.N_qubits

    def create_circuit(self):
        """
        Constructs the multi-angle X mixer circuit.

        Each qubit i receives an independent RX(-2 * beta_i) rotation.
        Parameter names are zero-padded to ensure consistent alphabetical ordering.
        """
        q = QuantumRegister(self.N_qubits)
        self.circuit = QuantumCircuit(q)

        # Use zero-padded names so alphabetical sort matches qubit order
        n_digits = len(str(self.N_qubits))
        params = [
            Parameter("x_beta_{:0{}d}".format(i, n_digits))
            for i in range(self.N_qubits)
        ]
        for i, qubit in enumerate(q):
            self.circuit.rx(-2 * params[i], qubit)
