from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

from .base_initialstate import InitialState


class PlusParameterized(InitialState):
    """
    Parameterized initial state with individual phase rotations per qubit.

    Applies a Hadamard gate to each qubit (creating the |+> state), followed
    by parameterized RZ rotations. This allows the initial state to be
    optimized as part of the variational ansatz.

    Attributes:
        num_phases (int or None): Number of phase parameters. If None, uses N_qubits.
        N_qubits (int): Number of qubits (set via setNumQubits).
        circuit (QuantumCircuit): The initial state circuit (created by create_circuit).

    Methods:
        get_num_parameters(): Returns the number of phase parameters.
        create_circuit(): Creates the circuit with H gates and parameterized RZ rotations.
    """

    def __init__(self, num_phases=None):
        """
        Initializes the PlusParameterized initial state.

        Args:
            num_phases (int, optional): Number of phase parameters. If None, uses N_qubits
                (one parameter per qubit). Must not exceed N_qubits.
        """
        super().__init__()
        self.num_phases = num_phases

    def get_num_parameters(self):
        """
        Returns the number of phase parameters.

        Returns:
            int: Number of phase parameters (num_phases if set, else N_qubits).
        """
        return self.num_phases if self.num_phases is not None else self.N_qubits

    def create_circuit(self):
        """
        Creates the parameterized initial state circuit.

        Applies Hadamard gates to all qubits, then applies RZ rotations with
        individual parameters to the first `get_num_parameters()` qubits.
        Parameter names are zero-padded to ensure consistent alphabetical ordering.
        """
        q = QuantumRegister(self.N_qubits)
        self.circuit = QuantumCircuit(q)

        # Apply Hadamard to all qubits to create |+> state
        self.circuit.h(q)

        # Apply parameterized RZ rotations
        n_params = self.get_num_parameters()
        n_digits = len(str(n_params))
        for i in range(n_params):
            param = Parameter("init_phase_{:0{}d}".format(i, n_digits))
            self.circuit.rz(param, q[i])
