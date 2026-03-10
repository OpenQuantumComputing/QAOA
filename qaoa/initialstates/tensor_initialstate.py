import numpy as np
from copy import deepcopy

from qiskit import QuantumRegister, QuantumCircuit

from .base_initialstate import InitialState


class Tensor(InitialState):
    """
    Tensor initial state.

    Subclass of the `InitialState` class that creates a tensor product of
    *num* copies of a sub-circuit (initial state **or** mixer) placed
    side-by-side on disjoint qubit registers.

    When drawn, each copy is shown as a labelled box using the sub-circuit's
    own ``label`` attribute, making the "lego" structure immediately visible.

    Attributions:
        subcircuit (InitialState): The circuit that is to be tensorised.
        num (int): Number of copies of the sub-circuit.

    Methods:
        create_circuit(): Creates the tensorised circuit.
    """

    def __init__(self, subcircuit: InitialState, num: int, label: str | None = None) -> None:
        """
        Args:
            subcircuit (InitialState): The circuit that is to be tensorised.
            num (int): Number of copies of the sub-circuit.
            label (str | None): Optional annotation label.  Defaults to
                ``"Tensor"``.
        """
        super().__init__(label=label)
        self.num = num
        self.subcircuit = subcircuit
        self.N_qubits = self.num * self.subcircuit.N_qubits

    def create_circuit(self) -> None:
        """
        Creates a circuit that places *num* copies of the sub-circuit on
        disjoint qubit registers.

        Each copy is wrapped as a labelled instruction so that circuit
        drawings show named "lego" blocks instead of raw gates.
        """
        self.subcircuit.create_circuit()
        sub_label = getattr(self.subcircuit, "label", self.subcircuit.__class__.__name__)
        sub_instr = self.subcircuit.circuit.to_instruction(label=sub_label)

        qr = QuantumRegister(self.N_qubits, name="q")
        self.circuit = QuantumCircuit(qr)
        n = self.subcircuit.N_qubits
        for i in range(self.num):
            self.circuit.append(sub_instr, list(range(i * n, (i + 1) * n)))
