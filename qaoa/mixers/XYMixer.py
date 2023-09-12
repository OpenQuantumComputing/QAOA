import math
import itertools

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister

from qiskit.circuit import Parameter
from .BaseMixer import Constrained

from qiskit.circuit.library import XXPlusYYGate


class PauliString:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError


class XY(Constrained):
    def __init__(self, parent) -> None:
        super().__init__(parent)

    def create_mixer(self):
        q = QuantumRegister(self.params["N_qubits"])
        self.mixer_circuit = QuantumCircuit(q)
        # self.best_mixer_terms, self.logical_X_operators = self.__XYMixerTerms()

        Beta = Parameter("x_beta")
        scale = 0.5  # Since every logical X has two stabilizers
        for i in range(self.params["N_qubits"] - 1):
            # Hard coded XY mixer
            current_gate = XXPlusYYGate(scale * Beta)
            self.mixer_circuit.append(current_gate, [i, i + 1])

        usebarrier = self.params.get("usebarrier", False)
        if usebarrier:
            self.mixer_circuit.barrier()

    def compute_feasible_subspace(self):
        print("Its now computing the feasible subspace")
        self.B.clear()
        for combination in itertools.combinations(
            range(self.params["N_qubits"]), self.k
        ):
            current_state = ["0"] * self.params["N_qubits"]
            for index in combination:
                current_state[index] = "1"
            self.B.append("".join(current_state))

    def isFeasible(self, string):
        x = self.__str2np(string)
        constraint = np.sum(x) - self.k
        return math.isclose(constraint, 0, abs_tol=1e-7)

    def __XYMixerTerms(self):
        logical_X_operators = [None] * (self.params["N_qubits"] - 1)
        mixer_terms = {}
        scale = 0.5  # 1/size, size of stabilizer space
        for i in range(self.params["N_qubits"] - 2):
            logical_X_operator = ["I"] * (self.params["N_qubits"] - 1)
            logical_X_operator[i] = "X"
            logical_X_operator[i + 1] = "X"
            logical_X_operator = "".join(logical_X_operator)
            logical_X_operators[i] = logical_X_operator

            mixer_terms[logical_X_operator] = [PauliString(scale, logical_X_operator)]

            YY_operator = ["I"] * (self.params["N_qubits"] - 1)
            YY_operator[i] = "Y"
            YY_operator[i + 1] = "Y"
            YY_operator = "".join(YY_operator)

            mixer_terms[logical_X_operator].append(PauliString(scale, YY_operator))

        return mixer_terms, logical_X_operators
