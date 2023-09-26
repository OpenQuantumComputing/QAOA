from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import XXPlusYYGate

from .constrained_mixer import Constrained

import math
import itertools

import numpy as np


class PauliString:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError


class XY(Constrained):
    def __init__(self, method="chain") -> None:
        self.method = method

    def create_circuit(self):
        q = QuantumRegister(self.N_qubits)
        self.circuit = QuantumCircuit(q)
        # self.best_mixer_terms, self.logical_X_operators = self.__XYMixerTerms()

        Beta = Parameter("x_beta")
        scale = 0.5  # Since every logical X has two stabilizers
        for i in range(self.N_qubits - 1):
            # Hard coded XY mixer
            current_gate = XXPlusYYGate(scale * Beta)
            self.circuit.append(current_gate, [i, i + 1])
        if self.method == "ring":
            current_gate = XXPlusYYGate(scale * Beta)
            self.circuit.append(current_gate, [self.N_qubits - 1, 0])

    def compute_feasible_subspace(self):
        print("Its now computing the feasible subspace")
        self.B.clear()
        for combination in itertools.combinations(range(self.N_qubits), self.k):
            current_state = ["0"] * selfN_qubits
            for index in combination:
                current_state[index] = "1"
            self.B.append("".join(current_state))

    def isFeasible(self, string):
        x = self.__str2np(string)
        constraint = np.sum(x) - self.k
        return math.isclose(constraint, 0, abs_tol=1e-7)

    def __XYMixerTerms(self):
        logical_X_operators = [None] * (selfN_qubits - 1)
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
