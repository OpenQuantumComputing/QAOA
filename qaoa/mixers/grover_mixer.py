import math
import itertools

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator
from qiskit.circuit.library import RZGate

from .constrained_mixer import Constrained

from qaoa.util import dicke_circuit


class Grover(Constrained):
    def __init__(self, k) -> None:
        self.k = k

    def create_circuit(self):
        self.U_s = dicke_circuit.DickeCircuit(
            self.N_qubits, self.k
        )  # .decompose().decompose().decompose()
        self.U_s_dagger = self.U_s.inverse()

        q = QuantumRegister(self.N_qubits)
        self.circuit = QuantumCircuit(q)

        Beta = Parameter("x_beta")
        rz = RZGate(Beta).control(self.N_qubits - 1)

        self.circuit.barrier()
        self.circuit.compose(self.U_s_dagger, q, inplace=True)
        self.circuit.barrier()
        self.circuit.x(range(self.N_qubits))
        self.circuit.barrier()
        self.circuit.append(rz, self.circuit.qubits)
        self.circuit.barrier()
        self.circuit.x(range(self.N_qubits))
        self.circuit.barrier()
        self.circuit.compose(self.U_s, self.circuit.qubits, inplace=True)
        self.circuit.barrier()

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

    # def compute_feasible_subspace(self):
    #    print("Its now computing the feasible subspace")
    #    self.B.clear()
    #    for combination in itertools.combinations(
    #        range(self.params["N_qubits"]), self.k
    #    ):
    #        current_state = ["0"] * self.params["N_qubits"]
    #        for index in combination:
    #            current_state[index] = "1"
    #        self.B.append("".join(current_state))

    # def isFeasible(self, string):
    #    x = self.__str2np(string)
    #    constraint = np.sum(x) - self.k
    #    return math.isclose(constraint, 0, abs_tol=1e-7)
