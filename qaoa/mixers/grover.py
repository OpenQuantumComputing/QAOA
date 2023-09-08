import math
import itertools

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator
from qiskit.circuit.library import RZGate

from .Basemixer import Constrained

from qaoa.util import dicke_state


class Grover(Constrained):
    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.__check_params()
        self.k = self.params["k"]
        self.n = self.params["N_qubits"]

        self.U_s = dicke_state(self.n, self.k)
        self.U_s_dagger = self.U_s.inverse()


    def set_initial_state(self, circuit, qubit_register):
        circuit.compose(self.U_s, qubit_register, inplace=True)

    def create_mixer(self):
        q = QuantumRegister(self.params['N_qubits'])
        self.mixer_circuit = QuantumCircuit(q)
        # self.best_mixer_terms, self.logical_X_operators = self.__XYMixerTerms()

        Beta = Parameter("x_beta")
        rz = RZGate(Beta).control(self.n - 1)

        usebarrier = self.params.get("usebarrier", False)

        self.mixer_circuit.compose(self.U_s_dagger, q, inplace=True)
        self.mixer_circuit.x(range(self.n))
        self.mixer_circuit.append(rz, self.mixer_circuit.qubits)
        self.mixer_circuit.x(range(self.n))
        self.mixer_circuit.compose(self.U_s, self.mixer_circuit.qubits, inplace=True)


        if usebarrier:
            self.mixer_circuit.barrier()

    def compute_feasible_subspace(self):
        print("Its now computing the feasible subspace")
        self.B.clear()
        for combination in itertools.combinations(range(self.params['N_qubits']), self.k):
            current_state = ["0"] * self.params['N_qubits']
            for index in combination:
                current_state[index] = "1"
            self.B.append("".join(current_state))

    def isFeasible(self, string):
        x = self.__str2np(string)
        constraint = np.sum(x) - self.k
        return math.isclose(constraint, 0, abs_tol=1e-7)


    def __check_params(self):
        for key in ["k", "N_qubits"]:
            assert key in self.params, "Problem or params need to specify " + key

