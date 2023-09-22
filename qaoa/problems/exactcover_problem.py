import math

import numpy as np

from .base_problem import Problem
from qiskit import QuantumCircuit, QuantumRegister

from qiskit.circuit import Parameter


class ExactCover(Problem):
    def __init__(
        self,
        FR,
        CR=None,
        mu=1,
    ) -> None:
        self.FR = FR
        self.CR = CR
        self.mu = 1

        fN = FR.shape[0]  ### number of flights
        rN = FR.shape[1]  ### number of routes

        self.N_qubits = rN

    def cost(self, string):
        x = np.array(list(map(int, string)))
        c_e = self.__exactCover(x)

        if self.CR is None:
            return -c_e
        else:
            return -(self.CR @ x + self.mu * c_e)

    def create_circuit(self):
        """
        Creates parameterized circuit corresponding to the cost function
        """
        q = QuantumRegister(self.N_qubits)
        self.circuit = QuantumCircuit(q)
        cost_param = Parameter("x_gamma")

        F, R = np.shape(self.FR)

        ### cost Hamiltonian
        for r in range(R):
            hr = self.mu * 0.5 * self.FR[:, r] @ (np.sum(self.FR, axis=1) - 2)
            if not self.CR is None:
                hr += 0.5 * self.CR[r]

            if not math.isclose(hr, 0, abs_tol=1e-7):
                self.circuit.rz(cost_param * hr, q[r])

            for r_ in range(r + 1, R):
                Jrr_ = self.mu * 0.5 * self.FR[:, r] @ self.FR[:, r_]

                if not math.isclose(Jrr_, 0, abs_tol=1e-7):
                    self.circuit.cx(q[r], q[r_])
                    self.circuit.rz(cost_param * Jrr_, q[r_])
                    self.circuit.cx(q[r], q[r_])

    def isFeasible(self, string):
        x = np.array(list(map(int, string)))
        c_e = self.__exactCover(x)
        return math.isclose(c_e, 0, abs_tol=1e-7)

    def __exactCover(self, x):
        return np.sum((1 - (self.FR @ x)) ** 2)
