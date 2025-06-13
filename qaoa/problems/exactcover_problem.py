import math

import numpy as np

from .base_problem import Problem
from qiskit import QuantumCircuit, QuantumRegister

from qiskit.circuit import Parameter


class ExactCover(Problem):
    """
    Exact cover problem.

    Subclass of the `Problem` class, and it ...

    Attributes:
        columns (arr):
        weights (float TODO???): Defaults to None.
        penalty_factor (int TODO???): Defaults to 1.

    Methods:
        cost():
        create_circuit():
        isFeasible():
        _exactCover():
    """
    def __init__(
        self,
        columns,
        weights=None,
        penalty_factor=1,
    ) -> None:
        """
        Args:
            columns (arr):
            weights: Defaults to None.
            penalty_factor: Defaults to 1.
        """
        super().__init__()
        self.columns = columns
        self.weights = weights
        self.penalty_factor = penalty_factor

        colSize = columns.shape[0]  ### Size per column
        numColumns = columns.shape[1]  ### number of columns/qubits

        self.N_qubits = numColumns

    def cost(self, string):
        """

        """
        x = np.array(list(map(int, string)))
        c_e = self.__exactCover(x)

        if self.weights is None:
            return -c_e
        else:
            return -(self.weights @ x + self.penalty_factor * c_e)

    def create_circuit(self):
        """
        Creates parameterized circuit corresponding to the cost function
        """
        q = QuantumRegister(self.N_qubits)
        self.circuit = QuantumCircuit(q)
        cost_param = Parameter("x_gamma")

        colSize, numColumns = np.shape(self.columns)

        ### cost Hamiltonian
        for col in range(numColumns):
            hr = (
                self.penalty_factor
                * 0.5
                * self.columns[:, col]
                @ (np.sum(self.columns, axis=1) - 2)
            )
            if not self.weights is None:
                hr += 0.5 * self.weights[col]

            if not math.isclose(hr, 0, abs_tol=1e-7):
                self.circuit.rz(cost_param * hr, q[col])

            for col_ in range(col + 1, numColumns):
                Jrr_ = (
                    self.penalty_factor
                    * 0.5
                    * self.columns[:, col]
                    @ self.columns[:, col_]
                )

                if not math.isclose(Jrr_, 0, abs_tol=1e-7):
                    self.circuit.cx(q[col], q[col_])
                    self.circuit.rz(cost_param * Jrr_, q[col_])
                    self.circuit.cx(q[col], q[col_])

    def isFeasible(self, string):
        """
        """
        x = np.array(list(map(int, string)))
        c_e = self.__exactCover(x)
        return math.isclose(c_e, 0, abs_tol=1e-7)

    def __exactCover(self, x):
        """

        """
        return np.sum((1 - (self.columns @ x)) ** 2)
