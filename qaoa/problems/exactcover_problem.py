import math

import numpy as np

from .base_problem import Problem
from qiskit import QuantumCircuit, QuantumRegister

from qiskit.circuit import Parameter


class ExactCover(Problem):
    """
    Exact cover problem.

    Subclass of the `Problem` class. Contains the methods to create the exact cover problem, which is the problem of whether 
    it is possible to cover all elements of a set exactly once by using some subsets.

    Attributes:
        columns (np.ndarray): Matrix where each column represents a subset.
        weights (np.ndarray or None): Optional weights for each subset. Defaults to None.
        penalty_factor (float or int): Penalty factor for constraint violations. Defaults to 1.

    Methods:
        cost(): Calculates the cost of a given solution.
        create_circuit(): Creates a parameterized circuit corresponding to the cost function.
        isFeasible(): Checks if a given bitstring represents a feasible solution to the problem.
        _exactCover(): Computes the penalty for a given solution vector x, measuring how far it is from being an exact cover.
    """
    def __init__(
        self,
        columns,
        weights=None,
        penalty_factor=1,
        allow_infeasible = False
    ) -> None:
        """
        Args:
            columns (np.ndarray): Matrix where each column represents a subset.
            weights (np.ndarray or None): Optional weights for each subset. Defaults to None.
            penalty_factor (float or int): Penalty factor for constraint violations. Defaults to 1.
        """
        super().__init__()
        self.columns = columns
        self.weights = weights
        self.penalty_factor = penalty_factor
        self.allow_infeasible = allow_infeasible

        colSize = columns.shape[0]  ### Size per column
        numColumns = columns.shape[1]  ### number of columns/qubits

        self.N_qubits = numColumns

    def cost(self, string):
        """
        Calculates the cost so that states where an element is not covered, or covered more than once, will be penalized, whereas
        sets that contain elements that are covered exactly once are favored.

        Args:
            string (str): Bitstring representing a candidate solution.
        """
        x = np.array(list(map(int, string)))
        c_e = self.__exactCover(x)

        if self.weights is None:
            return -self.penalty_factor * c_e
        else:
            return -(self.weights @ x + self.penalty_factor * c_e)

    def create_circuit(self):
        """
        Creates a parameterized quantum circuit corresponding to the cost function.
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
        Checks if a given bitstring represents a feasible solution to the exact cover problem.

        Args:
            string (str): Bitstring representing a candidate solution.
        """
        x = np.array(list(map(int, string)))
        c_e = self.__exactCover(x)
        return math.isclose(c_e, 0, abs_tol=1e-7) or self.allow_infeasible

    def __exactCover(self, x):
        """
        Computes the penalty for a given solution vector x, measuring how far it is from being an exact cover.
        
        Args:
            x (np.ndarray): Binary vector representing a candidate solution.
        """
        return np.sum((1 - (self.columns @ x)) ** 2)
