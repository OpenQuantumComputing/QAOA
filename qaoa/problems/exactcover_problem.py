import math

import numpy as np

# from .base_problem import Problem
from .qubo_problem import QUBO
from qiskit import QuantumCircuit, QuantumRegister

from qiskit.circuit import Parameter


class ExactCover(QUBO):
    """
    Exact cover problem.

    Subclass of the `Problem` class. Contains the methods to create the exact cover problem, which is the problem of whether 
    it is possible to cover all elements of a set exactly once by using some subsets.

    Attributes:
        columns (np.ndarray): Matrix where each column represents a subset.
        weights (np.ndarray or None): Optional weights for each subset. Defaults to None.
        penalty_factor (float or int): Penalty factor for constraint violations. Defaults to 1.
        hamming_weight (int): If the solution has a known Hamming weight (valid only between 0 and len(weights))
        scale_problem (bool): To scale the problem or not. Only implemented for problems with Hamming weight, and defaults therefore to false.

    Methods:
        cost(): Calculates the cost of a given solution.
        create_circuit(): Creates a parameterized circuit corresponding to the cost function.
        isFeasible(): Checks if a given bitstring represents a feasible solution to the problem.
        _exactCover(): Computes the penalty for a given solution vector x, measuring how far it is from being an exact cover.
        _scale_problem(hamming_weight): Scales the problem to a range of (0, 2 pi). Only implemented for known hamming weights
    """
    def __init__(
        self,
        columns,
        weights=None,
        penalty_factor=None,
        allow_infeasible = False,
        hamming_weight = None,
        scale_problem = False
    ) -> None:
        """
        Args:
            columns (np.ndarray): Matrix where each column represents a subset.
            weights (np.ndarray or None): Optional weights for each subset. Defaults to None.
            penalty_factor (float, int or None): Penalty factor for constraint violations. If None (default) it is constructed as sum(abs(weights)).
        """
        self.columns = columns
        self.weights = weights
        self.penalty_factor = penalty_factor
        self.allow_infeasible = allow_infeasible
        
        colSize = columns.shape[0]  ### Size per column
        numColumns = columns.shape[1]  ### number of columns/qubits

        if weights is None:
            self.weights = np.zeros(numColumns)

        if hamming_weight is not None:
            assert type(hamming_weight) == int, "hamming_weight must be int, but is "+str(hamming_weight)
            assert hamming_weight > 0 and hamming_weight < numColumns, "hamming_weight must between 0 and "+str(numColumns)+", but is "+str(hamming_weight)
        self.hamming_weight = hamming_weight

        if penalty_factor is None:
            if np.all(self.weights == 0):
                self.penalty_factor = 1
            elif self.hamming_weight:
                sorted_w = np.sort(self.weights)
                self.penalty_factor = np.sum(sorted_w[-hamming_weight:]) - np.sum(sorted_w[:hamming_weight])
            else:
                # Very conservative penalty term
                self.penalty_factor = np.sum(self.weights[self.weights > 0]) 
        
        if scale_problem:
            # Scaling through modification of self.weights and self.penalty_factor
            self._scale_problem()
 
        # Construct a QUBO for the penalized exact cover problem
        # C(x) = x^T Q x + c^T x + b
        c = self.weights - 2*self.penalty_factor*(np.ones(colSize) @ self.columns)
        Q = self.penalty_factor * (self.columns.T @ self.columns)
        b = self.penalty_factor*colSize

        assert(Q.shape == (numColumns, numColumns))
        assert(len(c) == numColumns)

        super().__init__(Q, c, b)

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

        return -(self.weights @ x + self.penalty_factor * c_e)


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

    def _scale_problem(self):
        assert (self.hamming_weight is not None), "scaling currently only supported for exact problems with a given hamming_weight"
        hm = self.hamming_weight
        col_size = self.columns.shape[0]
        
        # Ensure lower bound of cost function is zero
        w = np.copy(self.weights)
        sorted_w = np.sort(w)
        omega = w - np.sum(sorted_w[:hm])/hm

        # find new penalty
        sorted_omega = np.sort(omega)
        omega_penalty = np.sum(sorted_omega[-hm:]) - np.sum(sorted_omega[:hm])

        # conservative estimate of upper range
        upper_range = 2*np.pi
        scaling = upper_range/(omega_penalty*(1 + col_size*(hm - 1)**2))
        
        # apply scaling
        self.weights = scaling*omega
        self.penalty_factor = scaling*omega_penalty
