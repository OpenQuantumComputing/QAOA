import math

import numpy as np

from .qubo_problem import QUBO


class PortfolioOptimization(QUBO):
    """
    Portfolio optimization QUBO.

    Subclass of the `QUBO` class, and it is ...

    Attributes:
        risk (...):
        budget (...):
        cov_matrix (...):
        exp_return (...):
        penalty (...): Defaults to 0.

    Methods:
        cost_nonQUBO(string, penalize):
        cost_nonQUBO(string, penalize):
        isFeasible(string):
        __str2np(s):


    """
    def __init__(self, risk, budget, cov_matrix, exp_return, penalty=0) -> None:
        """
        ...
        Args:
            risk (...):
            budget (...):
            cov_matrix (...):
            exp_return (...):
            penalty (...): Defaults to 0.
        """
        self.risk = risk
        self.budget = budget
        self.cov_matrix = cov_matrix
        self.exp_return = exp_return
        self.penalty = penalty
        self.N_qubits = len(self.exp_return)

        # Reformulated as a QUBO
        # min x^T Q x + c^T x + b
        # Writing Q as lower triangular matrix since it otherwise is symmetric
        Q = self.risk * np.tril(
            self.cov_matrix + np.tril(self.cov_matrix, k=-1)
        ) + self.penalty * (
            np.eye(self.N_qubits)
            + 2 * np.tril(np.ones((self.N_qubits, self.N_qubits)), k=-1)
        )
        c = -self.exp_return - (
            2 * self.penalty * self.budget * np.ones_like(self.exp_return)
        )
        b = self.penalty * self.budget * self.budget

        super().__init__(Q=Q, c=c, b=b)

    def cost_nonQUBO(self, string, penalize=True):
        """
        ...
        Args:
            string (...):
            penalize (...):

        Returns:
            -cost (...):
        """
        # risk = self.params.get("risk")
        # budget = self.params.get("budget")
        # cov_matrix = self.params.get("cov_matrix")
        # exp_return = self.params.get("exp_return")
        # penalty = self.params.get("penalty", 0.0)

        x = np.array(list(map(int, string)))
        cost = risk * (x.T @ cov_matrix @ x) - exp_return.T @ x
        if penalize:
            cost += penalty * (x.sum() - budget) ** 2

        return -cost

    def isFeasible(self, string):
        """
        ...

        Args:
            string (...):
        
        Returns:
            float(TODO ????):
        """
        x = self.__str2np(string)
        constraint = np.sum(x) - self.budget
        return math.isclose(constraint, 0, abs_tol=1e-7)

    def __str2np(self, s):
        """
        ...

        Args:
            s (...):
        
        Returns:
            x (...):
        """
        x = np.array(list(map(int, s)))
        assert len(x) == len(self.exp_return), (
            "bitstring  "
            + s
            + " of wrong size. Expected "
            + str(len(self.exp_return))
            + " but got "
            + str(len(x))
        )
        return x
