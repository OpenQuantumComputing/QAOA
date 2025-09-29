import math

import numpy as np
import itertools

from .qubo_problem import QUBO


class PortfolioOptimization(QUBO):
    """
    Portfolio optimization QUBO.

    Subclass of the `QUBO` class. It reformulates the portfolio optimization problem as a QUBO problem, where the goal is to maximize the expected return while minimizing the risk, subject to a budget constraint.

    Attributes:
        risk (float): Risk aversion parameter (weight for the risk term).
        budget (int): The total number of assets to select (budget constraint).
        cov_matrix (np.ndarray): Covariance matrix of asset returns.
        exp_return (np.ndarray): Expected returns for each asset.
        penalty (float): Penalty parameter for enforcing the budget constraint. Defaults to 0.
        N_qubits (int): Number of assets/qubits in the problem.

    Methods:
        cost_nonQUBO(string, penalize): Computes the cost of a given portfolio bitstring, optionally including the penalty term.
        isFeasible(string): Checks if a given bitstring satisfies the budget constraint.
        __str2np(s): Converts a bitstring to a numpy array of integers.
    """
    def __init__(self, risk, budget, cov_matrix, exp_return, penalty=0) -> None:
        """
        
        Args:
            risk (float): Risk aversion parameter (weight for the risk term).
            budget (int): The total number of assets to select (budget constraint).
            cov_matrix (np.ndarray): Covariance matrix of asset returns.
            exp_return (np.ndarray): Expected returns for each asset.
            penalty (float): Penalty parameter for enforcing the budget constraint. Defaults to 0.
        """
        self.risk = risk
        self.budget = budget
        self.cov_matrix = cov_matrix
        self.exp_return = exp_return
        self.penalty = penalty
        self.N_qubits = len(self.exp_return)

        # Reformulated as a QUBO
        # min x^T Q x + c^T x + b
        
        Q = self.risk* self.cov_matrix + self.penalty*np.ones_like(self.cov_matrix)

        c = -self.exp_return - 2*self.penalty*self.budget*np.ones_like(self.exp_return)

        b = self.penalty * self.budget * self.budget

        super().__init__(Q=Q, c=c, b=b)

    def cost(self, string, penalize=True):
        """
        Computes the cost of a given portfolio bitstring, optionally including the penalty term for the budget constraint.

        Args:
            string (str): Bitstring representing the selected assets (portfolio).
            penalize (bool): Whether to include the penalty term for violating the budget constraint.

        Returns:
            cost (float): The negative of the portfolio objective value.
        """
        # risk = self.params.get("risk")
        # budget = self.params.get("budget")
        # cov_matrix = self.params.get("cov_matrix")
        # exp_return = self.params.get("exp_return")
        # penalty = self.params.get("penalty", 0.0)

        x = np.array(list(map(int, string)))
        cost = self.risk * (x.T @ self.cov_matrix @ x) - self.exp_return.T @ x
        #if penalize:
        cost += self.penalty * (x.sum() - self.budget) ** 2

        return -cost

    def isFeasible(self, string):
        """
        Checks if a given bitstring satisfies the budget constraint.

        Args:
            string (str): Bitstring representing the selected assets (portfolio).
        
        Returns:
            bool: True if the bitstring satisfies the budget constraint, False otherwise.
        """
        x = self.__str2np(string)
        constraint = np.sum(x) - self.budget
        return math.isclose(constraint, 0, abs_tol=1e-7)


    def brute_force_solve(self):
        """
        Finding optimal solution using brute force tactics by checking all feasible solutions.
        """
        
        def bitstrings_hamming_weight_generator(n, k):
            for ones in itertools.combinations(range(n), k):
                s = ['0'] * n
                for i in ones:
                    s[i] = '1'
                yield ''.join(s)
    
        opt_val = -np.inf
        opt_sol = None
        for bs in bitstrings_hamming_weight_generator(self.N_qubits, self.budget):
            cost = self.cost(bs)
            if cost > opt_val:
                opt_val = cost
                opt_sol = bs
        
        return opt_sol


    def __str2np(self, s):
        """
        Converts a bitstring to a numpy array of integers.

        Args:
            s (str): Bitstring representing the selected assets (portfolio).
        
        Returns:
            x (np.ndarray): Numpy array of integers corresponding to the bitstring.
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
