import math
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

from .base_problem import Problem

import structlog
LOG = structlog.get_logger(file=__name__)


class QUBO(Problem):
    """
    Quadratic Unconstrained Binary Optimization (QUBO) problem.

    Subclass of the `Problem` class. This class represents a generic QUBO problem, which can be used as a base for more specific QUBO-based problems.
    The QUBO problem is defined as minimizing a quadratic function over binary variables.

    Attributes:
        Q (np.ndarray): A 2-dimensional numpy ndarray representing the quadratic coefficients.
        c (np.ndarray): A 1-dimensional numpy ndarray representing the linear coefficients.
        b (float): Scalar offset term.
        N_qubits (int): Number of binary variables/qubits in the problem.
        is_lower_triangular_Q (bool): Whether Q is lower triangular.
        QUBO_Q (np.ndarray): The quadratic coefficient matrix.
        QUBO_c (np.ndarray): The linear coefficient vector.
        QUBO_b (float): The scalar offset.

    Methods:
        cost(string): Computes the cost of a given binary string according to the QUBO formulation.
        create_circuit(): Creates a parametrized quantum circuit corresponding to the cost function of the QUBO problem.
        createParameterizedCostCircuitTril(): Creates a parameterized circuit of the triangularized QUBO problem.
    """
    def __init__(self, Q=None, c=None, b=None) -> None:
        super().__init__()
        """
        Implements the mapping from the parameters in params to the QUBO problem.
        Is expected to be called by the child class.

        # The QUBO will be on this form:
        # min x^T Q x + c^T x + b

        Args:
            Q (np.ndarray): A 2-dimensional numpy ndarray representing the quadratic coefficients.
            c (np.ndarray): A 1-dimensional numpy ndarray representing the linear coefficients. Defaults to None.
            b (float): Scalar offset term. Defaults to None.

        Raises:
            AssertionError: If Q is not a square 2D numpy ndarray.
            AssertionError: If c is not a 1D numpy ndarray of compatible size.
            AssertionError: If b is not a scalar.
        """
        assert type(Q) is np.ndarray, "Q needs to be a numpy ndarray, but is " + str(
            type(Q)
        )
        assert (
            Q.ndim == 2
        ), "Q needs to be a 2-dimensional numpy ndarray, but has dim " + str(Q.ndim)
        assert Q.shape[0] == Q.shape[1], "Q needs to be a square matrix, but is " + str(
            Q.shape
        )
        n = Q.shape[0]

        self.N_qubits = n

        self.QUBO_Q = Q

        if c is None:
            c = np.zeros(n)
        assert type(c) is np.ndarray, "c needs to be a numpy ndarray, but is " + str(
            type(c)
        )
        assert (
            c.ndim == 1
        ), "c needs to be a 1-dimensional numpy ndarray, but has dim " + str(Q.ndim)
        assert c.shape[0] == n, (
            "c is of size "
            + str(c.shape[0])
            + " but should be compatible size to Q, meaning "
            + str(n)
        )
        self.QUBO_c = c

        if b is None:
            b = 0.0
        assert np.isscalar(b), "b is expected to be scalar, but is " + str(b)
        self.QUBO_b = b

    def cost(self, string):
        """
        Computes the cost of a given binary string according to the QUBO formulation.

        Args:
            string (str): Binary string representing a candidate solution to the QUBO problem.

        Returns:
            float: The cost of the solution.
        """
        return self.qubo_cost(string)

    def qubo_cost(self, string):
        """
        For other other problems that are mapped to a QUBO, it is often useful to compare that the 
        original cost function is equivalent to the qubo-transformed cost function.
        This wrapper enables that validation check
        """
        x = np.array(list(map(int, string)))
        return -(x.T @ self.QUBO_Q @ x + self.QUBO_c.T @ x + self.QUBO_b)


    def create_circuit(self):
        """
        Creates a parameterized quantum circuit that corresponds to the QUBO cost function
        C(x) = x^T Q x + c^T x 
        """

        # LOG.info("Creating parameterized cirquit for generic QUBO problem")

        # To simplify notation:
        N = self.N_qubits
        Q = self.QUBO_Q
        c = self.QUBO_c
        gamma = Parameter("x_gamma")

        # Ensure that Q is symmetric and add c to its diagonal
        Q = 0.5*(Q+Q.T) + np.diag(c)

        # Creating parameters b for rz gates, 
        # where b_i = -sum_j(Q_ij + Q_ji) = -2*sum(Q[:,i]) from symmetry
        b = -2.0 * Q.sum(axis=0)

        q = QuantumRegister(N)
        self.circuit = QuantumCircuit(q)


        # Add rz gates:
        # Note that we use multiply with factor 2 since qiskit implements exp^(-i theta/2 Z)
        for i in range(N):
            if not math.isclose(b[i], 0, abs_tol=1e-10):
                # LOG.info("Adding rz gate to qubit "+str(i)+" with parameter b[i] = "+str(b[i]))
                self.circuit.rz(2*0.25*b[i]*gamma, i)

        # Add rzz gates:
        # Since Q now is symmetric, we apply the RZZ-gate only to its lower triangular part while multiplying by 2 
        # We also multiply with factor 2 since qiskit implements exp^(-i theta/2 Z tensor Z)
        for i in range(N):
            for j in range(i+1, N):
                if not (math.isclose(Q[i, j], 0.0, abs_tol=1e-10)):
                    # LOG.info("Adding rzz gate between qubits "+str(i)+" and "+str(j)+" with parameter Q[i,j] = "+str(Q[i, j]))
                    self.circuit.rzz(2*2*0.25*Q[i,j]*gamma, i, j)        

    def validate_circuit(self, t=1, flip=True, atol=1e-8, rtol=1e-8):
        """
        Validates two elements:

        1) That the QUBO cost function (self.qubo_cost) is equivalent to the problem-specific 
        cost function (self.cost) 

        2) Exact check that the problem's circuit represents the problem's cost function.
        This tests checks that the unitary operator represented by the quantum circuit is
        equal to the excepted matrix with diagonal elements 
        exp(-j*t*cost(e)),
        where e is the corresponding binary state, up to a global phase.
        
        Suitable for <= 10 qubits as this check uses the full unitary matrix of size 2^n x 2^n).
        Returns: (ok: bool, report: dict)
        """

        # Validate cost function transformation:
        qubo_mapping_errors = 0
        max_abs_error = 0
        mismatches = []
        n = self.N_qubits
        for i in range(2**n):
            bitstring = format(i, f'0{n}b')
            cost = self.cost(bitstring)
            qubo_cost = self.qubo_cost(bitstring)
            abs_error = np.abs(cost - qubo_cost)
            if (abs_error > atol):
                qubo_mapping_errors += 1
                max_abs_error = max(max_abs_error, abs_error)
                if qubo_mapping_errors < 9:
                    mismatches.append({
                        "bitstring": list(bitstring),
                        "cost": cost,
                        "qubo_cost": qubo_cost,
                        "abs_error": abs_error
                    })

        # If this first test fails, there is no need to check the circuit --> return a report
        if qubo_mapping_errors > 0:
            report = {
                "n_qubits": self.N_qubits,
                "max_error": abs_error,
                "examples": mismatches
            }
            return False, report

        # Validate mapping from cost function to quantum circuit:
        circ_ok, circ_report = super().validate_circuit(t=t, flip=flip, atol=atol, rtol=rtol)
        circ_report["max_qubo_cost_vs_cost_error"] = max_abs_error
        return circ_ok, circ_report