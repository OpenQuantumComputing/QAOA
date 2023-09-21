import math
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

from .base_problem import Problem


class QUBO(Problem):
    def __init__(self, Q=None, c=None, b=None) -> None:
        """
        Implements the mapping from the parameters in params to the QUBO problem.
        Is expected to be called by the child class.
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

        # Check if Q is lower triangular
        self.lower_triangular_Q = np.allclose(Q, np.tril(Q))

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
        x = np.array(list(map(int, string)))
        return -(x.T @ self.QUBO_Q @ x + self.QUBO_c.T @ x + self.QUBO_b)

    def create_circuit(self):
        if not self.lower_triangular_Q:
            LOG.error("Function not implemented!", func=self.create_circuit.__name__)
            raise NotImplementedError
        self.createParameterizedCostCircuitTril()

    def createParameterizedCostCircuitTril(self):
        """
        Creates a parameterized circuit of the triangularized QUBO problem.
        """
        q = QuantumRegister(self.N_qubits)
        self.circuit = QuantumCircuit(q)
        cost_param = Parameter("x_gamma")

        ### cost Hamiltonian
        for i in range(self.N_qubits):
            w_i = 0.5 * (self.QUBO_c[i] + np.sum(self.QUBO_Q[:, i]))

            if not math.isclose(w_i, 0, abs_tol=1e-7):
                self.circuit.rz(cost_param * w_i, q[i])

            for j in range(i + 1, self.N_qubits):
                w_ij = 0.25 * self.QUBO_Q[j][i]

                if not math.isclose(w_ij, 0, abs_tol=1e-7):
                    self.circuit.cx(q[i], q[j])
                    self.circuit.rz(cost_param * w_ij, q[j])
                    self.circuit.cx(q[i], q[j])

    # def __str2np(self, s):
    #    x = np.array(list(map(int, s)))
    #    assert len(x) == self.N_qubits, (
    #        "bitstring  "
    #        + s
    #        + " of wrong size. Expected "
    #        + str(self.N_qubits)
    #        + " but got "
    #        + str(len(x))
    #    )
    #    return x
