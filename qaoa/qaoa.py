import structlog
LOG = structlog.get_logger(file=__name__)

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter

from qaoa.mixers import Mixer
from qaoa.problems import Problem

class QAOA:
    """ Main class
    """
    def __init__(self, problem, mixer, params=None) -> None:
        """
        A QAO-Ansatz consist of two parts:

            :param problem of Basetype Problem,
            implementing the phase circuit and the cost.

            :param mixer of Basetype Mixer,
            specifying the mixer circuit and the initial state.

        :params additional parameters

        :param backend: backend
        :param precision: precision to reach for expectation value based on error=variance/sqrt(shots)
        :param shots: if precision=None, the number of samples taken
                      if precision!=None, the minimum number of samples taken

        """

        assert issubclass(problem, Problem)
        assert issubclass(mixer, Mixer)

        self.params = params

        self.problem = problem(self.params)
        self.mixer = mixer(self.params)




        # Related to parameterized circuit
        self.parameterized_circuit = None
        self.parameterized_circuit_depth = 0
        self.gamma_params = None
        self.beta_params = None

    def createParameterizedCircuit(self, depth):
        if self.parameterized_circuit_depth == depth:
            LOG.info("Circuit is already of depth " + str(self.parameterized_circuit_depth))
            return

        self.problem.create_phase()
        self.mixer.create_mixer()

        q = QuantumRegister(self.problem.N_qubits)
        c = ClassicalRegister(self.problem.N_qubits)
        self.parameterized_circuit = QuantumCircuit(q, c)

        self.gamma_params = [None] * depth
        self.beta_params = [None] * depth

        ### Initial state
        self.mixer.set_initial_state(self.parameterized_circuit, q)

        for d in range(depth):
            self.gamma_params[d] = Parameter("gamma_" + str(d))
            cost_circuit = self.problem.phase_circuit.assign_parameters(
                {self.cost_circuit.parameters[0]: self.gamma_params[d]},
                inplace=False,
            )
            self.parameterized_circuit.compose(cost_circuit, inplace=True)

            self.beta_params[d] = Parameter("beta_" + str(d))
            mixer_circuit = self.mixer.mixer_circuit.assign_parameters(
                {self.mixer.mixer_circuit.parameters[0]: self.beta_params[d]},
                inplace=False,
            )
            self.parameterized_circuit.compose(mixer_circuit, inplace=True)

        self.parameterized_circuit.barrier()
        self.parameterized_circuit.measure(q, c)
        self.parametrized_circuit_depth = depth

