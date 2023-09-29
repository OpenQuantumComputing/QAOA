import structlog

LOG = structlog.get_logger(file=__name__)

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.circuit import Parameter
from qiskit.primitives import Sampler
from qiskit.algorithms.optimizers import COBYLA

from qaoa.initialstates import InitialState
from qaoa.mixers import Mixer
from qaoa.problems import Problem
from qaoa.util import Statistic


class OptResult:
    """
    Class for results of optimization at depth p
    """

    def __init__(self, depth):
        self.depth = depth

        self.angles = []
        self.Exp = []
        self.Var = []
        self.MaxCost = []
        self.MinCost = []
        self.shots = []

        self.index_Exp_min = -1

    def add_iteration(self, angles, stat, shots):
        self.angles.append(angles)
        self.Exp.append(-stat.get_CVaR())
        self.Var.append(stat.get_Variance())
        self.MaxCost.append(-stat.get_max())
        self.MinCost.append(-stat.get_min())
        self.shots.append(shots)

    def compute_best_index(self):
        self.index_Exp_min = self.Exp.index(min(self.Exp))

    def get_best_Exp(self):
        return self.Exp[self.index_Exp_min]

    def get_best_Var(self):
        return self.Var[self.index_Exp_min]

    def get_best_angles(self):
        return self.angles[self.index_Exp_min]

    def num_fval(self):
        return len(self.Exp)

    def num_shots(self):
        return sum(self.shots)


class QAOA:
    """Main class"""

    def __init__(
        self,
        problem,
        mixer,
        initialstate,
        backend=Aer.get_backend("qasm_simulator"),
        noisemodel=None,
        optimizer=[COBYLA, {}],  # optimizer, options
        precision=None,
        shots=1024,
        cvar=1,
    ) -> None:
        """
        A QAO-Ansatz consist of these parts:

        :problem of Basetype Problem,
        implementing the phase circuit and the cost.

        :mixer of Basetype Mixer,
        specifying the mixer circuit.

        :initialstate of Basetype InitialState,
        specifying the initial state.

        :backend: backend
        :noisemodel: noisemodel
        :optmizer: optimizer
        :precision: precision to reach for expectation value based on error=variance/sqrt(shots)
        :shots: if precision=None, the number of samples taken
                      if precision!=None, the minimum number of samples taken
        :cvar: used for CVar
        """

        assert issubclass(type(problem), Problem)
        assert issubclass(type(mixer), Mixer)
        assert issubclass(type(initialstate), InitialState)

        self.problem = problem
        self.mixer = mixer
        self.initialstate = initialstate
        self.initialstate.setNumQubits(self.problem.N_qubits)
        self.mixer.setNumQubits(self.problem.N_qubits)

        self.parameterized_circuit = None
        self.parameterized_circuit_depth = 0

        self.backend = backend
        self.optimizer = optimizer
        self.noisemodel = noisemodel
        self.shots = shots
        self.precision = precision
        self.stat = Statistic(cvar=cvar)
        self.cvar = cvar

        self.usebarrier = False
        self.isQNSPSA = False

        self.current_depth = 0  # depth at which local optimization has been done

        self.gamma_params = None
        self.beta_params = None

        self.Exp_sampled_p1 = None
        self.landscape_p1_angles = {}
        self.Var_sampled_p1 = None
        self.MaxCost_sampled_p1 = None
        self.MinCost_sampled_p1 = None

        self.optimization_results = {}

    def exp_landscape(self):
        ### at depth p = 1
        return self.Exp_sampled_p1

    def var_landscape(self):
        ### at depth p = 1
        return self.Var_sampled_p1

    def get_Exp(self, depth=None):
        if not depth:
            ret = []
            for i in range(1, self.current_depth + 1):
                ret.append(self.optimization_results[i].get_best_Exp())
            return ret
        if depth > self.current_depth + 1:
            raise ValueError
        return self.optimization_results[depth].get_best_Exp()

    def get_Var(self, depth):
        if depth > self.current_depth + 1:
            raise ValueError
        return self.optimization_results[depth].get_best_Var()

    def get_beta(self, depth):
        if depth > self.current_depth + 1:
            raise ValueError
        return self.optimization_results[depth].get_best_angles()[1::2]

    def get_gamma(self, depth):
        if depth > self.current_depth + 1:
            raise ValueError
        return self.optimization_results[depth].get_best_angles()[::2]

    def createParameterizedCircuit(self, depth):
        if self.parameterized_circuit_depth == depth:
            LOG.info(
                "Circuit is already of depth " + str(self.parameterized_circuit_depth)
            )
            return

        self.problem.create_circuit()
        self.mixer.create_circuit()
        self.initialstate.create_circuit()

        q = QuantumRegister(self.problem.N_qubits)
        c = ClassicalRegister(self.problem.N_qubits)
        self.parameterized_circuit = QuantumCircuit(q, c)

        self.gamma_params = [None] * depth
        self.beta_params = [None] * depth

        self.parameterized_circuit.compose(self.initialstate.circuit, inplace=True)

        if self.usebarrier:
            self.circuit.barrier()

        for d in range(depth):
            self.gamma_params[d] = Parameter("gamma_" + str(d))
            tmp_circuit = self.problem.circuit.assign_parameters(
                {self.problem.circuit.parameters[0]: self.gamma_params[d]},
                inplace=False,
            )
            self.parameterized_circuit.compose(tmp_circuit, inplace=True)

            if self.usebarrier:
                self.circuit.barrier()

            self.beta_params[d] = Parameter("beta_" + str(d))
            tmp_circuit = self.mixer.circuit.assign_parameters(
                {self.mixer.circuit.parameters[0]: self.beta_params[d]},
                inplace=False,
            )
            self.parameterized_circuit.compose(tmp_circuit, inplace=True)

            if self.usebarrier:
                self.circuit.barrier()

        self.parameterized_circuit.barrier()
        self.parameterized_circuit.measure(q, c)
        self.parametrized_circuit_depth = depth

    def sample_cost_landscape(
        self,
        angles={"gamma": [0, 2 * np.pi, 20], "beta": [0, 2 * np.pi, 20]},
    ):
        self.landscape_p1_angles = angles
        logger = LOG.bind(func=self.sample_cost_landscape.__name__)
        logger.info("Calculating energy landscape for depth p=1...")

        depth = 1

        tmp = angles["gamma"]
        self.gamma_grid = np.linspace(tmp[0], tmp[1], tmp[2])
        tmp = angles["beta"]
        self.beta_grid = np.linspace(tmp[0], tmp[1], tmp[2])

        if self.backend.configuration().local:
            print(depth, self.current_depth)
            self.createParameterizedCircuit(depth)
            gamma = [None] * angles["beta"][2] * angles["gamma"][2]
            beta = [None] * angles["beta"][2] * angles["gamma"][2]

            counter = 0
            for b in range(angles["beta"][2]):
                for g in range(angles["gamma"][2]):
                    gamma[counter] = self.gamma_grid[g]
                    beta[counter] = self.beta_grid[b]
                    counter += 1

            parameters = {self.gamma_params[0]: gamma, self.beta_params[0]: beta}

            logger.info("Executing sample_cost_landscape")
            logger.info(f"parameters: {len(parameters)}")
            job = execute(
                self.parameterized_circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=[parameters],
                optimization_level=0,
            )
            logger.info("Done execute")
            self.measurementStatistics(job)
            logger.info("Done measurement")

        else:
            raise NotImplementedError

        logger.info("Calculating Energy landscape done")

    def measurementStatistics(self, job):
        """
        implements a function for expectation value and variance

        :param job: job instance derived from BaseJob
        :return: expectation and variance, if the job is a list
        """
        jres = job.result()
        counts_list = jres.get_counts()
        if isinstance(counts_list, list):
            expectations = []
            variances = []
            maxcosts = []
            mincosts = []
            for i, counts in enumerate(counts_list):
                self.stat.reset()
                for string in counts:
                    # qiskit binary strings use little endian encoding, but our cost function expects big endian encoding. Therefore, we reverse the order
                    cost = self.problem.cost(string[::-1])
                    self.stat.add_sample(cost, counts[string])
                expectations.append(self.stat.get_CVaR())
                variances.append(self.stat.get_Variance())
                maxcosts.append(self.stat.get_max())
                mincosts.append(self.stat.get_min())
            angles = self.landscape_p1_angles
            self.Exp_sampled_p1 = -np.array(expectations).reshape(
                angles["beta"][2], angles["gamma"][2]
            )
            self.Var_sampled_p1 = np.array(variances).reshape(
                angles["beta"][2], angles["gamma"][2]
            )
            self.MaxCost_sampled_p1 = -np.array(maxcosts).reshape(
                angles["beta"][2], angles["gamma"][2]
            )
            self.MinCost_sampled_p1 = -np.array(mincosts).reshape(
                angles["beta"][2], angles["gamma"][2]
            )
        else:
            for string in counts_list:
                # qiskit binary strings use little endian encoding, but our cost function expects big endian encoding. Therefore, we reverse the order
                cost = self.problem.cost(string[::-1])
                self.stat.add_sample(cost, counts_list[string])

    def optimize(self, depth):
        ## run local optimization by iteratively increasing the depth until depth p is reached
        while self.current_depth < depth:
            if self.current_depth == 0:
                if self.Exp_sampled_p1 is None:
                    self.sample_cost_landscape()
                ind_Emin = np.unravel_index(
                    np.argmin(self.Exp_sampled_p1, axis=None), self.Exp_sampled_p1.shape
                )
                angles0 = np.array(
                    (self.gamma_grid[ind_Emin[1]], self.beta_grid[ind_Emin[0]])
                )
            else:
                gamma = self.optimization_results[self.current_depth].get_best_angles()[
                    ::2
                ]
                beta = self.optimization_results[self.current_depth].get_best_angles()[
                    1::2
                ]

                gamma_interp = self.interp(gamma)
                beta_interp = self.interp(beta)
                angles0 = np.zeros(2 * (self.current_depth + 1))
                angles0[::2] = gamma_interp
                angles0[1::2] = beta_interp

            self.optimization_results[self.current_depth + 1] = OptResult(
                self.current_depth + 1
            )
            # Create parameterized circuit at new depth
            self.createParameterizedCircuit(int(len(angles0) / 2))

            res = self.local_opt(angles0)

            self.optimization_results[self.current_depth + 1].compute_best_index()

            LOG.info(
                f"cost(depth { self.current_depth + 1} = {res.fun}",
                func=self.optimize.__name__,
            )

            self.current_depth += 1

    def local_opt(self, angles0):
        """

        :param angles0: initial guess
        """

        try:
            opt = self.optimizer[0](**self.optimizer[1])
        except TypeError as e:  ### QNSPSA needs fidelity
            self.isQNSPSA = True
            self.optimizer[1]["fidelity"] = self.optimizer[0].get_fidelity(
                self.parameterized_circuit, sampler=Sampler()
            )
            opt = self.optimizer[0](**self.optimizer[1])
        res = opt.minimize(self.loss, x0=angles0)
        if self.isQNSPSA:
            self.optimizer[1].pop("fidelity")
        return res

    def loss(self, angles):
        """
        loss function
        :return: an instance of the qiskit class QuantumCircuit
        """

        circuit = None
        n_target = self.shots
        self.stat.reset()
        shots_taken = 0
        shots = self.shots

        for i in range(3):  # this loop is used only used if precision is set
            if self.backend.configuration().local:
                params = self.getParametersToBind(
                    angles, self.parametrized_circuit_depth, asList=True
                )
                job = execute(
                    self.parameterized_circuit,
                    backend=self.backend,
                    noise_model=self.noisemodel,
                    shots=shots,
                    parameter_binds=[params],
                    optimization_level=0,
                )
            else:
                raise NotImplementedError
            shots_taken += shots
            self.measurementStatistics(job)
            if self.precision is None:
                break
            else:
                v = self.stat.get_Variance()
                shots = int((np.sqrt(v) / self.precision) ** 2) - shots_taken
                if shots <= 0:
                    break

        self.optimization_results[self.current_depth + 1].add_iteration(
            angles.copy(), self.stat, shots_taken
        )

        return -self.stat.get_CVaR()

    def getParametersToBind(self, angles, depth, asList=False):
        """
        Utility function to structure the parameterized parameter values
        so that they can be applied/bound to the parameterized circuit.

        :param angles: gamma and beta values
        :param depth: circuit depth
        :asList: Boolean that specify if the values in the dict should be a list or not
        :return: A dict containing parameters as keys and parameter values as values
        """
        assert len(angles) == 2 * depth

        params = {}
        for d in range(depth):
            if asList:
                params[self.gamma_params[d]] = [angles[2 * d + 0]]
                params[self.beta_params[d]] = [angles[2 * d + 1]]
            else:
                params[self.gamma_params[d]] = angles[2 * d + 0]
                params[self.beta_params[d]] = angles[2 * d + 1]
        return params

    def interp(self, angles):
        """
        INTERP heuristic/linear interpolation for initial parameters
        when going from depth p to p+1 (https://doi.org/10.1103/PhysRevX.10.021067)

        :param angles: angles for depth p
        :return: linear interpolation of angles for depth p+1
        """
        depth = len(angles)
        tmp = np.zeros(len(angles) + 2)
        tmp[1:-1] = angles.copy()
        w = np.arange(0, depth + 1)
        return w / depth * tmp[:-1] + (depth - w) / depth * tmp[1:]

    def hist(self, angles, shots):
        depth = int(len(angles) / 2)
        self.createParameterizedCircuit(depth)

        params = self.getParametersToBind(angles, depth, asList=True)
        if self.backend.configuration().local:
            job = execute(
                self.parameterized_circuit,
                self.backend,
                shots=shots,
                parameter_binds=[params],
                optimization_level=0,
            )
        else:
            raise NotImplementedError
        return job.result().get_counts()

    def random_init(self, gamma_bounds, beta_bounds, depth):
        """
        Enforces the bounds of gamma and beta based on the graph type.
        :param gamma_bounds: Parameter bound tuple (min,max) for gamma
        :param beta_bounds: Parameter bound tuple (min,max) for beta
        :return: np.array on the form (gamma_1, beta_1, gamma_2, ...., gamma_d, beta_d)
        """
        gamma_list = np.random.uniform(gamma_bounds[0], gamma_bounds[1], size=depth)
        beta_list = np.random.uniform(beta_bounds[0], beta_bounds[1], size=depth)
        initial = np.empty((gamma_list.size + beta_list.size,), dtype=gamma_list.dtype)
        initial[0::2] = gamma_list
        initial[1::2] = beta_list
        return initial
