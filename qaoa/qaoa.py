import structlog
LOG = structlog.get_logger(file=__name__)


import time
import numpy as np

from qiskit import (
    QuantumCircuit, QuantumRegister, ClassicalRegister,
    execute, Aer
)
from qiskit.circuit import Parameter
from qiskit.primitives import Sampler
from qiskit.algorithms.optimizers import COBYLA

from qaoa.mixers import Mixer
from qaoa.problems import Problem
from qaoa.util import Statistic

class QAOA:
    """ Main class
    """
    def __init__(self, problem, mixer, params=None) -> None:
        """
        A QAO-Ansatz consist of two parts:

            :problem of Basetype Problem,
            implementing the phase circuit and the cost.

            :mixer of Basetype Mixer,
            specifying the mixer circuit and the initial state.

        :params additional parameters

        :param backend: backend
        :param precision: precision to reach for expectation value based on error=variance/sqrt(shots)
        :param shots: if precision=None, the number of samples taken
                      if precision!=None, the minimum number of samples taken

        """

        assert issubclass(problem, Problem)
        assert issubclass(mixer, Mixer)

        # All values of params gets added as attributes, see end of __init__
        self.params = params


        self.problem = problem(self)
        self.mixer = mixer(self)


        self.Var = None
        self.isQNSPSA = False

        # Default values
        self.optimizer = [COBYLA, {}]
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = 1024
        self.noisemodel = None
        self.precision = None

        self.stat = Statistic(alpha=self.params.get("alpha", 1))


        self.current_depth = 0  # depth at which local optimization has been done
        self.angles_hist = {}  # initial and final angles during optimization per depth
        self.num_fval = {}  # number of function evaluations per depth
        self.t_per_fval = {}  # wall time per function evaluation per depth
        self.num_shots = (
            {}
        ) # number of total shots taken for local optimization per depth
        self.costval = {}  # optimal cost values per depth

        # Related to parameterized circuit
        self.parameterized_circuit = None
        self.parameterized_circuit_depth = 0
        self.gamma_params = None
        self.beta_params = None

        for key, val in self.params.items():
            setattr(self, key, val)

    @property
    def phase_circuit(self):
        return self.problem.phase_circuit

    @property
    def mixer_circuit(self):
        return self.mixer.mixer_circuit

    def isFeasible(self, string):
        return self.problem.isFeasible(string)

    def cost(self, string):
        return self.problem.cost(string)

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
                {self.problem.phase_circuit.parameters[0]: self.gamma_params[d]},
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

    def increase_depth(self):
        """
        sample cost landscape
        """

        t_start = time.time()
        if self.current_depth == 0:
            if self.E is None:
                self.sample_cost_landscape(
                    angles={"gamma": [0, 2 * np.pi, 20], "beta": [0, 2 * np.pi, 20]},
                )
            ind_Emin = np.unravel_index(np.argmin(self.E, axis=None), self.E.shape)
            angles0 = np.array(
                (self.gamma_grid[ind_Emin[1]], self.beta_grid[ind_Emin[0]])
            )
            self.angles_hist["d1_initial"] = angles0
        else:
            gamma = self.angles_hist["d" + str(self.current_depth) + "_final"][::2]
            beta = self.angles_hist["d" + str(self.current_depth) + "_final"][1::2]
            gamma_interp = self.interp(gamma)
            beta_interp = self.interp(beta)
            angles0 = np.zeros(2 * (self.current_depth + 1))
            angles0[::2] = gamma_interp
            angles0[1::2] = beta_interp
            self.angles_hist["d" + str(self.current_depth + 1) + "_initial"] = angles0

        self.g_it = 0
        self.g_values = {}
        self.g_angles = {}
        # Create parameterized circuit at new depth
        self.createParameterizedCircuit(int(len(angles0) / 2))

        res = self.local_opt(angles0)
        # if not res.success:
        #    raise Warning("Local optimization was not successful.", res)
        self.num_fval["d" + str(self.current_depth + 1)] = res.nfev
        self.t_per_fval["d" + str(self.current_depth + 1)] = (
            time.time() - t_start
        ) / res.nfev

        LOG.info(f"cost(depth { self.current_depth + 1} = {res.fun}", func=self.increase_depth.__name__)

        ind = min(self.g_values, key=self.g_values.get)
        self.angles_hist["d" + str(self.current_depth + 1) + "_final"] = self.g_angles[
            ind
        ]
        self.costval["d" + str(self.current_depth + 1) + "_final"] = self.g_values[ind]

        self.current_depth += 1

    def sample_cost_landscape(
            self,
            angles={"gamma": [0, 2 * np.pi, 20], "beta": [0, 2 * np.pi, 20]},
    ):
        logger = LOG.bind(func=self.sample_cost_landscape.__name__)
        logger.info("Calculating energy landscape for depth p=1...")

        depth = 1

        tmp = angles["gamma"]
        self.gamma_grid = np.linspace(tmp[0], tmp[1], tmp[2])
        tmp = angles["beta"]
        self.beta_grid = np.linspace(tmp[0], tmp[1], tmp[2])

        if self.backend.configuration().local:
            self.createParameterizedCircuit(depth)
            # parameters = []
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
            e, v = self.measurementStatistics(job)
            logger.info("Done measurement")
            self.E = -np.array(e).reshape(angles["beta"][2], angles["gamma"][2])
            self.Var = np.array(v).reshape(angles["beta"][2], angles["gamma"][2])

        else:
            raise NotImplementedError

        logger.info("Calculating Energy landscape done")


    def measurementStatistics(self, job):
        """
        implements a function for expectation value and variance

        :param job: job instance derived from BaseJob
        :return: expectation and variance
        """
        jres = job.result()
        counts_list = jres.get_counts()
        if isinstance(counts_list, list):
            expectations = []
            variances = []
            for i, counts in enumerate(counts_list):
                self.stat.reset()
                for string in counts:
                    # qiskit binary strings use little endian encoding, but our cost function expects big endian encoding. Therefore, we reverse the order
                    cost = self.cost(string[::-1])
                    self.stat.add_sample(cost, counts[string])
                expectations.append(self.stat.get_CVaR())
                variances.append(self.stat.get_Variance())
            return expectations, variances
        else:
            for string in counts_list:
                # qiskit binary strings use little endian encoding, but our cost function expects big endian encoding. Therefore, we reverse the order
                cost = self.cost(string[::-1])
                self.stat.add_sample(cost, counts_list[string])
            return self.stat.get_CVaR(), self.stat.get_Variance()

    def local_opt(self, angles0):
        """

        :param angles0: initial guess
        """

        self.num_shots["d" + str(self.current_depth + 1)] = 0

        try:
            opt = self.optimizer[0](**self.optimizer[1])
        except TypeError as e:  ### QNSPSA needs fidelity
            self.isQNSPSA=True
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
        self.g_it += 1

        # depth = int(len(angles) / 2)

        circuit = None
        n_target = self.shots
        self.stat.reset()
        shots_taken = 0
        shots = self.shots

        for i in range(3):
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
                # name = ""
                # job = start_or_retrieve_job(
                #     name + "_" + str(opt_iterations),
                #     self.backend,
                #     circuit,
                #     options={"shots": shots},
                # )
            shots_taken += shots
            _, _ = self.measurementStatistics(job)
            if self.precision is None:
                break
            else:
                v = self.stat.get_Variance()
                shots = int((np.sqrt(v) / self.precision) ** 2) - shots_taken
                if shots <= 0:
                    break

        self.num_shots["d" + str(self.current_depth + 1)] += shots_taken

        self.g_values[str(self.g_it)] = -self.stat.get_CVaR()
        self.g_angles[str(self.g_it)] = angles.copy()

        # opt_values[str(opt_iterations )] = e[0]
        # opt_angles[str(opt_iterations )] = angles
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
        E.g. [0., 2., 3., 6., 11., 0.] becomes [2., 2.75, 4.5, 7.25, 11.]

        :param angles: angles for depth p
        :return: linear interpolation of angles for depth p+1
        """
        depth = len(angles)
        tmp = np.zeros(len(angles) + 2)
        tmp[1:-1] = angles.copy()
        w = np.arange(0, depth + 1)
        return w / depth * tmp[:-1] + (depth - w) / depth * tmp[1:]

    def hist(self, angles):
        depth = int(len(angles) / 2)
        self.createParameterizedCircuit(depth)

        params = self.getParametersToBind(angles, depth, asList=True)
        if self.backend.configuration().local:
            job = execute(
                self.parameterized_circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=[params],
                optimization_level=0,
            )
        else:
            job = start_or_retrieve_job(
                "hist", self.backend, circ, options={"shots": self.shots}
            )
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






