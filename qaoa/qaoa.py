import structlog

LOG = structlog.get_logger(file=__name__)

import numpy as np

from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    AncillaRegister,
    transpile,
)
from qiskit.circuit import Parameter
from qiskit.primitives import Sampler
from qiskit_algorithms.optimizers import COBYLA

from qiskit_aer import Aer

from qaoa.initialstates import InitialState
from qaoa.mixers import Mixer
from qaoa.problems import Problem
from qaoa.util import Statistic, BitFlip, post_processing


class OptResult:
    """
    Class for results of optimization at depth p.

    Attributes:
        depth (int): The depth p of the optimization.
        angles (list): List of angles used in the optimization.
        Exp (list): List of expected values.
        Var (list): List of variances.
        WorstCost (list): List of worst costs.
        BestCost (list): List of best costs.
        BestSols (list): List of best solutions.
        shots (list): List of shots taken for each iteration.
        index_Exp_min (int): Index of the minimum expected value.

    Methods:
        add_iteration(): Adds an iteration's results to the OptResult object.
        compute_best_index(): Computes the index of the minimum expected value in `Exp`.
        get_best_Exp(): Returns the best expected value.
        get_best_Var(): Returns the best variance.
        get_best_angles(): Returns the best angles corresponding to the minimum expected value.
        num_fval(): Returns the number of function evaluations.
        num_shots(): Returns the total number of shots taken.
        get_best_solution(): Returns the best solutions and their corresponding cost.
    """

    def __init__(self, depth):
        """
        Initializes the OptResult object with the given depth.

        Args:
            depth (int): The depth p of the optimization.
        """
        self.depth = depth

        self.angles = []
        self.Exp = []
        self.Var = []
        self.WorstCost = []
        self.BestCost = []
        self.BestSols = []
        self.shots = []

        self.index_Exp_min = -1

    def add_iteration(self, angles, stat, shots):
        """
        Adds an iteration's results to the OptResult object.

        Args:
            angles (list): List of angles used in the iteration.
            stat (Statistic): Statistic object containing the results of the iteration.
            shots (int): Number of shots taken in the iteration.
        """
        self.angles.append(angles)
        self.Exp.append(-stat.get_CVaR())
        self.Var.append(stat.get_Variance())
        self.BestCost.append(-stat.get_max())
        self.WorstCost.append(-stat.get_min())
        self.BestSols.append(stat.get_max_sols())
        self.shots.append(shots)

    def compute_best_index(self):
        """
        Computes the index of the minimum expected value in `Exp`.
        This index is used to retrieve the best angles, expected value, and variance.
        """
        self.index_Exp_min = self.Exp.index(min(self.Exp))

    def get_best_Exp(self):
        """
        Returns:
            float: The best expected value from `Exp`.
        """
        return self.Exp[self.index_Exp_min]

    def get_best_Var(self):
        """
        Returns:
            float: The best variance from `Var`.
        """
        return self.Var[self.index_Exp_min]

    def get_best_angles(self):
        """
        Returns:
            list: The best angles from `angles` at the index of the minimum expected value.
        """
        return self.angles[self.index_Exp_min]

    def num_fval(self):
        """
        Returns:
            int: The number of function evaluations (the length of `Exp`).
        """
        return len(self.Exp)

    def num_shots(self):
        """
        Returns:
            int: The total number of shots taken across all iterations.
        """
        return sum(self.shots)

    def get_best_solution(self):
        """
        Iterates through the best solutions and returns the best cost and the corresponding solutions.

        Returns:
            tuple: A tuple containing
                - list: The best solutions (bit-strings) that yield the best cost.
                - float: The best cost found.
        """
        best_cost = np.min(self.BestCost)
        iterations_with_best_cost = np.where(self.BestCost == best_cost)[0]

        all_best_sols = []
        for i in iterations_with_best_cost:
            all_best_sols.append(self.BestSols[i])
        # flatten the list:
        all_best_sols = [item for sublist in all_best_sols for item in sublist]
        best_sols = np.unique(all_best_sols)
        return best_sols, best_cost


class QAOA:
    """
    Quantum Approximate Optimization Algorithm (QAOA) class.

    This class implements the QAOA algorithm for solving combinatorial optimization problems
    using quantum circuits.

    The class integrates a **problem**, **mixer**, and **initial state** to create a custom ansatz.
    The library already contains standard implementations.

    Attributes:
        problem (Problem): The optimization problem to solve.
        mixer (Mixer): The mixer circuit used in the QAOA algorithm.
        initialstate (InitialState): The initial state of the quantum circuit.

        backend (AerBackend): The backend to run the quantum circuits on. By default it uses quiskit's QuasmSimulator.
        noisemodel (optional): Optional Qiskit noise model.
        optimizer (list): [Optimizer, {options}]. Default is COBYLA with no options.
        precision (float, optional): Optional target precision for expectation value estimation.
        shots (int): Number of measurement shots per circuit evaluation.
        cvar (float): CVaR parameter for expectation value calculation.
        memorysize (int): Number of measurement outcomes to store (if > 0).
        interpolate (bool): Whether to use parameter interpolation for initial guesses.
        flip (bool): Whether to apply bit-flip boosting between layers.
        post (bool): Whether to apply post-processing to measurement results.
        number_trottersteps_mixer (int): Number of Trotter steps for the mixer.
        sequential (bool): Whether to run sequential circuit evaluations.

        parameterized_circuit (QuantumCircuit): The parameterized quantum circuit for QAOA.
        optimization_results (dict[int, OptResult]): Dictionary mapping depth (int) to corresponding optimization result objects (OptResult).

    Main methods:
        - createParameterizedCircuit(): Creates the parameterized circuit for a given depth.
        - sample_cost_landscape(): Samples the cost landscape for given angles at a specific depth.
        - optimize(): Runs the optimization process for a given depth.
        - get_Exp(): Returns the best expected value at a given depth.
        - get_Var(): Returns the best variance at a given depth.
        - get_beta(): Returns the beta angles at a given depth.
        - get_gamma(): Returns the gamma angles at a given depth.

    """

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
        memorysize=-1,
        interpolate=True,
        flip=False,
        post=False,
        number_trottersteps_mixer=1,
        sequential=False,
    ) -> None:
        """
        Initializes the QAOA class.

        Args:
            problem (Problem): The optimization problem to solve.
            mixer (Mixer): The mixer circuit used in the QAOA algorithm.
            initialstate (InitialState): The initial state of the quantum circuit.

            backend (AerBackend): The backend to run the quantum circuits on. Default is Quiskit's `qasm_simulator`.
            noisemodel (optional): Optional Qiskit noise model. Default is None.
            optimizer (list): [Optimizer, {options}]. Default is COBYLA with no options.
            precision (float, optional): Optional target precision for expectation value estimation. Default is None.
            shots (int): Number of measurement shots per circuit evaluation. Default is 1024.
            cvar (float): CVaR parameter for expectation value calculation. Default is 1.
            memorysize (int): Number of measurement outcomes to store (if > 0). Default is -1 (no memory).
            interpolate (bool): Whether to use parameter interpolation for initial guesses. Default is True.
            flip (bool): Whether to apply bit-flip boosting between layers. Default is False.
            post (bool): Whether to apply post-processing to measurement results. Default is False.
            number_trottersteps_mixer (int): Number of Trotter steps for the mixer. Default is 1.
            sequential (bool): Whether to run sequential circuit evaluations. Default is False.

        """
        # A QAO-Ansatz consist of these parts:

        # :problem of Basetype Problem,
        # implementing the phase circuit and the cost.

        # :mixer of Basetype Mixer,
        # specifying the mixer circuit.

        # :initialstate of Basetype InitialState,
        # specifying the initial state.

        # :backend: backend
        # :noisemodel: noisemodel
        # :optmizer: optimizer as a list containing [optimizer object, options dict]
        # :precision: precision to reach for expectation value based on error=variance/sqrt(shots)
        # :shots: if precision=None, the number of samples taken
        #               if precision!=None, the minimum number of samples taken
        # :cvar: used for CVar
        # """

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
        self.memorysize = memorysize
        self.memory = self.memorysize > 0
        self.interpolate = interpolate

        self.sequential = sequential

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
        self.memory_lists = []

        self.flip = flip
        self.flipper = BitFlip(self.problem.N_qubits)
        self.bitflips = []

        self.post = post
        self.Exp_post_processed = None
        self.Var_post_processed = None
        self.samplecount_hists = {}
        self.last_hist = {}
        self.number_trottersteps_mixer = number_trottersteps_mixer

    def exp_landscape(self):
        """
        Returns:
            float: The expected value of the cost landscape at depth p = 1.
        """
        ### at depth p = 1
        return self.Exp_sampled_p1

    def var_landscape(self):
        """
        Returns:
            float: The variance of the cost landscape at depth p = 1.
        """
        ### at depth p = 1
        return self.Var_sampled_p1

    def get_Exp(self, depth=None):
        """
        Args:
            depth (int, optional): The depth at which to retrieve the expected value.

        Returns:
            list/float: The best expected value(s) at the specified depth.
            If depth is None, returns a list of best expected values for all depths up to the current depth.
            If depth is specified, returns the best expected value at that depth.
        """
        if not depth:
            ret = []
            for i in range(1, self.current_depth + 1):
                ret.append(self.optimization_results[i].get_best_Exp())
            return ret
        if depth > self.current_depth + 1:
            raise ValueError
        return self.optimization_results[depth].get_best_Exp()

    def get_Var(self, depth):
        """
        Args:
            depth (int): The depth at which to retrieve the variance.

        Raises:
            ValueError: If the specified depth is greater than the current depth + 1.

        Returns:
            float: The best variance at the specified depth.
        """
        if depth > self.current_depth + 1:
            raise ValueError
        return self.optimization_results[depth].get_best_Var()

    def get_beta(self, depth):
        """
        Args:
            depth (int): The depth at which to retrieve the beta angles.

        Raises:
            ValueError: If the specified depth is greater than the current depth + 1.

        Returns:
            list: The best beta angles at the specified depth.
        """
        if depth > self.current_depth + 1:
            raise ValueError
        return self.optimization_results[depth].get_best_angles()[1::2]

    def get_gamma(self, depth):
        """
        Args:
            depth (int): The depth at which to retrieve the beta angles.

        Raises:
            ValueError: If the specified depth is greater than the current depth + 1.

        Returns:
            list: The best gamma angles at the specified depth.
        """
        if depth > self.current_depth + 1:
            raise ValueError
        return self.optimization_results[depth].get_best_angles()[::2]

    def get_angles(self, depth):
        """
        Args:
            depth (int): The depth at which to retrieve the angles.
        Raises:
            ValueError: If the specified depth is greater than the current depth + 1.

        Returns:
            list: The best angles at the specified depth.
        """
        if depth > self.current_depth + 1:
            raise ValueError
        return self.optimization_results[depth].get_best_angles()

    def get_memory_lists(self):
        """
        Returns:
            list: A list of memory lists containing measurement outcomes for each depth.
        """
        return self.memory_lists

    def createParameterizedCircuit(self, depth):
        """Creates the parameterized quantum circuit for the QAOA algorithm at a specified depth.

        The function generates quantum circuits for the problem and mixer Hamiltonians and the initial state.
        The main circuit is constructed layer by layer starting from the initial state.
        For each layer, a symbolic parameter is assigned is assigned to the problem unitary (gamma) and the mixer unitary (beta).

        If `flip` is set to True, a bit-flip circuit is added between layers to modify the state according to a pre-defined bit-flip pattern.
        If `usebarrier` is set to True, visual barrier instructions are added between layers to improve circuit readability and potentially
        guide compiler optimizations.

        Finally, the full parameterized circuit is measured and transpiled for the target backend. The constructed depth is stored for reference.

        Args:
            depth (int): The depth p (number of alternating mixing and phase separating layers) of the QAOA circuit.
        """
        if self.parameterized_circuit_depth == depth:
            LOG.info(
                "Circuit is already of depth " + str(self.parameterized_circuit_depth)
            )
            return

        self.problem.create_circuit()
        self.mixer.create_circuit()
        self.initialstate.create_circuit()

        a = AncillaRegister(self.problem.N_ancilla_qubits)
        q = QuantumRegister(self.problem.N_qubits)
        c = ClassicalRegister(self.problem.N_qubits)
        self.parameterized_circuit = QuantumCircuit(q, c, a)

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
                {
                    self.mixer.circuit.parameters[0]: self.beta_params[d]
                    / self.number_trottersteps_mixer
                },
                inplace=False,
            )
            for _ in range(0, self.number_trottersteps_mixer):
                self.parameterized_circuit.compose(tmp_circuit, inplace=True)

            if self.flip and (d != (depth - 1)):
                self.parameterized_circuit.barrier()
                self.flipper.create_circuit(self.bitflips[d])
                self.parameterized_circuit.compose(self.flipper.circuit, inplace=True)
                self.parameterized_circuit.barrier()

            if self.usebarrier:
                self.circuit.barrier()

        self.parameterized_circuit.barrier()
        self.parameterized_circuit.measure(q, c)
        self.parameterized_circuit = transpile(self.parameterized_circuit, self.backend)
        self.parametrized_circuit_depth = depth

    def sample_cost_landscape(
        self, angles={"gamma": [0, 2 * np.pi, 20], "beta": [0, 2 * np.pi, 20]}
    ):
        """
        Evaluates the cost function (lanscape) at depth p = 1 for a grid of angles.

        **Sequential mode**: If `sequential` is set to True, CVaR, variance and max/min cost are calculated sequentially for each combination of gamma and beta and stored in
        the lists  `Exp_sampled_p1`, `Var_sampled_p1`, `MaxCost_sampled_p1`, and `MinCost_sampled_p1`.

        **Batch mode**: If `sequential` is set to False, the angles are prepared and a single job is submitted to the backend with all parameter binds.
        After execution, `measurementStatistics` is called to retrieve the results.

        Args:
            angles (dict[str, list], optional): Dictionary mapping angle ranges to gamma and beta, where the range is defined by [start, stop, num]. Defaults to {"gamma": [0, 2 * np.pi, 20], "beta": [0, 2 * np.pi, 20]}.

        Raises:
            NotImplementedError: If the backend is not local or if the backend does not support the required operations.
        """
        self.landscape_p1_angles = angles
        logger = LOG.bind(func=self.sample_cost_landscape.__name__)
        logger.info("Calculating energy landscape for depth p=1...")

        depth = 1

        tmp = angles["gamma"]
        self.gamma_grid = np.linspace(tmp[0], tmp[1], tmp[2], endpoint=False)
        tmp = angles["beta"]
        self.beta_grid = np.linspace(tmp[0], tmp[1], tmp[2], endpoint=False)

        if self.backend.configuration().local:
            if self.sequential:
                expectations = []
                variances = []
                maxcosts = []
                mincosts = []

                self.createParameterizedCircuit(depth)
                logger.info("Executing sample_cost_landscape")
                for b in range(angles["beta"][2]):
                    for g in range(angles["gamma"][2]):
                        gamma = self.gamma_grid[g]
                        beta = self.beta_grid[b]
                        params = self.getParametersToBind(
                            [gamma, beta], self.parametrized_circuit_depth, asList=True
                        )

                        job = self.backend.run(
                            self.parameterized_circuit,
                            noise_model=self.noisemodel,
                            shots=self.shots,
                            parameter_binds=[params],
                            optimization_level=0,
                            memory=self.memory,
                        )

                        jres = job.result()
                        counts_list = jres.get_counts()

                        self.stat.reset()
                        for string in counts_list:
                            # qiskit binary strings use little endian encoding, but our cost function expects big endian encoding. Therefore, we reverse the order
                            cost = self.problem.cost(string[::-1])
                            self.stat.add_sample(
                                cost, counts_list[string], string[::-1]
                            )

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
                logger.info("Done measurement")
            else:
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
                job = self.backend.run(
                    self.parameterized_circuit,
                    noise_model=self.noisemodel,
                    shots=self.shots,
                    parameter_binds=[parameters],
                    optimization_level=0,
                    memory=self.memory,
                )
                logger.info("Done execute")
                self.measurementStatistics(job)
                logger.info("Done measurement")

        else:
            raise NotImplementedError

        logger.info("Calculating Energy landscape done")

    def measurementStatistics(self, job):
        # """
        # implements a function for expectation value and variance

        # :param job: job instance derived from BaseJob
        # :return: expectation and variance, if the job is a list
        # """
        """
        Processes the results of a job to extract measurement statistics -- CVar, variance and max/min costs.

        Extracts the job result and retrieves the counts of measurement outcomes (stored in `last_hist` for future reference), determining whether the result is a dictionary
        (single execution) or a list of dictionaries (batched executions with parameter sweeps).
        - If the result is a list of dictionaries, it computes the measurement statistics for each set of counts, which are stored in `Exp_sampled_p1`, `Var_sampled_p1`, `MaxCost_sampled_p1`, and `MinCost_sampled_p1`.
        - If the result is a single dictionary, it computes the measurement statistics for that single set of counts, storing the results in `stat`.

        The function also handles the memory of measurement outcomes and stores them in `memory_lists` if memory is enabled
        (`memorysize` > 0).

        Args:
            job (quiskit.providers.BaseJob): Executed Quiskit job instance containing the results of the quantum circuit execution.

        """
        jres = job.result()
        counts_list = jres.get_counts()

        if self.memorysize > 0:
            for i, _ in enumerate(jres.results):
                memory_list = jres.get_memory(experiment=i)
                if self.memorysize > 0:
                    for measurement in memory_list:
                        self.memory_lists.append(
                            [measurement, self.problem.cost(measurement[::-1])]
                        )
                        self.memorysize -= 1
                        if self.memorysize < 1:
                            break

        self.last_hist = counts_list

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
                    self.stat.add_sample(cost, counts[string], string[::-1])
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
                self.stat.add_sample(cost, counts_list[string], string[::-1])

    def optimize(
        self,
        depth,
        angles={"gamma": [0, 2 * np.pi, 20], "beta": [0, 2 * np.pi, 20]},
    ):
        """
        Runs the optimization process for the QAOA algorithm up to a specified depth.

        Drives the core iterative optimization loop of the QAOA algoritm up to a specified depth `p`, by incrementally building the circuit one layer at a time until the desired depth is reached.
        - If p=1, a **grid search** is performed over the angles to find a good starting point.
        - If p>1, the previously found angles are used as a starting point for local optimization at the next depth. If `interpolate` is set to True, the angles are interpolated to create a smoother transition between depths.

        At each depth, the optimization results are stored in `optimization_results`. Measurement statistics are collected and stored in `samplecount_hists`.
        If `flip` is set to True, bit-flip boosting is applied to the best solution found at each depth with masks stored in `bitflips`.
        If `post` is set to True, post-processing is applied to the measurement results at the final depth, and the processed expected value and variance are stored in `Exp_post_processed` and `Var_post_processed`.

        Args:
            depth (int): The maximum depth p to which the optimization should be run.
            angles (dict, optional): Dictionary mapping angle ranges to gamma and beta, where the range is defined by [start, stop, num]. Defaults to {"gamma": [0, 2 * np.pi, 20], "beta": [0, 2 * np.pi, 20]}.
        """
        ## run local optimization by iteratively increasing the depth until depth p is reached
        while self.current_depth < depth:
            if self.current_depth == 0:
                if self.Exp_sampled_p1 is None:
                    self.sample_cost_landscape(angles=angles)
                ind_Emin = np.unravel_index(
                    np.argmin(self.Exp_sampled_p1, axis=None), self.Exp_sampled_p1.shape
                )
                angles0 = np.array(
                    (self.gamma_grid[ind_Emin[1]], self.beta_grid[ind_Emin[0]])
                )
            else:
                gamma = self.get_gamma(self.current_depth)
                beta = self.get_beta(self.current_depth)

                if self.interpolate:
                    gamma_interp = self.interp(gamma)
                    beta_interp = self.interp(beta)
                else:
                    gamma_interp = np.append(gamma, 0)
                    beta_interp = np.append(beta, 0)
                angles0 = np.zeros(2 * (self.current_depth + 1))
                angles0[::2] = gamma_interp
                angles0[1::2] = beta_interp

            self.optimization_results[self.current_depth + 1] = OptResult(
                self.current_depth + 1
            )
            # Create parameterized circuit at new depth
            self.createParameterizedCircuit(int(len(angles0) / 2))

            res = self.local_opt(angles0)

            self.samplecount_hists[self.current_depth + 1] = self.last_hist

            self.optimization_results[self.current_depth + 1].compute_best_index()

            LOG.info(
                f"cost(depth { self.current_depth + 1} = {res.fun}",
                func=self.optimize.__name__,
            )

            if self.flip:
                hist = self.samplecount_hists[self.current_depth + 1]
                string = max(hist, key=hist.get)
                boosted = self.flipper.boost_samples(
                    problem=self.problem,
                    string=string,
                )
                string_xor = self.flipper.xor(string, boosted)
                self.bitflips.append(string_xor)

            self.current_depth += 1

        if self.post:
            samples = self.samplecount_hists[self.current_depth]
            post_processing(self, samples=samples, K=self.post)
            self.Exp_post_processed = -self.stat.get_CVaR()
            self.Var_post_processed = self.stat.get_Variance()

    def local_opt(self, angles0):
        """
        Performs classical local optimization of the QAOA circuit at a given depth by minimizing the loss function using the specified optimizer in `optimizer`.

        Args:
            angles0 (list): Initial guess for the angles to be optimized.

        Returns:
            res (OptimizationResult): The result of the optimization process, containing the optimized angles and the minimum loss value.

        Raises:
            TypeError: If the optimizer requires fidelity and it is not provided.
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
        Loss function.

        Args:
            angles (list): List of angles (gamma and beta).

        Returns:
            float: The negative expected value of the cost function (CVaR) for the given angles.
            This is used as the objective function to be minimized during optimization.

        Raises:
            NotImplementedError: If the backend is not local or not implemented.
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
                job = self.backend.run(
                    self.parameterized_circuit,
                    noise_model=self.noisemodel,
                    shots=shots,
                    parameter_binds=[params],
                    optimization_level=0,
                    memory=self.memory,
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

        Args:
            angles (array-like): 1D array of alternating gamma and beta values (length = 2 * depth)
            depth (int): Circuit depth.
            asList (bool): Boolean that specify if the values in the dict should be a list or not (required for batching).

        Returns:
            dict: Dictionary mapping each QAOA parameter (gamma/beta) to its corresponding value.
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
        when going from depth p to p+1 (https://doi.org/10.1103/PhysRevX.10.021067).

        Args:
            angles (array-like): 1D array of gamma or beta parameters at depth p (length = p).

        Returns:
            np.ndarray: Interpolated parameters for depth p+1 (length = p+1).
        """
        depth = len(angles)
        tmp = np.zeros(len(angles) + 2)
        tmp[1:-1] = angles.copy()
        w = np.arange(0, depth + 1)
        return w / depth * tmp[:-1] + (depth - w) / depth * tmp[1:]

    def hist(self, angles, shots):
        """
        Executes the QAOA circuit for a given set of angles and returns the corrected histogram of bitstring outcomes.

        Args:
            angles (array-like): Flat list or array of gamma and beta values (length = 2 * depth).
            shots (int): Number of circuit executions (samples) to collect.

        Returns:
            dict: Dictionary of measurement outcomes (bit-strings, little-endian format) and their counts for the given angles at the current depth.

        Raises:
            NotImplementedError: If the backend is not local or not implemented.
        """
        depth = int(len(angles) / 2)
        self.createParameterizedCircuit(depth)

        params = self.getParametersToBind(angles, depth, asList=True)
        if self.backend.configuration().local:
            job = self.backend.run(
                self.parameterized_circuit,
                shots=shots,
                parameter_binds=[params],
                optimization_level=0,
                memory=self.memory,
            )
        else:
            raise NotImplementedError

        # Qiskit uses big endian encoding, cost function uses litle endian encoding.
        # Therefore the string is reversed before passing it to the cost function.
        hist = job.result().get_counts()
        corrected_hist = {}
        for string in hist:
            corrected_hist[string[::-1]] = hist[string]
        return corrected_hist

    def random_init(self, gamma_bounds, beta_bounds, depth):
        """
        Enforces the bounds of gamma and beta based on the graph type.

        Args:
            gamma_bounds (tuple): Parameter bound tuple (min,max) for gamma.
            beta_bounds (tuple): Parameter bound tuple (min,max) for beta.
            depth (int): The depth of the QAOA circuit.

        Returns:
            np.ndarray: On the form (gamma_1, beta_1, gamma_2, ...., gamma_d, beta_d).
        """
        gamma_list = np.random.uniform(gamma_bounds[0], gamma_bounds[1], size=depth)
        beta_list = np.random.uniform(beta_bounds[0], beta_bounds[1], size=depth)
        initial = np.empty((gamma_list.size + beta_list.size,), dtype=gamma_list.dtype)
        initial[0::2] = gamma_list
        initial[1::2] = beta_list
        return initial

    def get_optimal_solutions(self):
        """
        Iterates through all optimization result objects looking for the bit-string(s) that gave optimal cost value.

        Returns:
            list: List with the obtained best bit-strings.
        """
        best_sols = []
        best_costs = []
        for i in self.optimization_results:
            best_sols_i, best_cost_i = self.optimization_results[i].get_best_solution()
            best_sols.append(best_sols_i)
            best_costs.append(best_cost_i)

        best_iterations = np.where(best_costs == np.min(best_costs))[0]
        opt_sols = []
        for i in best_iterations:
            opt_sols.append(best_sols[i])
        opt_sols = [item for sublist in opt_sols for item in sublist]
        return np.unique(opt_sols)
