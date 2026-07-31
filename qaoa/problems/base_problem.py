from abc import ABC, abstractmethod
from enum import Enum
import itertools

from qaoa.utils import validation


class ObjectiveSense(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class BaseProblem(ABC):
    """
    Base class for defining optimization problems.

    This is an abstract base class (ABC) that provides a common interface for
    optimization problems. Subclasses can inherit from this class to define
    specific optimization problems.

    Attributes:
        circuit (QuantumCircuit): The quantum circuit associated with the problem.
    """

    def __init__(self) -> None:
        """
        Initializes a BaseProblem object.

        The `circuit` attribute is set to None initially and can be
        assigned a quantum circuit later.
        """
        self.circuit = None
        self.N_ancilla_qubits = 0


class Problem(BaseProblem):
    """
    Abstract subclass for defining specific optimization problems.

    This abstract subclass of `BaseProblem` is meant for defining concrete
    optimization problems. Subclasses of `Problem` must implement the
    `objective_value` and `create_circuit` methods to define the natural
    objective function and the associated quantum circuit.

    Attributes:
        circuit (QuantumCircuit): The quantum circuit associated with the problem.

    Methods:
        objective_value(string): Calculate the natural objective value of a
            solution.
        create_circuit(): Abstract method to create the quantum circuit
            representing the problem.
        isFeasible(string): Checks if a given solution string is feasible.
            This method returns True by default and can be overridden by
            subclasses to implement custom feasibility checks.
        validate_circuit(): Checks if the implemented quantum circuit 
            corresponds to the given cost function. 

    Note:
        Subclasses of `Problem` must provide implementations for the
        `objective_value` and `create_circuit` methods.

    Example:
        ```python
        class MyProblem(Problem):
            def objective_value(self, string):
                # Define the objective calculation for the optimization problem.
                ...

            def create_circuit(self):
                # Define the quantum circuit for the optimization problem.
                ...
        ```
    """

    def __init__(self, objective_sense: ObjectiveSense = ObjectiveSense.MINIMIZE) -> None:
        super().__init__()
        if not isinstance(objective_sense, ObjectiveSense):
            try:
                objective_sense = ObjectiveSense(objective_sense)
            except Exception as exc:
                raise ValueError(
                    "objective_sense must be one of "
                    f"{[s.value for s in ObjectiveSense]}"
                ) from exc
        self.objective_sense = objective_sense

    def objective_value(self, string):
        raise NotImplementedError("Subclasses must implement objective_value().")

    def energy(self, string):
        value = self.objective_value(string)
        if self.objective_sense is ObjectiveSense.MINIMIZE:
            return value
        return -value

    def objective_from_energy(self, energy):
        if self.objective_sense is ObjectiveSense.MINIMIZE:
            return energy
        return -energy

    @abstractmethod
    def create_circuit(self):
        """
        Abstract method to create the quantum circuit representing the problem.

        Subclasses must implement this method to define the quantum circuit
        that represents the optimization problem.

        Returns:
            QuantumCircuit: The quantum circuit representing the problem.
        """
        pass

    def isFeasible(self, string):
        """
        Check if a solution string is feasible.

        This method provides a default implementation that always returns True.
        Subclasses can override this method to implement custom feasibility checks.

        Args:
            string (str): A solution string or configuration to check.

        Returns:
            bool: True if the solution is feasible; otherwise, False.
        """
        return True

    def get_num_parameters(self):
        """
        Returns the number of parameters this problem uses per layer.

        Returns:
            int: Number of parameters per layer (default: 1).
        """
        return 1

    def objective_bounds(self):
        min_objective = float("inf")
        max_objective = float("-inf")
        for s in map("".join, itertools.product("01", repeat=self.N_qubits)):
            if self.isFeasible(s):
                value = self.objective_value(s)
                min_objective = min(min_objective, value)
                max_objective = max(max_objective, value)
        return min_objective, max_objective

    def optimal_objective(self):
        min_objective, max_objective = self.objective_bounds()
        if self.objective_sense is ObjectiveSense.MINIMIZE:
            return min_objective
        return max_objective

    def energy_bounds(self):
        min_energy = float("inf")
        max_energy = float("-inf")
        for s in map("".join, itertools.product("01", repeat=self.N_qubits)):
            if self.isFeasible(s):
                value = self.energy(s)
                min_energy = min(min_energy, value)
                max_energy = max(max_energy, value)
        return min_energy, max_energy

    def validate_circuit(self, t=1, flip=True, atol=1e-8, rtol=1e-8):
        """
        Exact check that the problem's circuit represents the problem's cost function.
        This tests checks that the unitary operator represented by the quantum circuit is
        equal to the expected matrix with diagonal elements
        exp(-j*t*energy(e)),
        where e is the corresponding binary state, up to a global phase.
        
        Suitable for <= 10 qubits as this check uses the full unitary matrix of size 2^n x 2^n).
        Returns: (ok: bool, report: dict)
        """
        if self.circuit is None:
            self.create_circuit()
        return validation.check_phase_separator_exact_problem(self, t=t, flip=flip, atol=atol, rtol=rtol)
