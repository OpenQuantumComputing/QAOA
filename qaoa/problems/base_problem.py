from abc import ABC, abstractmethod


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


class Problem(BaseProblem):
    """
    Abstract subclass for defining specific optimization problems.

    This abstract subclass of `BaseProblem` is meant for defining concrete
    optimization problems. Subclasses of `Problem` must implement the `cost`
    and `create_circuit` methods to define the problem's cost function and
    create the associated quantum circuit.

    Attributes:
        circuit (QuantumCircuit): The quantum circuit associated with the problem.

    Methods:
        cost(string): Abstract method to calculate the cost of a solution.
        create_circuit(): Abstract method to create the quantum circuit
            representing the problem.
        isFeasible(string): Checks if a given solution string is feasible.
            This method returns True by default and can be overridden by
            subclasses to implement custom feasibility checks.

    Note:
        Subclasses of `Problem` must provide implementations for the `cost`
        and `create_circuit` methods.

    Example:
        ```python
        class MyProblem(Problem):
            def cost(self, string):
                # Define the cost calculation for the optimization problem.
                ...

            def create_circuit(self):
                # Define the quantum circuit for the optimization problem.
                ...
        ```
    """

    @abstractmethod
    def cost(self, string):
        """
        Abstract method to calculate the cost of a solution.

        Subclasses must implement this method to define how the cost of a
        solution is calculated for the specific optimization problem.

        Args:
            string (str): A solution string or configuration to evaluate.

        Returns:
            float: The cost of the given solution.
        """
        pass

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

    def computeMinMaxCosts(self):
        """
        Brute force method to compute min and max cost of feasible solution
        """
        import itertools

        max_cost = float("-inf")
        min_cost = float("inf")
        for s in ["".join(i) for i in itertools.product("01", repeat=self.N_qubits)]:
            if self.isFeasible(s):
                cost = -self.cost(s)
                max_cost = max(max_cost, cost)
                min_cost = min(min_cost, cost)
        return min_cost, max_cost
