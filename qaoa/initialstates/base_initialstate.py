from abc import ABC, abstractmethod


class BaseInitialState(ABC):
    """
    Base class for defining initial quantum states.

    This is an abstract base class (ABC) that defines the basic structure of
    initial quantum states. Subclasses must implement the `create_circuit`
    method to create a quantum circuit for the specific initial state.

    Attributes:
        circuit (QuantumCircuit): The quantum circuit representing the initial state.
        N_qubits (int): The number of qubits in the quantum circuit.
    """

    def __init__(self) -> None:
        """
        Initializes a BaseInitialState object.

        The `circuit` attribute is set to None initially, and `N_qubits`
        is not defined until `setNumQubits` is called.
        """
        self.circuit = None

    def setNumQubits(self, n):
        """
        Set the number of qubits for the quantum circuit.

        Args:
            n (int): The number of qubits to set.
        """
        self.N_qubits = n


class InitialState(BaseInitialState):
    """
    Abstract subclass for defining specific initial quantum states.

    This abstract subclass of `BaseInitialState` is meant for defining
    concrete initial quantum states. Subclasses of `InitialState` must
    implement the `create_circuit` method to create a quantum circuit
    representing the specific initial state.

    Note:
        Subclasses of `InitialState` must provide an implementation
        for the `create_circuit` method.

    Example:
        ```python
        class MyInitialState(InitialState):
            def create_circuit(self):
                # Define the quantum circuit for a custom initial state.
                ...
        ```
    """

    @abstractmethod
    def create_circuit(self):
        """
        Abstract method to create the quantum circuit for the initial state.

        Subclasses must implement this method to define the quantum circuit
        for the specific initial state they represent.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        pass
