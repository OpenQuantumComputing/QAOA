from abc import ABC, abstractmethod


class MixerBase(ABC):
    """
    Base class for defining quantum mixing operations.

    This is an abstract base class (ABC) that provides a common interface for
    quantum mixing operations. Subclasses can inherit from this class to define
    specific mixing operations.

    Attributes:
        circuit (QuantumCircuit): The quantum circuit associated with the mixer.
        N_qubits (int): The number of qubits in the mixer circuit.
    """

    def __init__(self) -> None:
        """
        Initializes a MixerBase object.

        The `circuit` attribute is set to None initially, and `N_qubits`
        is not defined until `setNumQubits` is called.
        """
        self.circuit = None

    def setNumQubits(self, n):
        """
        Set the number of qubits for the quantum mixer circuit.

        Args:
            n (int): The number of qubits to set.
        """
        self.N_qubits = n


class Mixer(MixerBase):
    """
    Abstract subclass for defining specific quantum mixing operations.

    This abstract subclass of `MixerBase` is meant for defining concrete quantum
    mixing operations. Subclasses of `Mixer` must implement the `create_circuit`
    method to create the associated quantum circuit for mixing.

    Attributes:
        circuit (QuantumCircuit): The quantum circuit associated with the mixer.

    Methods:
        create_circuit(): Abstract method to create the quantum circuit
            representing the mixing operation.

    Note:
        Subclasses of `Mixer` must provide an implementation for the `create_circuit`
        method.

    Example:
        ```python
        class MyMixer(Mixer):
            def create_circuit(self):
                # Define the quantum circuit for the custom mixing operation.
                ...
        ```
    """

    @abstractmethod
    def create_circuit(self):
        """
        Abstract method to create the quantum circuit representing the mixing operation.

        Subclasses must implement this method to define the quantum circuit
        that represents the specific mixing operation.

        Returns:
            QuantumCircuit: The quantum circuit representing the mixing operation.
        """
        pass
