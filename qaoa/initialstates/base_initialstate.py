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
        label (str): A human-readable annotation for this component, used as the
            circuit name when ``create_circuit`` is called.  Defaults to the
            class name.
    """

    def __init__(self, label: str | None = None) -> None:
        """
        Initializes a BaseInitialState object.

        The ``circuit`` attribute is set to None initially, and ``N_qubits``
        is not defined until ``setNumQubits`` is called.

        Args:
            label (str | None): Optional annotation label for this component.
                Defaults to the class name when *None*.
        """
        self.circuit = None
        self.label = label if label is not None else self.__class__.__name__

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

    def __init_subclass__(cls, **kwargs):
        """
        Automatically equips every concrete subclass with two behaviours:

        1. **Default label** – if the subclass ``__init__`` does *not* call
           ``super().__init__()``, the instance will still get a ``label``
           attribute (set to the class name) after ``__init__`` returns.
        2. **Circuit annotation** – after ``create_circuit`` returns the
           ``circuit.name`` attribute is set to ``self.label`` so that
           circuit drawings show the component name.
        """
        super().__init_subclass__(**kwargs)

        # Wrap __init__ to guarantee self.label exists even when super() is
        # not called by the subclass.
        if "__init__" in cls.__dict__:
            _orig_init = cls.__dict__["__init__"]

            def _make_init_wrapper(f):
                def _wrapped_init(self, *args, **kwargs):
                    f(self, *args, **kwargs)
                    if not hasattr(self, "label"):
                        self.label = self.__class__.__name__

                _wrapped_init.__doc__ = f.__doc__
                _wrapped_init.__name__ = f.__name__
                return _wrapped_init

            setattr(cls, "__init__", _make_init_wrapper(_orig_init))

        # Wrap create_circuit to set circuit.name from self.label.
        if "create_circuit" in cls.__dict__:
            _orig_cc = cls.__dict__["create_circuit"]
            if not getattr(_orig_cc, "__isabstractmethod__", False):

                def _make_cc_wrapper(f):
                    def _wrapped_cc(self, *args, **kwargs):
                        f(self, *args, **kwargs)
                        if self.circuit is not None:
                            self.circuit.name = getattr(
                                self, "label", self.__class__.__name__
                            )

                    _wrapped_cc.__doc__ = f.__doc__
                    _wrapped_cc.__name__ = f.__name__
                    return _wrapped_cc

                setattr(cls, "create_circuit", _make_cc_wrapper(_orig_cc))

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

    def get_num_parameters(self):
        """
        Returns the number of parameters this initial state uses.

        Returns:
            int: Total number of parameters (default: 0, meaning no parameterization).
        """
        return 0
