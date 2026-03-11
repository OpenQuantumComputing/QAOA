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
        label (str): A human-readable annotation for this component, used as the
            circuit name when ``create_circuit`` is called.  Defaults to the
            class name.
    """

    def __init__(self, label: str | None = None) -> None:
        """
        Initializes a MixerBase object.

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
        Abstract method to create the quantum circuit representing the mixing operation.

        Subclasses must implement this method to define the quantum circuit
        that represents the specific mixing operation.

        Returns:
            QuantumCircuit: The quantum circuit representing the mixing operation.
        """
        pass

    def get_num_parameters(self):
        """
        Returns the number of parameters this mixer uses per layer.

        Returns:
            int: Number of parameters per layer (default: 1).
        """
        return 1
