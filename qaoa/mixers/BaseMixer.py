import numpy as np

from abc import ABC, abstractmethod
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter


class MixerBase(ABC):
    def __init__(self, parent) -> None:
        super().__init__()
        self.parent = parent
        self.mixer_circuit = None

    @property
    def params(self):
        return self.parent.params

    def __getattr__(self, attr):
        """IMPORTANT
        One could change this to
            getattr(self.parent, attr)
        Which basically would result in Problem and Mixer being more intertwined.
        Using getattr(*) would be more elegant, but makes it easier to make mistakes
        (spelling errors would for example be dangerous)
        """
        return self.parent.params[attr]


class Mixer(MixerBase):
    @abstractmethod
    def create_mixer(self):
        pass

    def set_initial_state(self, circuit, qubit_register):
        """Initial state will by default be decided here, but can be overwritten with the
        :parm init_circ

        """
        raise NotImplementedError


class Constrained(Mixer):
    def __init__(self, parent) -> None:
        super().__init__(parent)

        self.B = []
        self.best_mixer_terms = []
        self.mixer_circuit = None
        self.reduced = self.params.get("reduced", True)

    def set_initial_state(self, circuit, qubit_register):
        # set to ground state of mixer hamilton??
        if not self.B:
            self.compute_feasible_subspace()
            # initial state
        ampl_vec = np.zeros(2 ** len(self.B[0]))
        ampl = 1 / np.sqrt(len(self.B))
        for state in self.B:
            ampl_vec[int(state, 2)] = ampl

        circuit.initialize(ampl_vec, qubit_register)

    def create_mixer(self):
        if not self.B:
            self.compute_feasible_subspace()

        m = Mixer(self.B, sort=True)
        m.compute_commuting_pairs()
        m.compute_family_of_graphs()
        m.get_best_mixer_commuting_graphs(reduced=self.reduced)
        (
            self.mixer_circuit,
            self.best_mixer_terms,
            self.logical_X_operators,
        ) = m.compute_parametrized_circuit(self.reduced)

        usebarrier = self.params.get("usebarrier", False)
        if usebarrier:
            self.mixer_circuit.barrier()

    @abstractmethod
    def compute_feasible_subspace(self):
        pass

    def set_feasible_subspace(self, space):
        self.B = space

    @abstractmethod
    def isFeasible(self, string):
        pass

