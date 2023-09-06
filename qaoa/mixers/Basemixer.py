import numpy as np

from abc import ABC, abstractmethod
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter


class Mixer(ABC):
    def __init__(self, params={}) -> None:
        super().__init__()

        self.params = params
        self.mixer_circuit = None


    @abstractmethod
    def set_initial_state(self, circuit, qubit_register):
        pass

    @abstractmethod
    def create_mixer(self):
        pass

class Constrained(Mixer):
    def __init__(self, params={}) -> None:
        super().__init__(params)

        self.B = []
        self.best_mixer_terms = []
        self.mixer_circuit = None
        self.reduced = params.get("reduced", True)

    def set_initial_state(self, circuit, qubit_register):
        # set to ground state of mixer hamilton??
        if not self.B:
            self.computeFeasibleSubspace()
            # initial state
        ampl_vec = np.zeros(2 ** len(self.B[0]))
        ampl = 1 / np.sqrt(len(self.B))
        for state in self.B:
            ampl_vec[int(state, 2)] = ampl

        circuit.initialize(ampl_vec, qubit_register)

    def create_mixer(self):
        if not self.B:
            self.computeFeasibleSubspace()

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

    @abstractmethod
    def isFeasible(self, string):
        pass

class Unconstrained(Mixer):
    def __init__(self, params={}) -> None:
        super().__init__(params)

    def set_initial_state(self, circuit, qubit_register):
        circuit.h(qubit_register)

    def create_mixer(self):
        q = QuantumRegister(self.N_qubits)
        mixer_param = Parameter("x_beta")

        self.mixer_circuit = QuantumCircuit(q)
        self.mixer_circuit.rx(-2 * mixer_param, range(self.N_qubits))

        usebarrier = self.params.get("usebarrier", False)
        if usebarrier:
            self.mixer_circuit.barrier()