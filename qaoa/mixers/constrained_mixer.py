from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

from abc import abstractmethod

from .base_mixer import Mixer


class Constrained(Mixer):
    def __init__(self) -> None:
        self.B = []
        self.best_mixer_terms = []
        self.circuit = None
        # self.reduced = self.params.get("reduced", True)

    # def set_initial_state(self, circuit, qubit_register):
    #    # set to ground state of mixer hamiltonian??
    #    if not self.B:
    #        self.compute_feasible_subspace()
    #        # initial state
    #    ampl_vec = np.zeros(2 ** len(self.B[0]))
    #    ampl = 1 / np.sqrt(len(self.B))
    #    for state in self.B:
    #        ampl_vec[int(state, 2)] = ampl

    #    circuit.initialize(ampl_vec, qubit_register)

    def create_circuit(self):
        if not self.B:
            self.compute_feasible_subspace()

        m = Mixer(self.B, sort=True)
        m.compute_commuting_pairs()
        m.compute_family_of_graphs()
        m.get_best_mixer_commuting_graphs(reduced=self.reduced)
        (
            self.circuit,
            self.best_mixer_terms,
            self.logical_X_operators,
        ) = m.compute_parametrized_circuit(self.reduced)

    @abstractmethod
    def compute_feasible_subspace(self):
        pass

    def set_feasible_subspace(self, space):
        self.B = space

    @abstractmethod
    def isFeasible(self, string):
        pass
