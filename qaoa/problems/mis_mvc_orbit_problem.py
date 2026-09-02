"""Orbit-shared phase separators for unweighted MIS and MVC."""

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

from qaoa.utils.graph_orbits import compute_node_orbits

from .base_problem import ObjectiveSense
from .mis_mvc_problem import MIS_MVC_Problem


class MIS_MVC_Orbit(MIS_MVC_Problem):
    """Unweighted MIS/MVC problem with one gamma per vertex orbit."""

    def __init__(self, graph, problem_kind, objective_sense=None):
        node_count = graph.number_of_nodes()
        super().__init__(
            graph,
            problem_kind=problem_kind,
            weights=np.ones(node_count, dtype=float),
            objective_sense=objective_sense,
        )
        self.orbit_qubits = tuple(range(self.N_qubits))
        self.node_orbits, self.node_to_orbit = compute_node_orbits(
            self.G, nodes=self.orbit_qubits
        )
        self.orbits = tuple(
            tuple(self.node_order[node] for node in orbit) for orbit in self.node_orbits
        )
        self.parameter_nodes = tuple(
            self.node_order[orbit[0]] for orbit in self.node_orbits
        )

    def get_num_parameters(self):
        return len(self.node_orbits)

    def create_circuit(self):
        width = max(1, len(str(len(self.node_orbits) - 1)))
        gamma_params = [
            Parameter(f"gamma_orbit_{orbit:0{width}d}")
            for orbit in range(len(self.node_orbits))
        ]

        q = QuantumRegister(self.N_qubits)
        self.circuit = QuantumCircuit(q)

        direction = -1.0 if self.objective_sense is ObjectiveSense.MINIMIZE else 1.0
        for qubit in self.orbit_qubits:
            gamma = gamma_params[self.node_to_orbit[qubit]]
            self.circuit.rz(direction * gamma, q[qubit])
