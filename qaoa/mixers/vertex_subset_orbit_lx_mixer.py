"""Orbit-tied logical-X mixer for MIS and MVC."""

from qiskit.circuit import Parameter

from qaoa.utils.graph_orbits import compute_node_orbits

from .vertex_subset_lx_mixer import VertexSubsetLXMixer


class VertexSubsetOrbitLXMixer(VertexSubsetLXMixer):
    r"""Degree-ordered LX mixer with one :math:`\beta` per vertex orbit."""

    def __init__(self, graph, problem_kind, label=None):
        super().__init__(graph, problem_kind=problem_kind, multi_angle=False, label=label)
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

        width = max(1, len(str(len(self.node_orbits) - 1)))
        self.mixer_params = [
            Parameter(f"x_beta_orbit_{orbit:0{width}d}")
            for orbit in range(len(self.node_orbits))
        ]

    def get_num_parameters(self):
        return len(self.node_orbits)

    def _beta_for_target(self, rank, target):
        del rank
        return self.mixer_params[self.node_to_orbit[target]]
