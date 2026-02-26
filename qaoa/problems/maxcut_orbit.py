"""
MaxCut problem with orbit-based parameter reduction.

Uses graph automorphisms to reduce the number of parameters
by identifying edges that are equivalent under symmetry.
"""

import logging
import networkx as nx
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

from .base_problem import Problem
from qaoa.util.graph_automorphism import compute_edge_orbits, get_edge_to_orbit_map

logger = logging.getLogger(__name__)


class MaxCutOrbit(Problem):
    """
    MaxCut problem with orbit symmetry reduction.

    Uses graph automorphisms to identify edge orbits and assign
    one parameter per orbit instead of one per edge.

    Attributes:
        G: NetworkX graph
        N_qubits: Number of qubits (= number of nodes)
        orbits: Dictionary of orbit_id -> list of edges
        edge_to_orbit: Dictionary of edge -> orbit_id
        num_orbits: Number of distinct orbits
    """

    def __init__(self, G: nx.Graph):
        """
        Args:
            G: NetworkX graph (unweighted)
        """
        super().__init__()
        self.G = G
        self.N_qubits = G.number_of_nodes()

        # Compute orbit structure
        self.orbits = compute_edge_orbits(G)
        self.edge_to_orbit = get_edge_to_orbit_map(G)
        self.num_orbits = len(self.orbits)

        logger.info(
            "MaxCut Orbit: %d nodes, %d edges, %d orbits",
            G.number_of_nodes(),
            G.number_of_edges(),
            self.num_orbits,
        )

    def get_num_parameters(self):
        """Number of parameters = number of orbits"""
        return self.num_orbits

    def cost(self, string: str) -> float:
        """
        Compute MaxCut cost: number of edges between different colors.

        Args:
            string: Binary string (big-endian)

        Returns:
            Negative cut value (for minimization)
        """
        cut = 0
        for i, j in self.G.edges():
            if string[i] != string[j]:
                cut += 1
        return -cut

    def create_circuit(self):
        """
        Create parameterized circuit with one parameter per orbit.

        Circuit structure:
        - For each edge (i, j):
            - Apply RZZ(2 * gamma[orbit_id] * weight, i, j)
        - All edges in the same orbit share the same gamma parameter
        """
        q = QuantumRegister(self.N_qubits)
        self.circuit = QuantumCircuit(q)

        # Create one parameter per orbit
        orbit_params = [Parameter(f"gamma_orbit_{i}") for i in range(self.num_orbits)]

        # Apply RZZ for each edge using its orbit's parameter
        for i, j in self.G.edges():
            orbit_id = self.edge_to_orbit[(i, j)]
            weight = self.G[i][j].get("weight", 1.0)

            # RZZ gate for edge (i,j)
            gamma_ij = orbit_params[orbit_id] * weight
            self.circuit.rzz(2 * gamma_ij, q[i], q[j])
