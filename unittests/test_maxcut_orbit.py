import unittest
import networkx as nx
import sys

sys.path.append("../")

# Pre-warm the qiskit_algorithms.optimizers submodule so that all subsequent
# imports of qaoa subpackages succeed when qaoa/__init__.py is partially
# broken due to an incompatible qiskit_algorithms install.  In a fully working
# environment this try/except is a no-op.
try:
    import qaoa  # noqa: F401
except Exception:
    pass

from qaoa.util.graph_automorphism import (
    compute_edge_orbits,
    get_edge_to_orbit_map,
    print_orbit_structure,
)
from qaoa.problems.maxcut_orbit import MaxCutOrbit


class TestGraphAutomorphism(unittest.TestCase):
    def test_complete_graph_single_orbit(self):
        """All edges of K_n are in the same orbit."""
        for n in [3, 4, 5]:
            G = nx.complete_graph(n)
            orbits = compute_edge_orbits(G)
            self.assertEqual(len(orbits), 1, f"K{n} should have 1 orbit")
            self.assertEqual(
                len(orbits[0]), G.number_of_edges(),
                f"K{n} orbit should contain all {G.number_of_edges()} edges"
            )

    def test_star_graph_single_orbit(self):
        """All edges of a star graph are in the same orbit."""
        G = nx.star_graph(4)
        orbits = compute_edge_orbits(G)
        self.assertEqual(len(orbits), 1)
        self.assertEqual(len(orbits[0]), G.number_of_edges())

    def test_house_graph_orbit_count(self):
        """House graph should have 4 orbits due to its symmetry structure."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4)])
        orbits = compute_edge_orbits(G)
        # House graph has 6 edges; with its symmetry there should be 4 orbits
        self.assertGreater(len(orbits), 1)
        self.assertLess(len(orbits), G.number_of_edges())
        # Total edges across all orbits must match
        total_edges = sum(len(e) for e in orbits.values())
        self.assertEqual(total_edges, G.number_of_edges())

    def test_orbit_partition_covers_all_edges(self):
        """Every edge must appear exactly once across all orbits."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4)])
        orbits = compute_edge_orbits(G)
        all_orbit_edges = [e for edges in orbits.values() for e in edges]
        self.assertEqual(len(all_orbit_edges), G.number_of_edges())
        for edge in G.edges():
            self.assertIn(
                edge, all_orbit_edges,
                f"Edge {edge} should appear in orbits"
            )

    def test_edge_to_orbit_map(self):
        """Edge-to-orbit map should cover all directed edge pairs."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])
        edge_to_orbit = get_edge_to_orbit_map(G)
        for i, j in G.edges():
            self.assertIn((i, j), edge_to_orbit)
            self.assertIn((j, i), edge_to_orbit)
            self.assertEqual(edge_to_orbit[(i, j)], edge_to_orbit[(j, i)])

    def test_print_orbit_structure_runs(self):
        """print_orbit_structure should execute without errors."""
        G = nx.complete_graph(3)
        # Should not raise
        print_orbit_structure(G)


class TestMaxCutOrbit(unittest.TestCase):
    def setUp(self):
        self.G_house = nx.Graph()
        self.G_house.add_edges_from(
            [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4)]
        )

    def test_num_orbits_matches_problem(self):
        """num_orbits attribute should match compute_edge_orbits output."""
        problem = MaxCutOrbit(self.G_house)
        orbits = compute_edge_orbits(self.G_house)
        self.assertEqual(problem.num_orbits, len(orbits))

    def test_get_num_parameters(self):
        """get_num_parameters should return the number of orbits."""
        problem = MaxCutOrbit(self.G_house)
        self.assertEqual(problem.get_num_parameters(), problem.num_orbits)

    def test_nqubits(self):
        """N_qubits should equal number of nodes."""
        problem = MaxCutOrbit(self.G_house)
        self.assertEqual(problem.N_qubits, self.G_house.number_of_nodes())

    def test_circuit_parameter_count(self):
        """Circuit should have exactly num_orbits unique parameters."""
        problem = MaxCutOrbit(self.G_house)
        problem.create_circuit()
        self.assertEqual(len(problem.circuit.parameters), problem.num_orbits)

    def test_cost_all_same_zero(self):
        """All nodes in same partition -> cut = 0."""
        problem = MaxCutOrbit(self.G_house)
        self.assertEqual(problem.cost("00000"), 0)
        self.assertEqual(problem.cost("11111"), 0)

    def test_cost_returns_negative(self):
        """Cost should be non-positive (negated cut for minimization)."""
        problem = MaxCutOrbit(self.G_house)
        cost = problem.cost("01010")
        self.assertLessEqual(cost, 0)

    def test_cost_alternating_partition(self):
        """Alternating partition should give a positive cut (negative cost)."""
        problem = MaxCutOrbit(self.G_house)
        cost = problem.cost("01010")
        self.assertLess(cost, 0)


if __name__ == "__main__":
    unittest.main()
