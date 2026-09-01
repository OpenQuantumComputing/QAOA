import unittest
import numpy as np
import networkx as nx

from qaoa import QAOA
from qaoa import problems, mixers, initialstates, initializers
from qiskit_aer import AerSimulator


class TestMultiAngleGridSearch(unittest.TestCase):
    """Test grid search works for both vanilla and multi-angle QAOA."""

    def setUp(self):
        self.backend = AerSimulator()
        self.angles = {"gamma": [0, 2 * np.pi, 5], "beta": [0, 2 * np.pi, 5]}

    def test_vanilla_grid_search_sequential(self):
        """Vanilla QAOA (1 gamma, 1 beta) should complete grid search in sequential mode."""
        G = nx.path_graph(3)
        problem = problems.MaxCut(G)

        qaoa = QAOA(
            problem,
            mixers.X(),
            initialstates.Plus(),
            backend=self.backend,
            sequential=True,
        )

        qaoa.sample_cost_landscape(angles=self.angles)

        self.assertEqual(qaoa.Energy_sampled_p1.shape, (5, 5))

    def test_vanilla_grid_search_batch(self):
        """Vanilla QAOA (1 gamma, 1 beta) should complete grid search in batch mode."""
        G = nx.path_graph(3)
        problem = problems.MaxCut(G)

        qaoa = QAOA(
            problem,
            mixers.X(),
            initialstates.Plus(),
            backend=self.backend,
            sequential=False,
        )

        qaoa.sample_cost_landscape(angles=self.angles)

        self.assertEqual(qaoa.Energy_sampled_p1.shape, (5, 5))

    def test_multiangle_grid_search_sequential(self):
        """Multi-angle QAOA should complete 2D grid search (not 20^n) in sequential mode."""
        G = nx.house_graph()
        problem = problems.MaxCutOrbit(G)

        qaoa = QAOA(
            problem,
            mixers.X(),
            initialstates.Plus(),
            backend=self.backend,
            sequential=True,
        )

        # Should complete without error (not 5^5 evaluations)
        qaoa.sample_cost_landscape(angles=self.angles)

        # Result should still be 5x5 grid regardless of parameter count
        self.assertEqual(qaoa.Energy_sampled_p1.shape, (5, 5))

    def test_multiangle_grid_search_batch(self):
        """Multi-angle QAOA should complete 2D grid search (not 20^n) in batch mode."""
        G = nx.house_graph()
        problem = problems.MaxCutOrbit(G)

        qaoa = QAOA(
            problem,
            mixers.X(),
            initialstates.Plus(),
            backend=self.backend,
            sequential=False,
        )

        # Should complete without error
        qaoa.sample_cost_landscape(angles=self.angles)

        # Result should still be 5x5 grid
        self.assertEqual(qaoa.Energy_sampled_p1.shape, (5, 5))

    def test_vanilla_warmstart_initialization(self):
        """Multi-angle optimize should broadcast vanilla angles to all params."""
        G = nx.house_graph()
        problem = problems.MaxCutOrbit(G)

        qaoa = QAOA(
            problem,
            mixers.X(),
            initialstates.Plus(),
            backend=self.backend,
            sequential=True,
        )

        qaoa.optimize(depth=1, angles=self.angles)

        self.assertEqual(qaoa.current_depth, 1)
        self.assertIn(1, qaoa.optimization_results)

    def test_layer_grid_search_depth2(self):
        """
        Layer-by-layer grid search (LayerGrid initializer) at depth 2 should:
          1. Complete without error.
          2. Store results at depth 2.
          3. Yield energy(2) ≤ energy(1) (monotonic) because the grid includes (0,0).
        """
        G = nx.path_graph(3)
        problem = problems.MaxCut(G)

        qaoa = QAOA(
            problem,
            mixers.X(),
            initialstates.Plus(),
            backend=self.backend,
            sequential=True,
            initializer=initializers.LayerGrid(),
        )

        qaoa.optimize(depth=2, angles=self.angles)

        self.assertEqual(qaoa.current_depth, 2)
        self.assertIn(1, qaoa.optimization_results)
        self.assertIn(2, qaoa.optimization_results)

        energy1 = qaoa.get_energy(depth=1)
        energy2 = qaoa.get_energy(depth=2)
        # Energy is minimized, so a better result is more negative for MaxCut.
        self.assertLessEqual(energy2, energy1 + 1e-6)  # small tolerance for shot noise

    def test_layer_grid_search_multiangle(self):
        """Layer grid search should work for free (multi-angle) ansatz."""
        G = nx.path_graph(3)

        qaoa = QAOA(
            problems.MaxCutFree(G),
            mixers.XMultiAngle(),
            initialstates.Plus(),
            backend=self.backend,
            sequential=True,
            initializer=initializers.LayerGrid(),
        )

        qaoa.optimize(depth=2, angles=self.angles)

        self.assertEqual(qaoa.current_depth, 2)
        self.assertIn(2, qaoa.optimization_results)


if __name__ == "__main__":
    unittest.main()
