"""
Unit tests for qaoa.initializers.

Tests are pure (no quantum backend) where possible; only LayerGrid tests
require a backend to evaluate energy.
"""

import unittest
import numpy as np
import networkx as nx


def _simple_qaoa(depth=1, sequential=True):
    """Return a cheap QAOA instance (3-node MaxCut, AerSimulator) at depth 0."""
    from qaoa import QAOA, problems, mixers, initialstates
    from qiskit_aer import AerSimulator

    G = nx.path_graph(3)
    qaoa = QAOA(
        problems.MaxCut(G),
        mixers.X(),
        initialstates.Plus(),
        backend=AerSimulator(),
        shots=128,
        sequential=sequential,
    )
    return qaoa


class TestInitializerImports(unittest.TestCase):
    def test_all_classes_importable(self):
        from qaoa import initializers
        for cls in [
            "Initializer", "LayerGrid", "Interp", "LinearRamp",
            "TQA", "Random", "FixedAngles", "Fourier",
        ]:
            self.assertTrue(hasattr(initializers, cls), f"Missing: {cls}")


# ---------------------------------------------------------------------------
# LinearRamp
# ---------------------------------------------------------------------------

class TestLinearRamp(unittest.TestCase):
    def _fake_qaoa(self, n_gamma=1, n_beta=1, n_init=0):
        class Q:
            pass
        q = Q()
        q.n_gamma = n_gamma
        q.n_beta = n_beta
        q.n_init = n_init
        return q

    def test_depth1_shape(self):
        from qaoa.initializers import LinearRamp
        q = self._fake_qaoa()
        candidates = LinearRamp().get_candidates(q, depth=1)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(len(candidates[0]), 2)  # n_init + 1*(1+1)

    def test_depth3_shape(self):
        from qaoa.initializers import LinearRamp
        q = self._fake_qaoa()
        candidates = LinearRamp().get_candidates(q, depth=3)
        self.assertEqual(len(candidates[0]), 6)

    def test_last_gamma_equals_t(self):
        from qaoa.initializers import LinearRamp
        q = self._fake_qaoa()
        t = np.pi / 4
        c = LinearRamp(t=t).get_candidates(q, depth=4)[0]
        # gamma_l = l*t/p, for l=p → t
        self.assertAlmostEqual(c[-2], t, places=10)

    def test_first_beta_decreasing(self):
        from qaoa.initializers import LinearRamp
        q = self._fake_qaoa()
        c = LinearRamp().get_candidates(q, depth=4)[0]
        betas = c[1::2]  # beta values at each layer
        self.assertGreater(betas[0], betas[-1])


# ---------------------------------------------------------------------------
# TQA
# ---------------------------------------------------------------------------

class TestTQA(unittest.TestCase):
    def _fake_qaoa(self):
        class Q:
            n_gamma = 1
            n_beta = 1
            n_init = 0
        return Q()

    def test_depth1_shape(self):
        from qaoa.initializers import TQA
        c = TQA().get_candidates(self._fake_qaoa(), depth=1)[0]
        self.assertEqual(len(c), 2)

    def test_zero_beta_at_last_layer(self):
        from qaoa.initializers import TQA
        # At l=p: beta_l = (1 - p/p)*dt = 0
        c = TQA(dt=0.5).get_candidates(self._fake_qaoa(), depth=4)[0]
        last_beta = c[-1]
        self.assertAlmostEqual(last_beta, 0.0, places=10)

    def test_custom_dt(self):
        from qaoa.initializers import TQA
        dt = 0.3
        c = TQA(dt=dt).get_candidates(self._fake_qaoa(), depth=2)[0]
        # gamma_1 = (1/2)*dt
        self.assertAlmostEqual(c[0], 0.5 * dt, places=10)


# ---------------------------------------------------------------------------
# Random
# ---------------------------------------------------------------------------

class TestRandom(unittest.TestCase):
    def _fake_qaoa(self):
        class Q:
            n_gamma = 1
            n_beta = 1
            n_init = 0
        return Q()

    def test_reproducible(self):
        from qaoa.initializers import Random
        q = self._fake_qaoa()
        c1 = Random(seed=7).get_candidates(q, depth=3)[0]
        c2 = Random(seed=7).get_candidates(q, depth=3)[0]
        np.testing.assert_array_equal(c1, c2)

    def test_n_candidates(self):
        from qaoa.initializers import Random
        q = self._fake_qaoa()
        cs = Random(seed=0, n_candidates=5).get_candidates(q, depth=2)
        self.assertEqual(len(cs), 5)

    def test_within_scale(self):
        from qaoa.initializers import Random
        q = self._fake_qaoa()
        scale = np.pi
        cs = Random(seed=1, n_candidates=20, scale=scale).get_candidates(q, depth=3)
        for c in cs:
            self.assertTrue(np.all(c >= 0))
            self.assertTrue(np.all(c <= scale))


# ---------------------------------------------------------------------------
# FixedAngles
# ---------------------------------------------------------------------------

class TestFixedAngles(unittest.TestCase):
    def _fake_qaoa(self):
        class Q:
            n_gamma = 1
            n_beta = 1
            n_init = 0
        return Q()

    def test_exact_length(self):
        from qaoa.initializers import FixedAngles
        q = self._fake_qaoa()
        angles = np.array([0.5, 0.3])
        c = FixedAngles(angles).get_candidates(q, depth=1)[0]
        np.testing.assert_array_equal(c, angles)

    def test_shorter_pads_zeros(self):
        from qaoa.initializers import FixedAngles
        q = self._fake_qaoa()
        angles = np.array([0.5])
        c = FixedAngles(angles).get_candidates(q, depth=2)[0]
        self.assertEqual(len(c), 4)
        self.assertEqual(c[0], 0.5)
        self.assertEqual(c[1], 0.0)

    def test_longer_truncated(self):
        from qaoa.initializers import FixedAngles
        q = self._fake_qaoa()
        angles = np.arange(10, dtype=float)
        c = FixedAngles(angles).get_candidates(q, depth=1)[0]
        self.assertEqual(len(c), 2)
        np.testing.assert_array_equal(c, angles[:2])


# ---------------------------------------------------------------------------
# Fourier
# ---------------------------------------------------------------------------

class TestFourier(unittest.TestCase):
    def _fake_qaoa(self, n_gamma=1, n_beta=1):
        class Q:
            n_init = 0
        Q.n_gamma = n_gamma
        Q.n_beta = n_beta
        return Q()

    def test_zero_coefficients_give_zero(self):
        from qaoa.initializers import Fourier
        q = self._fake_qaoa()
        c = Fourier(u=[0.0], v=[0.0]).get_candidates(q, depth=3)[0]
        np.testing.assert_allclose(c, 0.0, atol=1e-12)

    def test_depth_shape(self):
        from qaoa.initializers import Fourier
        q = self._fake_qaoa()
        c = Fourier(u=[0.1, 0.05], v=[0.2, 0.1]).get_candidates(q, depth=4)[0]
        self.assertEqual(len(c), 8)  # 4 layers × (1+1)

    def test_rejects_multiangle(self):
        from qaoa.initializers import Fourier
        q = self._fake_qaoa(n_gamma=2, n_beta=2)
        with self.assertRaises(ValueError):
            Fourier().get_candidates(q, depth=2)

    def test_uv_length_mismatch(self):
        from qaoa.initializers import Fourier
        with self.assertRaises(ValueError):
            Fourier(u=[0.1, 0.2], v=[0.3])


# ---------------------------------------------------------------------------
# Interp (pure function)
# ---------------------------------------------------------------------------

class TestInterp(unittest.TestCase):
    def test_depth1_to_depth2(self):
        from qaoa.initializers.interp import _interp
        angles = np.array([0.5, 0.3])
        result = _interp(angles, n_init=0, n_gamma=1, n_beta=1)
        self.assertEqual(len(result), 4)

    def test_boundary_values(self):
        """At depth 1, _interp should give 0, v0, v0 for each param (INTERP formula)."""
        from qaoa.initializers.interp import _interp
        angles = np.array([1.0, 2.0])
        result = _interp(angles, n_init=0, n_gamma=1, n_beta=1)
        # INTERP: new layer 1 gets 0*v0 + 1*v0 = v0; layer 2 gets 1*v0 + 0*0 = v0
        # For depth=1: tmp=[0, g, 0], w=[0,1], result_l0 = 0/1 * 0 + 1/1 * g = g, result_l1 = 1/1 * g + 0/1 * 0 = g
        np.testing.assert_allclose(result, [1.0, 2.0, 1.0, 2.0], atol=1e-12)


# ---------------------------------------------------------------------------
# LayerGrid (requires backend)
# ---------------------------------------------------------------------------

class TestLayerGridWithBackend(unittest.TestCase):
    def setUp(self):
        from qaoa import QAOA, problems, mixers, initialstates, initializers
        from qiskit_aer import AerSimulator
        G = nx.path_graph(3)
        self.qaoa = QAOA(
            problems.MaxCut(G),
            mixers.X(),
            initialstates.Plus(),
            backend=AerSimulator(),
            shots=128,
            sequential=True,
            initializer=initializers.LayerGrid(
                gamma_values=[0, 2 * np.pi, 5],
                beta_values=[0, 2 * np.pi, 5],
            ),
        )

    def test_depth1_returns_candidate(self):
        angles = {"gamma": [0, 2 * np.pi, 5], "beta": [0, 2 * np.pi, 5]}
        self.qaoa.optimize(depth=1, angles=angles)
        self.assertEqual(self.qaoa.current_depth, 1)
        self.assertIn(1, self.qaoa.optimization_results)

    def test_depth2_monotonic(self):
        angles = {"gamma": [0, 2 * np.pi, 5], "beta": [0, 2 * np.pi, 5]}
        self.qaoa.optimize(depth=2, angles=angles)
        e1 = self.qaoa.get_energy(depth=1)
        e2 = self.qaoa.get_energy(depth=2)
        self.assertLessEqual(e2, e1 + 1e-6)

    def test_zero_always_evaluated(self):
        """LayerGrid with a non-zero-starting range still evaluates (0,0)."""
        from qaoa import QAOA, problems, mixers, initialstates, initializers
        from qiskit_aer import AerSimulator
        G = nx.path_graph(3)
        qaoa = QAOA(
            problems.MaxCut(G),
            mixers.X(),
            initialstates.Plus(),
            backend=AerSimulator(),
            shots=128,
            sequential=True,
            initializer=initializers.LayerGrid(
                gamma_values=[0.1, 2 * np.pi, 4],
                beta_values=[0.1, 2 * np.pi, 4],
            ),
        )
        # depth=2: grid doesn't include 0, but we explicitly add it
        qaoa.optimize(depth=2, angles={"gamma": [0.1, 2 * np.pi, 4], "beta": [0.1, 2 * np.pi, 4]})
        e1 = qaoa.get_energy(depth=1)
        e2 = qaoa.get_energy(depth=2)
        # monotonic because (0,0) is always tried
        self.assertLessEqual(e2, e1 + 1e-6)


if __name__ == "__main__":
    unittest.main()
