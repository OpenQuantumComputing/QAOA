"""
Unit tests for Statistic, BitFlip and post-processing utilities.

Covers:
- qaoa.util.Statistic: expectation, variance, min/max, CVaR
- qaoa.util.BitFlip: boost_samples and xor
- qaoa.util.post_processing: functional post-processing
"""

import math
import unittest

import numpy as np
import networkx as nx


class TestStatistic(unittest.TestCase):
    """Tests for the Statistic class."""

    def setUp(self):
        from qaoa.util import Statistic
        self.Statistic = Statistic

    def test_single_sample_expectation(self):
        stat = self.Statistic()
        stat.add_sample(3.0, 1.0, "0")
        self.assertAlmostEqual(stat.get_E(), 3.0)

    def test_two_equal_weight_samples(self):
        stat = self.Statistic()
        stat.add_sample(2.0, 1.0, "0")
        stat.add_sample(4.0, 1.0, "1")
        self.assertAlmostEqual(stat.get_E(), 3.0)

    def test_weighted_expectation(self):
        stat = self.Statistic()
        stat.add_sample(0.0, 1.0, "0")
        stat.add_sample(10.0, 9.0, "1")
        # E = (1*0 + 9*10) / 10 = 9
        self.assertAlmostEqual(stat.get_E(), 9.0)

    def test_max_and_min(self):
        stat = self.Statistic()
        for v in [5.0, 1.0, 3.0, 8.0, 2.0]:
            stat.add_sample(v, 1.0, str(v))
        self.assertAlmostEqual(stat.get_max(), 8.0)
        self.assertAlmostEqual(stat.get_min(), 1.0)

    def test_max_sols_and_min_sols(self):
        stat = self.Statistic()
        stat.add_sample(5.0, 1.0, "five")
        stat.add_sample(5.0, 1.0, "also_five")
        stat.add_sample(1.0, 1.0, "one")
        self.assertIn("five", stat.get_max_sols())
        self.assertIn("also_five", stat.get_max_sols())
        self.assertIn("one", stat.get_min_sols())

    def test_variance(self):
        stat = self.Statistic()
        stat.add_sample(2.0, 1.0, "a")
        stat.add_sample(4.0, 1.0, "b")
        # population variance = 1, Bessel-corrected: S/(n-1) = 2/1 = 2
        self.assertAlmostEqual(stat.get_Variance(), 2.0)

    def test_cvar_below_one(self):
        """CVaR with alpha=0.5 should average over top 50% of values."""
        stat = self.Statistic(cvar=0.5)
        # add 4 samples: 1, 2, 3, 4
        for v in [1.0, 2.0, 3.0, 4.0]:
            stat.add_sample(v, 1.0, str(v))
        # Top 50% = [3, 4], CVaR = 3.5
        self.assertAlmostEqual(stat.get_CVaR(), 3.5)

    def test_cvar_equal_one_is_expectation(self):
        stat = self.Statistic(cvar=1)
        for v in [1.0, 2.0, 3.0]:
            stat.add_sample(v, 1.0, str(v))
        self.assertAlmostEqual(stat.get_CVaR(), stat.get_E())

    def test_reset(self):
        stat = self.Statistic()
        stat.add_sample(10.0, 1.0, "a")
        stat.reset()
        stat.add_sample(5.0, 1.0, "b")
        self.assertAlmostEqual(stat.get_E(), 5.0)
        self.assertAlmostEqual(stat.get_max(), 5.0)

    def test_get_CVaR_is_get_E_when_cvar_is_1(self):
        stat = self.Statistic(cvar=1)
        stat.add_sample(3.0, 2.0, "x")
        stat.add_sample(7.0, 2.0, "y")
        self.assertAlmostEqual(stat.get_CVaR(), stat.get_E())


class TestBitFlip(unittest.TestCase):
    """Tests for the BitFlip class."""

    def setUp(self):
        from qaoa.util import BitFlip
        self.BitFlip = BitFlip

    def _make_maxcut_problem(self):
        """Create a small MaxCut problem on K2 for testing."""
        import networkx as nx
        from qaoa.problems import MaxCut
        G = nx.complete_graph(2)
        return MaxCut(G)

    def test_boost_samples_returns_string(self):
        problem = self._make_maxcut_problem()
        flipper = self.BitFlip(problem.N_qubits)
        result = flipper.boost_samples(problem=problem, string="10", K=5)
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 2)

    def test_boost_finds_optimal_for_k2(self):
        """For MaxCut on K2, boost should find "01" or "10" from "00"."""
        problem = self._make_maxcut_problem()
        flipper = self.BitFlip(problem.N_qubits)
        result = flipper.boost_samples(problem=problem, string="00", K=10)
        # The optimal solutions are "01" and "10"
        self.assertIn(result, ["01", "10"])

    def test_xor_no_change(self):
        flipper = self.BitFlip(3)
        xor = flipper.xor("101", "101")
        self.assertEqual(xor, [False, False, False])

    def test_xor_all_flipped(self):
        flipper = self.BitFlip(3)
        xor = flipper.xor("000", "111")
        self.assertEqual(xor, [True, True, True])

    def test_xor_partial(self):
        flipper = self.BitFlip(4)
        xor = flipper.xor("1010", "0010")
        self.assertEqual(xor[0], True)
        self.assertEqual(xor[1], False)
        self.assertEqual(xor[2], False)
        self.assertEqual(xor[3], False)


class TestPostProcessing(unittest.TestCase):
    """Tests for the post_processing utility."""

    def test_post_processing_returns_dict(self):
        import networkx as nx
        from qaoa.problems import MaxCut
        from qaoa.util import BitFlip, Statistic
        from qaoa.util import post_processing

        G = nx.path_graph(3)
        problem = MaxCut(G)

        # Create a minimal mock instance
        class MockInstance:
            pass

        instance = MockInstance()
        instance.problem = problem
        instance.flipper = BitFlip(problem.N_qubits)
        instance.stat = Statistic()

        samples = {"100": 5, "010": 3}
        result = post_processing(instance, samples, K=3)
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_post_processing_string_input(self):
        import networkx as nx
        from qaoa.problems import MaxCut
        from qaoa.util import BitFlip, Statistic
        from qaoa.util import post_processing

        G = nx.path_graph(3)
        problem = MaxCut(G)

        class MockInstance:
            pass

        instance = MockInstance()
        instance.problem = problem
        instance.flipper = BitFlip(problem.N_qubits)
        instance.stat = Statistic()

        result = post_processing(instance, "100", K=3)
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
