import unittest
import os
import tempfile
import numpy as np

from qaoa.utils import qaoaIO as qio


class TestQAOAIORoundtrip(unittest.TestCase):
    """Roundtrip load/save test using the ExactCover example data."""

    @classmethod
    def setUpClass(cls):
        cls.example_path = os.path.join(
            os.path.dirname(__file__),
            "..", "examples", "ExactCover", "data",
            "exact_cover_path_problem_example.json",
        )
        if not os.path.exists(cls.example_path):
            raise unittest.SkipTest(f"Example file not found: {cls.example_path}")

    def test_load_fields(self):
        result = qio.QAOAResult.load(self.example_path)

        self.assertIsInstance(result.problem, qio.ExactCoverProblemData)
        self.assertEqual(result.problem.problem_type, "ExactCover")
        self.assertEqual(result.problem.hamming_weight, 2)
        self.assertIsInstance(result.problem.columns, np.ndarray)
        self.assertIsInstance(result.problem.weights, np.ndarray)
        self.assertEqual(result.problem.columns.shape[1], result.qaoa_params.N_qubits)

        self.assertEqual(result.qaoa_params.mixer_method, qio.MixerMethod.GROVER)
        self.assertEqual(result.qaoa_params.init_method, qio.InitMethod.DICKE)
        self.assertGreater(len(result.qaoa_params.depths), 0)
        self.assertIn("timestamp", result.metadata)

    def test_roundtrip_preserves_data(self):
        original = qio.QAOAResult.load(self.example_path)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            original.save(tmp_path)
            reloaded = qio.QAOAResult.load(tmp_path)

            # Problem data
            np.testing.assert_array_almost_equal(
                reloaded.problem.columns, original.problem.columns
            )
            np.testing.assert_array_almost_equal(
                reloaded.problem.weights, original.problem.weights
            )
            np.testing.assert_array_almost_equal(
                reloaded.problem.solution, original.problem.solution
            )
            self.assertEqual(
                reloaded.problem.hamming_weight, original.problem.hamming_weight
            )
            self.assertEqual(
                reloaded.problem.problem_type, original.problem.problem_type
            )

            # QAOA parameters
            self.assertAlmostEqual(
                reloaded.qaoa_params.cvar, original.qaoa_params.cvar
            )
            self.assertEqual(
                reloaded.qaoa_params.init_method, original.qaoa_params.init_method
            )
            self.assertEqual(
                reloaded.qaoa_params.mixer_method, original.qaoa_params.mixer_method
            )
            self.assertEqual(
                reloaded.qaoa_params.backend, original.qaoa_params.backend
            )
            self.assertEqual(
                reloaded.qaoa_params.optimizer, original.qaoa_params.optimizer
            )
            self.assertEqual(
                reloaded.qaoa_params.N_qubits, original.qaoa_params.N_qubits
            )

            # Depth results
            self.assertEqual(
                set(reloaded.qaoa_params.depths.keys()),
                set(original.qaoa_params.depths.keys()),
            )
            for depth_key in original.qaoa_params.depths:
                orig_dr = original.qaoa_params.depths[depth_key]
                reload_dr = reloaded.qaoa_params.depths[depth_key]
                self.assertEqual(reload_dr.optimal_angles, orig_dr.optimal_angles)
                self.assertEqual(reload_dr.histogram, orig_dr.histogram)
                self.assertAlmostEqual(reload_dr.opt_time, orig_dr.opt_time)

        finally:
            os.unlink(tmp_path)

    def test_roundtrip_reconstructs_problem(self):
        result = qio.QAOAResult.load(self.example_path)
        ec_problem = result.get_problem_instance()

        self.assertEqual(ec_problem.N_qubits, result.qaoa_params.N_qubits)
        np.testing.assert_array_almost_equal(
            ec_problem.columns, result.problem.columns
        )

        opt_sol = ec_problem.brute_force_solve()
        self.assertTrue(ec_problem.isFeasible(opt_sol))


if __name__ == "__main__":
    unittest.main()
