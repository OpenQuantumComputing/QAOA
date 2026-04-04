"""
Round-trip IO tests for QAOAResult with BucketExactCover problems.
"""

import os
import tempfile
import unittest

import numpy as np

from qaoa.utils.qaoaIO import (
    BucketExactCoverProblemData,
    DepthResult,
    InitMethod,
    MixerMethod,
    QAOAParameters,
    QAOAResult,
)


class TestBucketExactCoverIO(unittest.TestCase):
    """Round-trip IO tests for BucketExactCover."""

    # Same small problem used in test_problems.py
    _COLUMNS = np.array([
        [1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1],
        [1, 0, 0, 1, 0, 1, 1],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 1, 0, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 1, 0],
    ], dtype=float)
    _WEIGHTS = np.array([20., 40., 30., 35., 10., 25., 40.])
    _NUM_BUCKETS = 2

    def _make_result(self):
        from qaoa.problems import BucketExactCover

        bec = BucketExactCover(
            self._COLUMNS,
            num_buckets=self._NUM_BUCKETS,
            weights=self._WEIGHTS,
        )

        problem_data = BucketExactCoverProblemData(
            columns=self._COLUMNS,
            weights=self._WEIGHTS,
            num_buckets=self._NUM_BUCKETS,
        )

        qaoa_params = QAOAParameters(
            cvar=1.0,
            init_method=InitMethod.PLUS,
            mixer_method=MixerMethod.X,
            backend="statevector_simulator",
            optimizer="COBYLA",
            N_qubits=bec.N_qubits,
            depths={},
        )

        return QAOAResult(problem=problem_data, qaoa_params=qaoa_params)

    def _temp_json(self):
        """Return a path to a temporary JSON file and register cleanup."""
        f = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        f.close()
        self.addCleanup(os.remove, f.name)
        return f.name

    def test_save_load_problem_type(self):
        result = self._make_result()
        fname = self._temp_json()
        result.save(fname)
        loaded = QAOAResult.load(fname)
        self.assertEqual(loaded.problem.problem_type, "BucketExactCover")

    def test_save_load_columns(self):
        result = self._make_result()
        fname = self._temp_json()
        result.save(fname)
        loaded = QAOAResult.load(fname)
        np.testing.assert_array_equal(loaded.problem.columns, self._COLUMNS)

    def test_save_load_weights(self):
        result = self._make_result()
        fname = self._temp_json()
        result.save(fname)
        loaded = QAOAResult.load(fname)
        np.testing.assert_allclose(loaded.problem.weights, self._WEIGHTS)

    def test_save_load_num_buckets(self):
        result = self._make_result()
        fname = self._temp_json()
        result.save(fname)
        loaded = QAOAResult.load(fname)
        self.assertEqual(loaded.problem.num_buckets, self._NUM_BUCKETS)

    def test_save_load_mixer_bucketwise_grover(self):
        """BucketwiseGrover serializes as BUCKETWISEGROVER and loads back."""
        result = self._make_result()
        result.qaoa_params.mixer_method = MixerMethod.BUCKETWISEGROVER
        fname = self._temp_json()
        result.save(fname)
        loaded = QAOAResult.load(fname)
        self.assertEqual(loaded.qaoa_params.mixer_method, MixerMethod.BUCKETWISEGROVER)

    def test_get_problem_instance_cost_matches(self):
        """Reconstruct via get_problem_instance() and verify cost() matches."""
        from qaoa.problems import BucketExactCover

        result = self._make_result()
        fname = self._temp_json()
        result.save(fname)
        loaded = QAOAResult.load(fname)

        original_bec = BucketExactCover(
            self._COLUMNS,
            num_buckets=self._NUM_BUCKETS,
            weights=self._WEIGHTS,
        )
        reconstructed_bec = loaded.get_problem_instance()

        # Test cost of [c0, c4] bitstring (bucket0 v=0→c0, bucket1 v=1→c4)
        test_bitstring = "0010"
        self.assertAlmostEqual(
            original_bec.unscaled_cost(test_bitstring),
            reconstructed_bec.unscaled_cost(test_bitstring),
        )


if __name__ == "__main__":
    unittest.main()
