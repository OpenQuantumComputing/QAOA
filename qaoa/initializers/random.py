"""
Random initializer – uniformly random angles.

Samples each angle independently from a uniform distribution over
``[0, 2π)``.  Supports reproducible runs via a random seed.

No previous depth is required.

Citation: Guo et al., arXiv:2606.05311 (2025).
"""

import numpy as np

from .base import Initializer


class Random(Initializer):
    """
    Uniformly random angle initializer.

    Args:
        seed (int | None): Random seed for reproducibility.  Default ``None``.
        n_candidates (int): Number of random candidates to return.
            :meth:`QAOA.optimize` will evaluate each and keep the best.
            Default 1.
        scale (float): Upper bound of the uniform distribution.
            Default ``2π``.

    Example::

        from qaoa import QAOA, initializers
        qaoa = QAOA(problem, mixer, initialstate,
                    initializer=initializers.Random(seed=42, n_candidates=5))
        qaoa.optimize(depth=3)
    """

    def __init__(self, seed=None, n_candidates=1, scale=None):
        self.seed = seed
        self.n_candidates = n_candidates
        self.scale = scale if scale is not None else 2 * np.pi

    def get_candidates(self, qaoa, depth: int) -> list:
        rng = np.random.default_rng(self.seed)
        n = qaoa.n_init + depth * (qaoa.n_gamma + qaoa.n_beta)
        candidates = []
        for _ in range(self.n_candidates):
            angles = np.zeros(n)
            # Initial-state params stay zero; randomise the rest.
            angles[qaoa.n_init:] = rng.uniform(0, self.scale, n - qaoa.n_init)
            candidates.append(angles)
        return candidates
