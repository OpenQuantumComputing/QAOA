"""
FixedAngles initializer – user-supplied or transferred angles.

Accepts a pre-computed flat angle array (e.g. from a previous run or from
literature) and broadcasts it to the requested depth.

If the supplied array is shorter than ``n_init + depth * n_per_layer`` the
remaining angles are padded with zeros (useful for transferring p-1 angles
to depth p).  If it is longer, it is truncated.

No additional circuit evaluations are performed.

Citation: Guo et al., arXiv:2606.05311 (2025).
"""

import numpy as np

from .base import Initializer


class FixedAngles(Initializer):
    """
    Fixed / transferred angle initializer.

    Args:
        angles (array-like): Flat angle array to use as the starting point.

    Example::

        from qaoa import QAOA, initializers
        import numpy as np

        known_angles = np.array([0.5, 0.3])          # depth-1 vanilla angles
        qaoa = QAOA(problem, mixer, initialstate,
                    initializer=initializers.FixedAngles(known_angles))
        qaoa.optimize(depth=1)
    """

    def __init__(self, angles):
        self.angles = np.asarray(angles, dtype=float)

    def get_candidates(self, qaoa, depth: int) -> list:
        n_total = qaoa.n_init + depth * (qaoa.n_gamma + qaoa.n_beta)
        src = self.angles
        if len(src) >= n_total:
            candidate = src[:n_total].copy()
        else:
            candidate = np.zeros(n_total)
            candidate[: len(src)] = src
        return [candidate]
