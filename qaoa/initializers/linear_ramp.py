"""
LinearRamp initializer – linearly-spaced angles.

Sets gamma_l = l * t / p  and  beta_l = (1 - l/p) * t
for l = 1 … p, where *t* is a single scalar (default π/4).

This mimics a simple linear annealing schedule.  No previous depth is
required.

Citation: Farhi et al., arXiv:1411.4028 (2014); Guo et al.,
    arXiv:2606.05311 (2025).
"""

import numpy as np

from .base import Initializer


class LinearRamp(Initializer):
    """
    Linear-ramp angle initializer.

    Args:
        t (float): Total ramp time parameter.  Defaults to ``π/4``.

    Example::

        from qaoa import QAOA, initializers
        qaoa = QAOA(problem, mixer, initialstate,
                    initializer=initializers.LinearRamp(t=np.pi / 4))
        qaoa.optimize(depth=3)
    """

    def __init__(self, t=None):
        self.t = t if t is not None else np.pi / 4

    def get_candidates(self, qaoa, depth: int) -> list:
        n_init = qaoa.n_init
        n_gamma = qaoa.n_gamma
        n_beta = qaoa.n_beta
        t = self.t

        init_part = [0.0] * n_init
        layer_part = []
        for layer in range(1, depth + 1):
            gamma_val = layer * t / depth
            beta_val = (1.0 - layer / depth) * t
            layer_part += [gamma_val] * n_gamma + [beta_val] * n_beta

        return [np.array(init_part + layer_part)]
