"""
TQA initializer – Trotterised Quantum Annealing schedule.

Sets gamma_l = (l / p) * dt  and  beta_l = (1 - l/p) * dt
for l = 1 … p, where *dt* is the Trotter step size (default ``π/4 / p``
makes the total time independent of *p*).

This corresponds to a first-order Trotterisation of the adiabatic
interpolation.  No previous depth is required.

Citation: Farhi et al., arXiv:2012.06523 (2020);
    Sack & Serbyn, PRX Quantum 2, 020322 (2021).
"""

import numpy as np

from .base import Initializer


class TQA(Initializer):
    """
    Trotterised Quantum Annealing (TQA) initializer.

    Args:
        dt (float | None): Trotter step size.  If ``None`` (default) the
            step ``π / (4 * depth)`` is used so that the total annealing
            time is always ``π/4``.

    Example::

        from qaoa import QAOA, initializers
        qaoa = QAOA(problem, mixer, initialstate,
                    initializer=initializers.TQA())
        qaoa.optimize(depth=3)
    """

    def __init__(self, dt=None):
        self.dt = dt

    def get_candidates(self, qaoa, depth: int) -> list:
        n_init = qaoa.n_init
        n_gamma = qaoa.n_gamma
        n_beta = qaoa.n_beta
        dt = self.dt if self.dt is not None else np.pi / (4 * depth)

        init_part = [0.0] * n_init
        layer_part = []
        for layer in range(1, depth + 1):
            gamma_val = (layer / depth) * dt
            beta_val = (1.0 - layer / depth) * dt
            layer_part += [gamma_val] * n_gamma + [beta_val] * n_beta

        return [np.array(init_part + layer_part)]
