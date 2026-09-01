"""
Interp initializer – INTERP heuristic (linear interpolation).

Produces an initial guess for depth p+1 by linearly interpolating the
optimised angles from depth p.  Works well for vanilla QAOA but does not
provide a monotonic approximation-ratio guarantee.

Citation: Zhou et al., PRX 10, 021067 (2020).
    https://doi.org/10.1103/PhysRevX.10.021067
"""

import numpy as np

from .base import Initializer


class Interp(Initializer):
    """
    INTERP heuristic initializer.

    Requires the previous depth to have been optimised.  No additional
    circuit evaluations are performed.

    Example::

        from qaoa import QAOA, initializers
        qaoa = QAOA(problem, mixer, initialstate,
                    initializer=initializers.Interp())
        qaoa.optimize(depth=3)
    """

    def get_candidates(self, qaoa, depth: int) -> list:
        """
        Return one candidate obtained by interpolating the previous depth's
        optimised angles.

        At depth 1 a grid search is always performed first (same as
        ``LayerGrid``) to obtain a meaningful starting point.
        """
        if depth == 1:
            # Bootstrap with grid at depth 1.
            angles_spec = getattr(qaoa, "_angles_spec", None) or {
                "gamma": [0, 2 * np.pi, 20],
                "beta": [0, 2 * np.pi, 20],
            }
            if qaoa.Energy_sampled_p1 is None:
                qaoa.sample_cost_landscape(angles=angles_spec)
            ind = np.unravel_index(
                np.argmin(qaoa.Energy_sampled_p1),
                qaoa.Energy_sampled_p1.shape,
            )
            gamma_best = qaoa.gamma_grid[ind[1]]
            beta_best = qaoa.beta_grid[ind[0]]
            candidate = np.array(
                [0.0] * qaoa.n_init
                + [gamma_best] * qaoa.n_gamma
                + [beta_best] * qaoa.n_beta
            )
            return [candidate]

        prev_angles = qaoa.get_angles(depth - 1)
        return [_interp(prev_angles, qaoa.n_init, qaoa.n_gamma, qaoa.n_beta)]


# ---------------------------------------------------------------------------
# Pure function – also used by QAOA internally
# ---------------------------------------------------------------------------

def _interp(angles, n_init, n_gamma, n_beta):
    """Apply the INTERP heuristic to *angles*, returning depth+1 parameters."""
    n_per_layer = n_gamma + n_beta
    depth = (len(angles) - n_init) // n_per_layer

    init_part = list(angles[:n_init])
    layer_angles = np.array(angles[n_init:]).reshape(depth, n_per_layer)

    result_layers = np.zeros((depth + 1, n_per_layer))
    for i in range(n_per_layer):
        param_vals = layer_angles[:, i]
        tmp = np.zeros(depth + 2)
        tmp[1:-1] = param_vals
        w = np.arange(0, depth + 1)
        result_layers[:, i] = w / depth * tmp[:-1] + (depth - w) / depth * tmp[1:]

    return np.concatenate([init_part, result_layers.flatten()])
