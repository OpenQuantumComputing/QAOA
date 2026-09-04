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

    Args:
        init_angles (array-like | None): Fixed values for the initial-state
            parameters.  When provided these values are used at every depth
            instead of whatever the optimiser found.  Only gamma and beta
            layers are interpolated; the initial-state prefix is **not**
            interpolated.  When *None* (default) the init-state parameters
            found at depth p-1 are carried forward unchanged, which is the
            same behaviour as before.

    Example::

        from qaoa import QAOA, initializers
        import numpy as np

        # Pin the initial-state angle to a fixed warm-start value
        qaoa = QAOA(problem, mixer, initialstate,
                    initializer=initializers.Interp(init_angles=[np.pi / 5]))
        qaoa.optimize(depth=3)
    """

    def __init__(self, init_angles=None):
        self.init_angles = None if init_angles is None else np.asarray(init_angles, dtype=float)

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
            init_part = (
                list(self.init_angles)
                if self.init_angles is not None
                else [getattr(qaoa, "best_init_val", 0.0)] * qaoa.n_init
            )
            candidate = np.array(
                init_part
                + [gamma_best] * qaoa.n_gamma
                + [beta_best] * qaoa.n_beta
            )
            return [candidate]

        prev_angles = qaoa.get_angles(depth - 1)
        return [_interp(prev_angles, qaoa.n_init, qaoa.n_gamma, qaoa.n_beta,
                        self.init_angles)]


# ---------------------------------------------------------------------------
# Pure function – also used by QAOA internally
# ---------------------------------------------------------------------------

def _interp(angles, n_init, n_gamma, n_beta, init_angles=None):
    """Apply the INTERP heuristic to *angles*, returning depth+1 parameters.

    Only the gamma/beta layer parameters are interpolated.  The leading
    *n_init* init-state parameters are carried forward from *angles* unless
    *init_angles* is provided, in which case those fixed values are used.
    """
    n_per_layer = n_gamma + n_beta
    depth = (len(angles) - n_init) // n_per_layer

    if init_angles is not None:
        init_part = list(init_angles)
    else:
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
