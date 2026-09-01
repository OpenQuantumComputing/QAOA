"""
LayerGrid initializer – layer-by-layer 2-D grid search.

At depth 1 the depth-1 grid from :meth:`QAOA.sample_cost_landscape` is
reused directly.  At depth p > 1 the previous best angles are kept fixed
and a 2-D grid search is performed over the *new* layer's parameters in the
vanilla (symmetric) subspace.

The candidate ``(gamma=0, beta=0)`` — which corresponds to an identity layer
that exactly reproduces the depth-(p-1) circuit — is **always** evaluated,
regardless of the grid range.  This guarantees a deterministic monotonic
approximation ratio: ``energy(p) ≤ energy(p-1)``.

Citation: Guo et al., arXiv:2606.05311 (2025).
"""

import numpy as np
import structlog

from .base import Initializer

LOG = structlog.get_logger(file=__name__)


class LayerGrid(Initializer):
    """
    Layer-by-layer 2-D grid initializer with monotonic guarantee.

    Args:
        gamma_values (list): ``[start, stop, num]`` for the gamma grid.
            Defaults to ``[0, 2π, 20]``.
        beta_values (list): ``[start, stop, num]`` for the beta grid.
            Defaults to ``[0, 2π, 20]``.

    Example::

        from qaoa import QAOA, initializers
        qaoa = QAOA(
            problem, mixer, initialstate,
            initializer=initializers.LayerGrid(
                gamma_values=[0, 2*np.pi, 10],
                beta_values=[0, 2*np.pi, 10],
            ),
        )
        qaoa.optimize(depth=3)
    """

    def __init__(self, gamma_values=None, beta_values=None):
        if gamma_values is None:
            gamma_values = [0, 2 * np.pi, 20]
        if beta_values is None:
            beta_values = [0, 2 * np.pi, 20]
        self.gamma_values = gamma_values
        self.beta_values = beta_values

    # ------------------------------------------------------------------
    # Initializer protocol
    # ------------------------------------------------------------------

    def get_candidates(self, qaoa, depth: int) -> list:
        """
        Return a single best candidate for *depth*.

        At depth 1 the landscape already computed by
        :meth:`~qaoa.QAOA.sample_cost_landscape` is reused.  At depth > 1
        a grid search over the new layer is performed with previous layers
        locked at their optimal angles.
        """
        angles_spec = getattr(qaoa, "_angles_spec", None) or {
            "gamma": self.gamma_values,
            "beta": self.beta_values,
        }
        # Prefer the user-supplied values when present, fall back to own spec.
        angles_spec = {
            "gamma": angles_spec.get("gamma", self.gamma_values),
            "beta": angles_spec.get("beta", self.beta_values),
        }

        if depth == 1:
            # Reuse (or compute) the depth-1 landscape grid.
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

        # depth > 1 – grid search over the new layer.
        prev_angles = qaoa.get_angles(depth - 1)
        return [self._grid_search_layer(qaoa, prev_angles, angles_spec)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _grid_search_layer(self, qaoa, prev_angles, angles_spec):
        """Grid over the new layer with previous layers locked."""
        n_per_layer = qaoa.n_gamma + qaoa.n_beta
        new_depth = (len(prev_angles) - qaoa.n_init) // n_per_layer + 1
        qaoa.createParameterizedCircuit(new_depth)

        gamma_grid = np.linspace(
            angles_spec["gamma"][0],
            angles_spec["gamma"][1],
            angles_spec["gamma"][2],
            endpoint=False,
        )
        beta_grid = np.linspace(
            angles_spec["beta"][0],
            angles_spec["beta"][1],
            angles_spec["beta"][2],
            endpoint=False,
        )

        logger = LOG.bind(func="LayerGrid._grid_search_layer")
        logger.info(
            f"Layer grid search at depth {new_depth}: "
            f"{len(gamma_grid)}×{len(beta_grid)} points (+explicit zero)"
        )

        best_energy = np.inf
        best_candidate = None

        # Collect all (gamma, beta) pairs; always include (0, 0).
        pairs = [(g, b) for b in beta_grid for g in gamma_grid]
        if not any(g == 0.0 and b == 0.0 for g, b in pairs):
            pairs.append((0.0, 0.0))

        for gamma_val, beta_val in pairs:
            new_layer = np.array(
                [gamma_val] * qaoa.n_gamma + [beta_val] * qaoa.n_beta
            )
            candidate = np.concatenate([prev_angles, new_layer])
            energy = qaoa._eval_energy(candidate)
            if energy < best_energy:
                best_energy = energy
                best_candidate = candidate.copy()

        logger.info(f"Layer grid search done, best energy: {best_energy:.6f}")
        return best_candidate
