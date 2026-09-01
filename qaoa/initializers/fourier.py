"""
Fourier initializer – Fourier (u, v) parameterisation.

Represents the QAOA angles via *q* Fourier coefficients instead of *p*
layer-by-layer angles.  The mapping is:

    gamma_l = Σ_{k=1}^{q}  u_k * sin((k-½)(l-½) π/p)
    beta_l  = Σ_{k=1}^{q}  v_k * cos((k-½)(l-½) π/p)

for l = 1 … p.  The initializer accepts u and v coefficient arrays of
length *q* (default: zeros) and returns the corresponding flat angle array.

Supports only **standard (vanilla) QAOA** (one gamma and one beta per
layer).  Raises ``ValueError`` for multi-angle ansätze.

No previous depth is required.  No additional circuit evaluations.

Citation: Zhou et al., PRX 10, 021067 (2020).
    https://doi.org/10.1103/PhysRevX.10.021067
"""

import numpy as np

from .base import Initializer


class Fourier(Initializer):
    """
    Fourier-parameterisation initializer.

    Args:
        u (array-like | None): Fourier coefficients for gamma.  Length
            determines *q*.  Defaults to ``[0.0]``.
        v (array-like | None): Fourier coefficients for beta.  Same length
            as *u*.  Defaults to ``[0.0]``.

    Raises:
        ValueError: If the ansatz has more than one gamma or beta parameter
            per layer (multi-angle), since the Fourier parameterisation
            assumes a single scalar per layer.

    Example::

        from qaoa import QAOA, initializers
        import numpy as np

        qaoa = QAOA(problem, mixer, initialstate,
                    initializer=initializers.Fourier(
                        u=[0.1, 0.05],
                        v=[0.2, 0.1],
                    ))
        qaoa.optimize(depth=4)
    """

    def __init__(self, u=None, v=None):
        self.u = np.asarray(u if u is not None else [0.0], dtype=float)
        self.v = np.asarray(v if v is not None else [0.0], dtype=float)
        if len(self.u) != len(self.v):
            raise ValueError("u and v must have the same length.")

    def get_candidates(self, qaoa, depth: int) -> list:
        if qaoa.n_gamma != 1 or qaoa.n_beta != 1:
            raise ValueError(
                "Fourier initializer supports only standard (vanilla) QAOA "
                f"(n_gamma=1, n_beta=1), but got n_gamma={qaoa.n_gamma}, "
                f"n_beta={qaoa.n_beta}.  Use a different initializer for "
                "multi-angle ansätze."
            )

        u, v = self.u, self.v
        q = len(u)
        layers = np.arange(1, depth + 1)
        ks = np.arange(1, q + 1)

        # gamma_l = Σ u_k sin((k-½)(l-½)π/p)
        # beta_l  = Σ v_k cos((k-½)(l-½)π/p)
        arg = np.outer(ks - 0.5, layers - 0.5) * np.pi / depth  # shape (q, p)
        gammas = (u[:, None] * np.sin(arg)).sum(axis=0)  # shape (p,)
        betas = (v[:, None] * np.cos(arg)).sum(axis=0)

        init_part = [0.0] * qaoa.n_init
        layer_part = []
        for l in range(depth):
            layer_part += [gammas[l], betas[l]]

        return [np.array(init_part + layer_part)]
