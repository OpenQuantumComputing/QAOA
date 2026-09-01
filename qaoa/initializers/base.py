"""Abstract base class for QAOA parameter initializers."""

from abc import ABC, abstractmethod


class Initializer(ABC):
    """
    Abstract base for QAOA parameter initializers.

    An initializer is responsible for producing one or more candidate angle
    arrays for a given target depth.  :meth:`QAOA.optimize` collects the
    candidates, evaluates them (when needed), selects the best, and passes
    it to the local optimiser.

    Subclasses must implement :meth:`get_candidates`.

    The flat angle array format used throughout this package is::

        [init_0, …, init_{n_init-1},
         gamma_{0,0}, …, gamma_{0,n_gamma-1},
         beta_{0,0},  …, beta_{0,n_beta-1},
         gamma_{1,0}, …
         …]

    where the leading ``n_init`` values belong to the initial state.

    Attributes:
        monotone (bool): When ``True``, :meth:`QAOA.optimize` will enforce a
            strict monotone guarantee after local optimisation: if the depth-p
            result is worse than depth-(p-1), the optimizer falls back to the
            zero-new-layer candidate (which is circuit-equivalent to depth p-1).
            Only :class:`LayerGrid` sets this to ``True``.
    """

    monotone: bool = False

    @abstractmethod
    def get_candidates(self, qaoa, depth: int) -> list:
        """
        Return a list of candidate angle arrays for *depth*.

        Args:
            qaoa: The :class:`~qaoa.QAOA` instance (read-only).
            depth (int): Target circuit depth (>= 1).

        Returns:
            list[np.ndarray]: One or more flat angle arrays, each of length
            ``n_init + depth * (n_gamma + n_beta)``.
        """
