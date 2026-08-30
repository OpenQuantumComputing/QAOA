"""Penalty-free phase separators for constrained vertex-subset problems."""

import numpy as np

from .base_problem import ObjectiveSense
from .qubo_problem import QUBO
from qaoa.utils.vertex_subset import (
    canonical_graph,
    is_independent_set,
    is_vertex_cover,
    resolve_node_weights,
    validate_bitstring,
)

class MIS_MVC_Problem(QUBO):
    """Shared linear objective phase for MIS and MVC.

    This deliberately reuses :class:`QUBO`'s existing signed phase-separator
    implementation.  The QUBO is purely linear (``Q = 0``), while the graph
    subclass adds the appropriate feasibility predicate.  Consequently,
    :class:`ObjectiveSense` is the single source of truth for the phase sign.
    """

    def __init__(self, graph, problem_kind, weights=None, objective_sense=None):
        self.problem_kind = problem_kind
        if self.problem_kind not in ("mis", "mvc"):
            raise TypeError("problem_kind must be 'mis' or 'mvc'")
        if self.problem_kind == "mis":
            default_sense = ObjectiveSense.MAXIMIZE
        elif self.problem_kind == "mvc":
            default_sense = ObjectiveSense.MINIMIZE
        else:  # pragma: no cover - protects future subclasses
            raise TypeError("problem_kind must be 'mis' or 'mvc'")

        self.G, self.node_order = canonical_graph(graph)
        self.weights = resolve_node_weights(graph, self.node_order, weights)
        if objective_sense is None:
            objective_sense = default_sense
        super().__init__(
            Q=np.zeros((self.G.number_of_nodes(), self.G.number_of_nodes())),
            c=self.weights.copy(),
            b=0.0,
            objective_sense=objective_sense,
        )

    def objective_value(self, string):
        validate_bitstring(string, self.N_qubits)
        return float(super().objective_value(string))

    def isFeasible(self, string):
        if self.problem_kind == "mis":
            return is_independent_set(self.G, string)
        return is_vertex_cover(self.G, string)

    def selected_nodes(self, string):
        """Decode a q0-first bitstring to original graph labels."""

        validate_bitstring(string, self.N_qubits)
        return [node for bit, node in zip(string, self.node_order) if bit == "1"]
