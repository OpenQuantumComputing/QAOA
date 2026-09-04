"""Shared graph helpers for independent-set and vertex-cover components."""

from collections.abc import Mapping, Sequence
from numbers import Real

import networkx as nx
import numpy as np


def canonical_graph(graph):
    """Return a simple integer-labelled copy and its original node order.

    Qubit ``i`` represents ``node_order[i]``. NetworkX insertion order is used
    deliberately so independently-created phase, mixer, and initial-state
    objects agree on the encoding.
    """

    if not isinstance(graph, nx.Graph) or graph.is_directed():
        raise TypeError("graph must be an undirected networkx.Graph")
    if graph.is_multigraph():
        raise ValueError("multigraphs are not supported")
    if graph.number_of_nodes() == 0:
        raise ValueError("graph must contain at least one node")
    if nx.number_of_selfloops(graph):
        raise ValueError("self-loops are not supported")

    node_order = tuple(graph.nodes())
    node_to_qubit = {node: i for i, node in enumerate(node_order)}

    canonical = nx.Graph()
    for i, node in enumerate(node_order):
        canonical.add_node(i, **dict(graph.nodes[node]))
    for u, v, data in graph.edges(data=True):
        canonical.add_edge(node_to_qubit[u], node_to_qubit[v], **dict(data))

    return canonical, node_order


def degree_descending_order(canonical):
    """Return qubits ordered by decreasing degree, then by qubit index."""

    return tuple(
        sorted(canonical.nodes(), key=lambda node: (-canonical.degree[node], node))
    )


def resolve_node_angles(node_order, angle):
    """Resolve a scalar, node mapping, or q0-first sequence of fixed angles."""

    if isinstance(angle, Mapping):
        missing = [node for node in node_order if node not in angle]
        if missing:
            raise ValueError(f"angle is missing graph nodes: {missing!r}")
        values = [angle[node] for node in node_order]
    elif (
        isinstance(angle, (Sequence, np.ndarray))
        and not isinstance(angle, (str, bytes))
    ):
        if len(angle) != len(node_order):
            raise ValueError(
                f"expected {len(node_order)} angles, received {len(angle)}"
            )
        values = list(angle)
    else:
        values = [angle] * len(node_order)

    for value in values:
        if not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError("angles must be finite real numbers")
    return np.asarray(values, dtype=float)


def resolve_node_weights(graph, node_order, weights=None):
    """Resolve one finite real weight per node.

    When ``weights`` is omitted, the node attribute ``"weight"`` is used and
    defaults to one. Explicit weights may be a mapping keyed by original node
    labels or a sequence in ``node_order``.
    """

    if weights is None:
        values = [graph.nodes[node].get("weight", 1.0) for node in node_order]
    elif isinstance(weights, Mapping):
        missing = [node for node in node_order if node not in weights]
        if missing:
            raise ValueError(f"weights is missing graph nodes: {missing!r}")
        values = [weights[node] for node in node_order]
    elif (
        isinstance(weights, (Sequence, np.ndarray))
        and not isinstance(weights, (str, bytes))
    ):
        if len(weights) != len(node_order):
            raise ValueError(
                f"expected {len(node_order)} node weights, received {len(weights)}"
            )
        values = list(weights)
    else:
        raise TypeError("weights must be a mapping, a sequence, or None")

    for value in values:
        if not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError("node weights must be finite real numbers")
    return np.asarray(values, dtype=float)


def degree_buckets(canonical):
    """Return a list of degree buckets for the canonical graph.

    Each bucket is a list of qubit indices that share the same degree, ordered
    by increasing degree value.  Vertices within a bucket are sorted by qubit
    index for determinism.

    Example::

        G = nx.path_graph(4)  # degrees: 0→1, 1→2, 2→2, 3→1
        canonical, _ = canonical_graph(G)
        degree_buckets(canonical)
        # [[0, 3], [1, 2]]   (degree-1 vertices, then degree-2 vertices)
    """
    from collections import defaultdict

    buckets = defaultdict(list)
    for node in canonical.nodes():
        buckets[canonical.degree[node]].append(node)
    return [sorted(v) for _, v in sorted(buckets.items())]


def mis_mvc_warm_start_angle(graph, grouping="uniform"):
    """Return warm-start angle(s) for a parameterized MIS/MVC initial state.

    Args:
        graph: The problem graph (passed directly, not yet canonicalized).
        grouping (str): One of ``"uniform"``, ``"degree"``, or ``"per_vertex"``.

    Returns:
        float | list[float]: A single angle for ``"uniform"``, or one angle per
        bucket/vertex for the other groupings.  Each angle is
        ``arctan(1 / sqrt(mean_degree_in_bucket))`` (clamped to ``[0, π/2]``),
        falling back to ``π/4`` for isolated vertices (degree 0).
    """
    canonical, _ = canonical_graph(graph)

    def _angle_for_degree(d):
        if d == 0:
            return np.pi / 4
        return np.arctan(1.0 / np.sqrt(d))

    if grouping == "uniform":
        mean_deg = np.mean([canonical.degree[n] for n in canonical.nodes()])
        return _angle_for_degree(mean_deg)

    if grouping == "degree":
        buckets = degree_buckets(canonical)
        return [_angle_for_degree(canonical.degree[bucket[0]]) for bucket in buckets]

    if grouping == "per_vertex":
        return [_angle_for_degree(canonical.degree[n]) for n in canonical.nodes()]

    raise ValueError("grouping must be 'uniform', 'degree', or 'per_vertex'")


def validate_bitstring(string, num_qubits):
    """Validate the public q0-first bitstring convention used by qaoa."""

    if not isinstance(string, str):
        raise TypeError("solution must be a bitstring")
    if len(string) != num_qubits:
        raise ValueError(
            f"expected a string of length {num_qubits}, received {len(string)}"
        )
    if any(bit not in "01" for bit in string):
        raise ValueError("solution must contain only '0' and '1'")


def is_independent_set(canonical, string):
    """Return whether ``string`` selects an independent set."""

    validate_bitstring(string, canonical.number_of_nodes())
    return all(
        not (string[u] == "1" and string[v] == "1")
        for u, v in canonical.edges()
    )


def is_vertex_cover(canonical, string):
    """Return whether ``string`` selects a vertex cover."""

    validate_bitstring(string, canonical.number_of_nodes())
    return all(
        string[u] == "1" or string[v] == "1" for u, v in canonical.edges()
    )
