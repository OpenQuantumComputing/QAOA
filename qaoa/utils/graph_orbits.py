"""Automorphism-orbit helpers for graph-based QAOA components."""

from collections import defaultdict

from networkx.algorithms.isomorphism import GraphMatcher


def _orbit_partition(items, merge_fn):
    """Return orbit groups and an item-to-orbit lookup in ``items`` order."""

    n_items = len(items)
    parent = list(range(n_items))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_left] = root_right

    for left, right in merge_fn():
        union(left, right)
        if len({find(index) for index in range(n_items)}) == 1:
            break

    groups_by_root = defaultdict(list)
    for index, item in enumerate(items):
        groups_by_root[find(index)].append(item)

    groups = list(groups_by_root.values())
    item_to_orbit = {}
    for orbit_index, group in enumerate(groups):
        for item in group:
            item_to_orbit[item] = orbit_index
    return groups, item_to_orbit


def compute_node_orbits(graph, nodes=None):
    """Compute node orbits under ``Aut(graph)``."""

    if nodes is None:
        nodes = tuple(sorted(graph.nodes()))
    else:
        nodes = tuple(nodes)
    node_to_index = {node: index for index, node in enumerate(nodes)}
    matcher = GraphMatcher(graph, graph)

    def merge_pairs():
        for automorphism in matcher.isomorphisms_iter():
            for index, node in enumerate(nodes):
                yield index, node_to_index[automorphism[node]]

    return _orbit_partition(nodes, merge_pairs)


def compute_edge_orbits(graph, edges=None):
    """Compute edge orbits under ``Aut(graph)``."""

    if edges is None:
        edges = tuple(graph.edges())
    else:
        edges = tuple(edges)

    edge_to_index = {}
    for index, edge in enumerate(edges):
        u, v = edge
        edge_to_index[(u, v)] = index
        edge_to_index[(v, u)] = index

    matcher = GraphMatcher(graph, graph)

    def merge_pairs():
        for automorphism in matcher.isomorphisms_iter():
            for index, (u, v) in enumerate(edges):
                yield index, edge_to_index[(automorphism[u], automorphism[v])]

    edge_orbits, edge_to_orbit = _orbit_partition(edges, merge_pairs)
    for u, v in edges:
        edge_to_orbit[(v, u)] = edge_to_orbit[(u, v)]
    return edge_orbits, edge_to_orbit
