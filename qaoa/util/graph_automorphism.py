"""
Graph automorphism utilities for orbit-based QAOA parameter reduction.

Based on: https://arxiv.org/pdf/2410.05187
Uses NetworkX's graph isomorphism functionality.
"""

import networkx as nx
from typing import List, Dict, Tuple
from collections import defaultdict


def compute_edge_orbits(G: nx.Graph) -> Dict[int, List[Tuple[int, int]]]:
    """
    Compute edge orbits using graph automorphisms.

    Edges in the same orbit are equivalent under the graph's symmetry group
    and should share the same QAOA parameter.

    Args:
        G: NetworkX graph

    Returns:
        Dictionary mapping orbit_id -> list of edges in that orbit

    Algorithm:
        1. Use NetworkX's VF2 algorithm to find automorphisms
        2. Group edges that map to each other under any automorphism
        3. Return orbit partition
    """
    # Find all automorphisms (graph isomorphisms to itself)
    automorphisms = list(nx.vf2pp_all_isomorphisms(G, G))

    edges = list(G.edges())

    if len(automorphisms) == 0:
        # No automorphisms found, each edge is its own orbit
        return {i: [edge] for i, edge in enumerate(edges)}

    # Build equivalence classes of edges under automorphisms
    edge_set = set(edges)
    edge_to_orbit = {}
    orbit_id = 0

    for edge in edges:
        if edge in edge_to_orbit:
            continue

        # Find all edges equivalent to this one under the automorphism group
        orbit = set()
        orbit.add(edge)

        for auto in automorphisms:
            # Map edge under this automorphism
            mapped_edge = (auto[edge[0]], auto[edge[1]])
            # Canonicalize to the direction present in edge list
            if mapped_edge in edge_set:
                orbit.add(mapped_edge)
            elif (mapped_edge[1], mapped_edge[0]) in edge_set:
                orbit.add((mapped_edge[1], mapped_edge[0]))

        # Assign the same orbit ID to all edges in this orbit
        for e in orbit:
            edge_to_orbit[e] = orbit_id

        orbit_id += 1

    # Group edges by orbit
    orbits = defaultdict(list)
    for edge in edges:
        orbits[edge_to_orbit[edge]].append(edge)

    return dict(orbits)


def get_edge_to_orbit_map(G: nx.Graph) -> Dict[Tuple[int, int], int]:
    """
    Get mapping from each edge to its orbit ID.

    Args:
        G: NetworkX graph

    Returns:
        Dictionary mapping edge -> orbit_id
    """
    orbits = compute_edge_orbits(G)
    edge_to_orbit = {}

    for orbit_id, edges in orbits.items():
        for edge in edges:
            edge_to_orbit[edge] = orbit_id
            edge_to_orbit[(edge[1], edge[0])] = orbit_id  # Undirected

    return edge_to_orbit


def print_orbit_structure(G: nx.Graph):
    """
    Print human-readable orbit structure for debugging.

    Args:
        G: NetworkX graph
    """
    orbits = compute_edge_orbits(G)
    print(f"Graph has {len(orbits)} edge orbits:")
    for orbit_id, edges in orbits.items():
        print(f"  Orbit {orbit_id}: {edges}")
    print(f"Parameter reduction: {len(G.edges())} -> {len(orbits)} parameters")
