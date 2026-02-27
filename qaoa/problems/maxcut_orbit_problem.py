from collections import defaultdict

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
from qiskit import AncillaRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

from .maxcut_problem import MaxCut


class MaxCutOrbit(MaxCut):
    """
    Orbit-based Max Cut problem.

    Subclass of :class:`MaxCut` that uses the automorphism group of the graph
    to identify structurally equivalent edges.  Edges in the same *orbit*
    (i.e. related by a graph automorphism) share a single :math:`\\gamma`
    parameter, reducing the search space compared to a fully multi-angle
    formulation while retaining more flexibility than the vanilla single-
    :math:`\\gamma` approach.

    The number of :math:`\\gamma` parameters per QAOA layer equals the number
    of distinct edge orbits, accessible via :meth:`get_num_parameters`.

    Attributes:
        edge_orbits (list[list[tuple]]): List of edge groups; each group
            contains all edges that belong to the same automorphism orbit.
        edge_to_orbit (dict): Mapping from an edge (or its reverse) to its
            orbit index.

    Methods:
        get_num_parameters(): Returns the number of edge orbits.
        create_circuit(): Builds the phase-separation circuit with one
            :class:`~qiskit.circuit.Parameter` per edge orbit.
    """

    def __init__(self, G: nx.Graph) -> None:
        """
        Initialises the MaxCutOrbit problem.

        Args:
            G (nx.Graph): The input graph on which Max Cut is defined.
        """
        super().__init__(G)
        self._compute_edge_orbits()

    # ------------------------------------------------------------------
    # Orbit computation
    # ------------------------------------------------------------------

    def _compute_edge_orbits(self) -> None:
        """
        Compute edge orbits of ``self.graph_handler.G`` under its
        automorphism group.

        Sets:
            self.edge_orbits: list of edge-lists, one list per orbit.
            self.edge_to_orbit: dict mapping each (directed) edge to its
                orbit index.
        """
        G = self.graph_handler.G  # relabelled graph used by the circuit
        edges = list(G.edges())
        n_edges = len(edges)

        # --- Union-Find -----------------------------------------------
        parent = list(range(n_edges))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        # Build a fast edge -> index lookup (both orientations)
        edge_to_idx: dict[tuple[int, int], int] = {}
        for idx, (u, v) in enumerate(edges):
            edge_to_idx[(u, v)] = idx
            edge_to_idx[(v, u)] = idx

        # --- Enumerate automorphisms and union equivalent edges --------
        # Note: for graphs with large automorphism groups the enumeration can
        # be expensive.  Early exit when all edges are already merged.
        gm = GraphMatcher(G, G)
        for auto in gm.isomorphisms_iter():
            for idx, (u, v) in enumerate(edges):
                mapped_u = auto[u]
                mapped_v = auto[v]
                mapped_idx = edge_to_idx.get((mapped_u, mapped_v))
                if mapped_idx is not None:
                    union(idx, mapped_idx)
            # Early exit: stop once all edges are in a single orbit
            if len({find(i) for i in range(n_edges)}) == 1:
                break

        # --- Group edges by orbit root ---------------------------------
        orbit_groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for idx in range(n_edges):
            orbit_groups[find(idx)].append(edges[idx])

        self.edge_orbits: list[list[tuple[int, int]]] = list(orbit_groups.values())

        # Build edge -> orbit index map (both orientations)
        self.edge_to_orbit: dict[tuple[int, int], int] = {}
        for orbit_idx, orbit_edges in enumerate(self.edge_orbits):
            for u, v in orbit_edges:
                self.edge_to_orbit[(u, v)] = orbit_idx
                self.edge_to_orbit[(v, u)] = orbit_idx

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def get_num_parameters(self) -> int:
        """
        Returns the number of :math:`\\gamma` parameters per layer.

        One parameter is used per edge orbit of the graph.

        Returns:
            int: Number of edge orbits (≥ 1).
        """
        return len(self.edge_orbits)

    def create_circuit(self) -> None:
        """
        Constructs the phase-separation circuit with one parameter per edge
        orbit.

        Edges in the same automorphism orbit share a single
        :class:`~qiskit.circuit.Parameter`, so the total number of free
        :math:`\\gamma` values equals :meth:`get_num_parameters`.

        Parameters are named ``gamma_orbit_00``, ``gamma_orbit_01``, … (zero-
        padded) so that their alphabetical ordering matches their orbit index.
        """
        n_orbits = self.get_num_parameters()
        n_digits = len(str(n_orbits - 1)) if n_orbits > 1 else 1
        orbit_params = [
            Parameter(f"gamma_orbit_{i:0{n_digits}d}") for i in range(n_orbits)
        ]

        q = QuantumRegister(self.N_qubits)
        a = AncillaRegister(self.N_ancilla_qubits)
        self.circuit = QuantumCircuit(q, a)

        G = self.graph_handler.G
        for _, group in self.graph_handler.parallel_edges.items():
            for i, j in group:
                orbit_idx = self.edge_to_orbit[(i, j)]
                theta = orbit_params[orbit_idx] * G[i][j].get("weight", 1)

                I = i * self.N_qubits_per_node
                J = j * self.N_qubits_per_node

                if self.num_V - self.fix_one_node not in (i, j):
                    qubits_to_map = list(range(I, I + self.N_qubits_per_node)) + list(
                        range(J, J + self.N_qubits_per_node)
                    )
                    ancilla_to_map = (
                        list(range(self.N_ancilla_qubits))
                        if self.N_ancilla_qubits > 0
                        else []
                    )
                    edge_circ = self.create_edge_circuit(theta)
                    self.circuit.append(
                        edge_circ,
                        q[qubits_to_map]
                        + (a[ancilla_to_map] if ancilla_to_map else []),
                    )
                else:
                    min_I = min(I, J)
                    fixed_circ = self.create_edge_circuit_fixed_node(theta)
                    self.circuit.append(
                        fixed_circ,
                        q[list(range(min_I, min_I + self.N_qubits_per_node))],
                    )
