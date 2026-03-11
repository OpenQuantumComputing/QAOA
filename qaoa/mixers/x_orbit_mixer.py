from collections import defaultdict

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

from .base_mixer import Mixer
from qaoa.util import GraphHandler


class XOrbit(Mixer):
    """
    X mixer with one independent rotation angle per node orbit.

    Uses the automorphism group of the graph to identify structurally
    equivalent nodes (qubits).  Nodes in the same orbit under the graph's
    automorphism group share a single :math:`\\beta` parameter, implementing
    the orbit-equivariant mixer described in *arXiv:2410.05187*.

    Combined with :class:`~qaoa.problems.MaxCutOrbit` (one :math:`\\gamma`
    per edge orbit), this gives the orbit QAOA ansatz that is equivariant
    under the full symmetry group of the graph.

    Attributes:
        node_orbits (list[list]): List of node groups; each group contains
            all nodes that belong to the same automorphism orbit.
        node_to_orbit (dict): Mapping from a canonical node label to its
            orbit index.

    Args:
        G (nx.Graph): The graph whose node orbits define the parameter
            sharing.  A :class:`~qaoa.util.GraphHandler` is used internally
            to obtain the same canonical node ordering as the problem circuit.
    """

    def __init__(self, G: nx.Graph) -> None:
        """
        Initialises the XOrbit mixer.

        Args:
            G (nx.Graph): The input graph used to compute node orbits.
        """
        super().__init__()
        graph_handler = GraphHandler(G)
        self._canonical_G = graph_handler.G
        self._compute_node_orbits()

    # ------------------------------------------------------------------
    # Orbit computation
    # ------------------------------------------------------------------

    def _compute_node_orbits(self) -> None:
        """
        Compute node orbits of ``self._canonical_G`` under its automorphism
        group.

        Sets:
            self.node_orbits: list of node-lists, one list per orbit.
            self.node_to_orbit: dict mapping each canonical node label to
                its orbit index.
        """
        G = self._canonical_G
        # Use sorted node list so indices are stable
        nodes: list = sorted(G.nodes())
        n_nodes = len(nodes)

        # --- Union-Find -----------------------------------------------
        parent = list(range(n_nodes))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        node_to_idx: dict[int, int] = {v: i for i, v in enumerate(nodes)}

        # --- Enumerate automorphisms and union equivalent nodes --------
        # Note: for graphs with large automorphism groups the enumeration can
        # be expensive.  For typical QAOA problem graphs (tens of nodes) this
        # is fast; for highly symmetric graphs (e.g. complete graphs) the
        # number of automorphisms can be n! and you may want to limit
        # iterations via early termination once all nodes are merged.
        gm = GraphMatcher(G, G)
        for auto in gm.isomorphisms_iter():
            for idx, v in enumerate(nodes):
                mapped_v = auto[v]
                mapped_idx = node_to_idx[mapped_v]
                union(idx, mapped_idx)
            # Early exit: if all nodes are already in one orbit, stop
            if len({find(i) for i in range(n_nodes)}) == 1:
                break

        # --- Group nodes by orbit root ---------------------------------
        orbit_groups: dict[int, list] = defaultdict(list)
        for idx in range(n_nodes):
            orbit_groups[find(idx)].append(nodes[idx])

        self.node_orbits: list[list] = list(orbit_groups.values())

        # Build node → orbit index map
        self.node_to_orbit: dict[int, int] = {}
        for orbit_idx, orbit_nodes in enumerate(self.node_orbits):
            for v in orbit_nodes:
                self.node_to_orbit[v] = orbit_idx

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def get_num_parameters(self) -> int:
        """
        Returns the number of :math:`\\beta` parameters per layer.

        One parameter is used per node orbit of the graph.

        Returns:
            int: Number of node orbits (≥ 1).
        """
        return len(self.node_orbits)

    def create_circuit(self) -> None:
        """
        Constructs the orbit-equivariant X mixer circuit.

        Each qubit (node) :math:`i` receives an RX rotation whose parameter
        is shared with all nodes in the same automorphism orbit.  Parameters
        are named ``x_beta_orbit_0``, ``x_beta_orbit_1``, … (zero-padded so
        alphabetical ordering matches orbit index order).
        """
        n_orbits = self.get_num_parameters()
        n_digits = len(str(n_orbits - 1)) if n_orbits > 1 else 1
        orbit_params = [
            Parameter(f"x_beta_orbit_{i:0{n_digits}d}") for i in range(n_orbits)
        ]

        # Stable node ordering (sorted integers) matches qubit indices
        nodes: list = sorted(self._canonical_G.nodes())

        q = QuantumRegister(self.N_qubits)
        self.circuit = QuantumCircuit(q)

        for qubit_idx, v in enumerate(nodes):
            orbit_idx = self.node_to_orbit[v]
            self.circuit.rx(-2 * orbit_params[orbit_idx], q[qubit_idx])
