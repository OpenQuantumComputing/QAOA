import networkx as nx
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

from .base_mixer import Mixer
from qaoa.utils import GraphHandler
from qaoa.utils.graph_orbits import compute_node_orbits


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
            sharing.  A :class:`~.utils.GraphHandler` is used internally
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
        self.node_orbits, self.node_to_orbit = compute_node_orbits(self._canonical_G)

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
