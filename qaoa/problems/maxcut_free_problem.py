from qiskit import AncillaRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

import networkx as nx

from .maxcut_problem import MaxCut


class MaxCutFree(MaxCut):
    """
    Free (multi-angle) Max Cut problem with one :math:`\\gamma` parameter
    per edge.

    Subclass of :class:`MaxCut` that assigns an independent
    :class:`~qiskit.circuit.Parameter` to every edge in the graph.
    Combined with an :class:`~qaoa.mixers.XMultiAngle` mixer (one
    :math:`\\beta` per node), this gives the *free* QAOA ansatz: the
    maximum number of distinct parameters allowed by the circuit structure
    without imposing any symmetry.

    Attributes:
        graph_handler: Canonical graph handler (inherited from
            :class:`~qaoa.problems.GraphProblem`).

    Methods:
        get_num_parameters(): Returns the number of edges.
        create_circuit(): Builds the phase-separation circuit with one
            parameter per edge.
    """

    def get_num_parameters(self) -> int:
        """
        Returns the number of :math:`\\gamma` parameters per layer.

        One parameter is used per edge of the (canonical) graph.

        Returns:
            int: Number of edges.
        """
        return self.graph_handler.num_edges

    def create_circuit(self) -> None:
        """
        Constructs the phase-separation circuit with one parameter per edge.

        Each edge :math:`(i,j)` in the graph receives its own independent
        :class:`~qiskit.circuit.Parameter` named ``gamma_edge_00``,
        ``gamma_edge_01``, … (zero-padded so alphabetical ordering matches
        edge index order).
        """
        G = self.graph_handler.G
        # List edges in a stable order so parameter names are reproducible
        edges = list(G.edges())
        n_edges = len(edges)
        n_digits = len(str(n_edges - 1)) if n_edges > 1 else 1
        edge_params = [
            Parameter(f"gamma_edge_{i:0{n_digits}d}") for i in range(n_edges)
        ]

        # Build a fast edge → parameter-index lookup (both orientations)
        edge_to_param_idx: dict[tuple[int, int], int] = {}
        for idx, (u, v) in enumerate(edges):
            edge_to_param_idx[(u, v)] = idx
            edge_to_param_idx[(v, u)] = idx

        q = QuantumRegister(self.N_qubits)
        a = AncillaRegister(self.N_ancilla_qubits)
        self.circuit = QuantumCircuit(q, a)

        for _, group in self.graph_handler.parallel_edges.items():
            for i, j in group:
                param_idx = edge_to_param_idx[(i, j)]
                theta = edge_params[param_idx] * G[i][j].get("weight", 1)

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
