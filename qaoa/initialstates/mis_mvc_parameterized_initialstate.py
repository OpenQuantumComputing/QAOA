"""Variationally-parameterized LX initial state for MIS and MVC."""

import math

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import RXGate

from .base_initialstate import InitialState
from qaoa.utils.vertex_subset import (
    canonical_graph,
    degree_buckets,
    degree_descending_order,
)


class MIS_MVC_ParameterizedInitialState(InitialState):
    r"""LX initial state with variational angle parameters.

    Applies the same degree-ordered controlled-RX construction as
    :class:`MIS_MVC_InitialState`, but exposes one or more of the rotation
    angles as Qiskit ``Parameter`` objects so they can be optimised as part of
    the QAOA variational ansatz.

    The ``grouping`` argument selects how many parameters are used:

    * ``"uniform"`` (default) – **one** shared angle θ for all vertices.
      This is the most scalable choice: a single extra variational parameter
      that is independent of system size.
    * ``"degree"`` – one angle per unique degree value.  Vertices of the same
      degree share a parameter, so the parameter count scales with the number
      of distinct degrees rather than N.
    * ``"per_vertex"`` – one independent angle per vertex (maximum
      expressivity; only practical for small graphs).

    Args:
        graph: An undirected ``networkx.Graph``.
        problem_kind (str): ``"mis"`` or ``"mvc"``.
        grouping (str): ``"uniform"``, ``"degree"``, or ``"per_vertex"``.
        phase_correct (bool): Apply phase-correction gates after the sweep
            (same semantics as in :class:`MIS_MVC_InitialState`).
        label (str | None): Optional circuit label.

    Attributes:
        param_names (list[str]): Names of the Qiskit ``Parameter`` objects in
            the order they appear in the flat angle array.
        vertex_to_param (list[int]): Maps qubit index to its parameter index.
    """

    def __init__(
        self,
        graph,
        problem_kind,
        grouping="uniform",
        phase_correct=True,
        label=None,
    ):
        super().__init__(label=label)
        if problem_kind not in ("mis", "mvc"):
            raise TypeError("problem_kind must be 'mis' or 'mvc'")
        if grouping not in ("uniform", "degree", "per_vertex"):
            raise ValueError("grouping must be 'uniform', 'degree', or 'per_vertex'")

        self.problem_kind = problem_kind
        self.grouping = grouping
        self.phase_correct = bool(phase_correct)

        self.G, self.node_order = canonical_graph(graph)
        self.N_qubits = self.G.number_of_nodes()
        self.N_ancilla_qubits = 0
        self.vertex_order = degree_descending_order(self.G)

        # Build vertex → parameter-index mapping
        if grouping == "uniform":
            self.vertex_to_param = [0] * self.N_qubits
            n_params = 1
        elif grouping == "degree":
            buckets = degree_buckets(self.G)
            vertex_to_bucket = {}
            for bucket_idx, bucket in enumerate(buckets):
                for v in bucket:
                    vertex_to_bucket[v] = bucket_idx
            self.vertex_to_param = [vertex_to_bucket[v] for v in range(self.N_qubits)]
            n_params = len(buckets)
        else:  # per_vertex
            self.vertex_to_param = list(range(self.N_qubits))
            n_params = self.N_qubits

        self._n_params = n_params
        n_digits = len(str(n_params))
        self.param_names = [
            "init_{:0{}d}".format(i, n_digits) for i in range(n_params)
        ]

    def setNumQubits(self, n):
        if n != self.G.number_of_nodes():
            raise ValueError(
                "initial-state qubit count must equal the number of graph vertices"
            )
        self.N_qubits = n

    def get_num_parameters(self):
        return self._n_params

    def create_circuit(self):
        # Reuse stable Parameter objects so repeated calls produce the same
        # parameter instances (important if callers cache references).
        if not hasattr(self, "_params"):
            self._params = [Parameter(name) for name in self.param_names]

        q = QuantumRegister(self.N_qubits, name="q")
        self.circuit = QuantumCircuit(q)

        if self.problem_kind == "mvc":
            self.circuit.x(q)

        for target in self.vertex_order:
            controls = sorted(self.G.neighbors(target))
            theta = self._params[self.vertex_to_param[target]]
            if not controls:
                self.circuit.rx(-2 * theta, q[target])
                continue

            # MIS: fire when all controls are |0⟩ (none selected yet).
            # MVC: fire when all controls are |1⟩ (all neighbours in cover).
            control_state = 0 if self.problem_kind == "mis" else (1 << len(controls)) - 1
            controlled_rx = RXGate(-2 * theta).control(
                len(controls), ctrl_state=control_state
            )
            self.circuit.append(
                controlled_rx,
                [q[control] for control in controls] + [q[target]],
            )

        if self.phase_correct:
            if self.problem_kind == "mis":
                self.circuit.sdg(q)
            else:
                self.circuit.s(q)
