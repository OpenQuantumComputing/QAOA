"""Degree-ordered logical-X initial states for MIS and MVC."""

import math

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RXGate

from .base_initialstate import InitialState
from qaoa.utils.vertex_subset import (
    canonical_graph,
    degree_descending_order,
    resolve_node_angles,
)


class MIS_MVC_InitialState(InitialState):
    r"""Prepare a biased feasible state with one fixed-angle LX sweep.

    The local terms are applied in decreasing-degree order. ``angle`` may be a
    scalar, a mapping keyed by original graph nodes, or a sequence in the
    public q0-first ``node_order``. The circuit is unitary and ancilla-free, so
    it can also be used as the state preparation inside a Grover mixer.
    """

    def __init__(self, graph, problem_kind, angle=math.pi / 4, phase_correct=True, label=None):
        super().__init__(label=label)
        self.problem_kind = problem_kind
        if self.problem_kind not in ("mis", "mvc"):
            raise TypeError("problem_kind must be 'mis' or 'mvc'")
        self.G, self.node_order = canonical_graph(graph)
        self.N_qubits = self.G.number_of_nodes()
        self.N_ancilla_qubits = 0
        self.vertex_order = degree_descending_order(self.G)
        self.angles = resolve_node_angles(self.node_order, angle)
        self.phase_correct = bool(phase_correct)

    def setNumQubits(self, n):
        if n != self.G.number_of_nodes():
            raise ValueError(
                "initial-state qubit count must equal the number of graph vertices"
            )
        self.N_qubits = n

    def create_circuit(self):
        q = QuantumRegister(self.N_qubits, name="q")
        self.circuit = QuantumCircuit(q)

        # Empty independent set and full vertex cover are feasible seeds.
        if self.problem_kind == "mvc":
            self.circuit.x(q)

        for target in self.vertex_order:
            controls = sorted(self.G.neighbors(target))
            theta = self.angles[target]
            if not controls:
                self.circuit.rx(-2 * theta, q[target])
                continue

            control_state = 0 if self.problem_kind == "mis" else None
            controlled_rx = RXGate(-2 * theta).control(
                len(controls), ctrl_state=control_state
            )
            self.circuit.append(
                controlled_rx,
                [q[control] for control in controls] + [q[target]],
            )

        # RX contributes i for every performed transition. Correcting those
        # phases leaves sampling unchanged and improves overlap with the
        # positive-amplitude uniform feasible state.
        if self.phase_correct:
            if self.problem_kind == "mis":
                self.circuit.sdg(q)
            else:
                self.circuit.s(q)
