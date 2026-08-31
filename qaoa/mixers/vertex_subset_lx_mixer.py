"""First-order logical-X mixers for graph feasibility constraints."""

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import RXGate

from .base_mixer import Mixer
from qaoa.utils.vertex_subset import canonical_graph, degree_descending_order


class VertexSubsetLXMixer(Mixer):
    r"""Shared neighbor-controlled-RX implementation.

    One ordered product of local terms is emitted. The QAOA driver's
    ``number_trottersteps_mixer`` option can repeat this circuit with ``beta/r``.
    """

    problem_kind = None

    def __init__(self, graph, problem_kind, multi_angle=True, label=None):
        super().__init__(label=label)
        self.problem_kind = problem_kind
        if self.problem_kind not in ("mis", "mvc"):
            raise TypeError("problem_kind must be 'mis' or 'mvc'")
        self.G, self.node_order = canonical_graph(graph)
        self.N_qubits = self.G.number_of_nodes()
        self.N_ancilla_qubits = 0
        self.multi_angle = bool(multi_angle)
        self.vertex_order = degree_descending_order(self.G)
        self.parameter_nodes = tuple(
            self.node_order[target] for target in self.vertex_order
        )

        if self.multi_angle:
            width = max(1, len(str(self.N_qubits - 1)))
            self.mixer_params = [
                Parameter(f"x_beta_{rank:0{width}d}")
                for rank in range(self.N_qubits)
            ]
        else:
            self.mixer_params = [Parameter("x_beta")]

    def get_num_parameters(self):
        return self.N_qubits if self.multi_angle else 1

    def create_circuit(self):
        if self.N_qubits != self.G.number_of_nodes():
            raise ValueError(
                "mixer qubit count must equal the number of graph vertices"
            )

        q = QuantumRegister(self.N_qubits, name="q")
        self.circuit = QuantumCircuit(q)

        for rank, target in enumerate(self.vertex_order):
            controls = sorted(self.G.neighbors(target))
            beta = self.mixer_params[rank] if self.multi_angle else self.mixer_params[0]
            if not controls:
                self.circuit.rx(-2 * beta, q[target])
                continue

            # MIS permits X_i only when all neighbours are 0. MVC permits X_i
            # only when all neighbours are 1. RX(-2 beta) follows the sign
            # convention of qaoa.mixers.X.
            control_state = 0 if self.problem_kind == "mis" else None
            controlled_rx = RXGate(-2 * beta).control(
                len(controls), ctrl_state=control_state
            )
            self.circuit.append(
                controlled_rx,
                [q[control] for control in controls] + [q[target]],
            )
