import networkx as nx
import numpy as np
import itertools
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import PhaseGate

from qiskit.circuit.library import PauliEvolutionGate

from .maxkcut_binary_PauliBasis import getPauliOperator

from .graph_problem import GraphProblem
from .maxkcut_binary_powertwo import MaxKCutBinaryPowerOfTwo


class MaxKCutBinaryFullH(GraphProblem):
    def __init__(
        self,
        G: nx.Graph,
        k_cuts: int,
        color_encoding: str,
        method: str = "Diffusion",
        fix_one_node: bool = False,  # this fixes the last node to color 1, i.e., one qubit gets removed
    ) -> None:
        MaxKCutBinaryFullH.validate_parameters(k_cuts, method, fix_one_node)

        self.k_cuts = k_cuts
        self.color_encoding = color_encoding
        self.method = method

        N_qubits_per_node = int(np.ceil(np.log2(self.k_cuts)))
        super().__init__(G, N_qubits_per_node, fix_one_node)

        if self.method == "PauliBasis":
            self.op, self.ophalf = getPauliOperator(
                self.k_cuts, color_encoding=self.color_encoding
            )
        elif self.method == "PowerOfTwo":
            if self.k_cuts == 3:
                k = 4
            else:
                k = 8
            self.mkcb_pot = MaxKCutBinaryPowerOfTwo(G, k, method="Diffusion")
            self.N_ancilla_qubits = 2
        else:  # if self.method == "Diffusion":
            pass

        self.construct_colors()

    @staticmethod
    def validate_parameters(k, method, fix_one_node) -> None:
        ### 1) k_cuts needs to be 3, 5, 6, or 7
        valid_ks = [3, 5, 6, 7]
        if k not in valid_ks:
            raise ValueError("k_cuts must be in " + str(valid_ks))

        ### 2) method
        valid_methods = ["PauliBasis", "PowerOfTwo", "Diffusion"]
        if method not in valid_methods:
            raise ValueError("method must be in " + str(valid_methods))

        if method == "PowerOfTwo" and fix_one_node:
            raise ValueError(
                'For the PowerOfTwo method it "fix_one_node" is not implemented. Use "PauliBasis" or "Diffusion" instead.'
            )

    def construct_colors(self):
        if self.k_cuts == 3:
            if self.color_encoding == "LessThanK":
                self.colors = {
                    "color1": ["00"],
                    "color2": ["01"],
                    "color3": ["10", "11"],
                }
            else:
                raise ValueError("invalid or unspecified color_encoding")
        elif self.k_cuts == 5:
            if self.color_encoding == "LessThanK":
                self.colors = {
                    "color1": ["000"],
                    "color2": ["001"],
                    "color3": ["010"],
                    "color4": ["011"],
                    "color5": ["100", "101", "110", "111"],
                }
            elif self.color_encoding == "max_balanced":
                self.colors = {
                    "color1": ["000", "001"],
                    "color2": ["010"],
                    "color3": ["011"],
                    "color4": ["100", "101"],
                    "color5": ["110", "111"],
                }
            else:
                raise ValueError("invalid or unspecified color_encoding")
        elif self.k_cuts == 6:
            if self.color_encoding == "LessThanK":
                self.colors = {
                    "color1": ["000"],
                    "color2": ["001"],
                    "color3": ["010"],
                    "color4": ["011"],
                    "color5": ["100"],
                    "color6": ["101", "110", "111"],
                }
            elif self.color_encoding in ["max_balanced"]:
                self.colors = {
                    "color1": ["000", "001"],
                    "color2": ["010"],
                    "color3": ["011"],
                    "color4": ["100", "101"],
                    "color5": ["110"],
                    "color6": ["111"],
                }
            else:
                raise ValueError("invalid or unspecified color_encoding")
        else:  # if self.k_cuts == 7:
            if self.color_encoding == "LessThanK":
                self.colors = {
                    "color1": ["000"],
                    "color2": ["001"],
                    "color3": ["010"],
                    "color4": ["011"],
                    "color5": ["100"],
                    "color6": ["101"],
                    "color7": ["110", "111"],
                }
            else:
                raise ValueError("invalid or unspecified color_encoding")
        # Create a dictionary to map each index to its corresponding set
        self.bitstring_to_color = {}
        for key, indices in self.colors.items():
            for index in indices:
                self.bitstring_to_color[index] = key

    def apply_N(self, circuit, binary_str1, binary_str2):
        # Apply X-gates based on the first binary string
        for i, bit in enumerate(binary_str1):
            if bit == "0":
                circuit.x(i)

        # Apply X-gates based on the second binary string
        for i, bit in enumerate(binary_str2):
            if bit == "0":
                circuit.x(self.N_qubits_per_node + i)

        return circuit

    def add_equalize_color(self, qc, bs1, bs2, theta):
        qc = self.apply_N(qc, bs1, bs2)
        qc.barrier()
        qc.mcx(
            [qc.qubits[i] for i in range(0, self.N_qubits_per_node)],
            [qc.qubits[2 * self.N_qubits_per_node]],
        )
        qc.mcx(
            [
                qc.qubits[self.N_qubits_per_node + i]
                for i in range(0, self.N_qubits_per_node)
            ],
            [qc.qubits[2 * self.N_qubits_per_node + 1]],
        )
        # C^{n-1}Phase
        phase_gate = PhaseGate(-theta).control(2)
        qc.append(
            phase_gate,
            [
                2 * self.N_qubits_per_node + 1,
                2 * self.N_qubits_per_node,
                2 * self.N_qubits_per_node - 1,
            ],
        )
        qc.mcx(
            [
                qc.qubits[self.N_qubits_per_node + i]
                for i in range(0, self.N_qubits_per_node)
            ],
            [qc.qubits[2 * self.N_qubits_per_node + 1]],
        )
        qc.mcx(
            [qc.qubits[i] for i in range(0, self.N_qubits_per_node)],
            [qc.qubits[2 * self.N_qubits_per_node]],
        )
        qc.barrier()
        qc = self.apply_N(qc, bs1, bs2)

        return qc

    def create_edge_circuit(self, theta):
        q = QuantumRegister(2 * self.N_qubits_per_node)
        if self.method == "PauliBasis":
            qc = QuantumCircuit(q)
            qc.append(PauliEvolutionGate(self.op, time=theta), qc.qubits)
        elif self.method == "PowerOfTwo":
            a = AncillaRegister(self.N_ancilla_qubits)
            qc = QuantumCircuit(q, a)

            qubits_to_map = list(range(2 * self.N_qubits_per_node))
            ancilla_to_map = []
            qc.append(
                self.mkcb_pot.create_edge_circuit(theta),
                q[qubits_to_map] + (a[ancilla_to_map] if ancilla_to_map else []),
            )

            for _, bitstrings in self.colors.items():
                if len(bitstrings) > 1:
                    pairs = list(itertools.combinations(bitstrings, 2))
                    for bs1, bs2 in pairs:
                        qc = self.add_equalize_color(qc, bs1, bs2, theta)
                        qc = self.add_equalize_color(qc, bs2, bs1, theta)

            # target_qubits = [q[i] for i in range(2*self.N_qubits_per_node)] + [
            #    a[i] for i in range(2)
            # ]  # Map k qubits and 2 ancillas
            # parameterized_circuit.append(small_circuit.to_instruction(), target_qubits)

        else:  # if self.method == "Diffusion":
            qc = QuantumCircuit(q)
            if self.k_cuts == 3:
                phase_gate = PhaseGate(-theta).control(1)
                qc.append(phase_gate, [0, 2])
                qc.cx(1, 3)
                qc.x([0, 2, 3])
                phase_gate = PhaseGate(-theta).control(2)
                qc.append(phase_gate, [0, 2, 3])
                qc.x([0, 2, 3])
                qc.cx(1, 3)
            elif self.k_cuts == 5:
                if self.color_encoding == "max_balanced":
                    qc.cx(0, 3)
                    qc.cx(1, 4)
                    qc.cx(2, 5)
                    qc.x([3, 4, 5])
                    phase_gate = PhaseGate(-theta).control(2)
                    qc.append(phase_gate, [3, 4, 5])
                    qc.x([3, 4, 5])
                    qc.cx(2, 5)
                    qc.cx(1, 4)
                    qc.cx(5, 2, ctrl_state=0)
                    qc.x([1, 2, 3, 4])
                    phase_gate = PhaseGate(-theta).control(3)
                    qc.append(phase_gate, [1, 2, 3, 4])
                    qc.x([1, 2, 3, 4])
                    qc.cx(5, 2, ctrl_state=0)
                    qc.cx(0, 3)
                    qc.cx(2, 5)
                    phase_gate = PhaseGate(-theta).control(4)
                    qc.append(phase_gate, [0, 1, 3, 4, 5])
                    qc.cx(2, 5)
                else:  # self.color_encoding=="LessThanK":
                    phase_gate = PhaseGate(-theta).control(1)
                    qc.append(phase_gate, [0, 3])
                    qc.cx(1, 4)
                    qc.cx(2, 5)
                    qc.x([0, 3, 4, 5])
                    phase_gate = PhaseGate(-theta).control(3)
                    qc.append(phase_gate, [0, 3, 4, 5])
                    qc.x([0, 3, 4, 5])
                    qc.cx(2, 5)
                    qc.cx(1, 4)
            elif self.k_cuts == 6:
                if self.color_encoding == "max_balanced":
                    qc.cx(0, 3)
                    qc.cx(1, 4)
                    qc.cx(2, 5)
                    qc.x([3, 4, 5])
                    phase_gate = PhaseGate(-theta).control(2)
                    qc.append(phase_gate, [3, 4, 5])
                    qc.x([3, 4, 5])
                    qc.cx(2, 5)
                    qc.cx(1, 4)
                    qc.cx(5, 2, ctrl_state=0)
                    qc.x([1, 2, 3, 4])
                    phase_gate = PhaseGate(-theta).control(3)
                    qc.append(phase_gate, [1, 2, 3, 4])
                    qc.x([1, 2, 3, 4])
                    qc.cx(5, 2, ctrl_state=0)
                    qc.cx(0, 3)
                else:  # self.color_encoding=="LessThanK":
                    qc.cx(0, 3)
                    qc.cx(1, 4)
                    qc.cx(2, 5)
                    qc.x([3, 4, 5])
                    phase_gate = PhaseGate(-theta).control(2)
                    qc.append(phase_gate, [3, 4, 5])
                    qc.x([3, 4, 5])
                    qc.cx(2, 5)
                    qc.cx(1, 4)
                    qc.cx(0, 3)
                    qc.ccx(2, 4, 5)
                    phase_gate = PhaseGate(-theta).control(3)
                    qc.append(phase_gate, [0, 1, 3, 5])
                    qc.ccx(2, 4, 5)
                    qc.x(1)
                    phase_gate = PhaseGate(-theta).control(4)
                    qc.append(phase_gate, [0, 1, 2, 3, 4])
                    qc.x(1)
            elif self.k_cuts == 7:
                qc.cx(0, 3)
                qc.cx(1, 4)
                qc.cx(2, 5)
                qc.x([3, 4, 5])
                phase_gate = PhaseGate(-theta).control(2)
                qc.append(phase_gate, [3, 4, 5])
                qc.x(5)
                phase_gate = PhaseGate(-theta).control(4)
                qc.append(phase_gate, [0, 1, 3, 4, 5])
                qc.x([3, 4])
                qc.cx(2, 5)
                qc.cx(1, 4)
                qc.cx(0, 3)

        return qc

    def create_edge_circuit_fixed_node(self, theta):
        if self.method == "PauliBasis":
            qc = QuantumCircuit(self.N_qubits_per_node)
            qc.append(PauliEvolutionGate(self.ophalf, time=theta), qc.qubits)
        else:
            qc = self.mkcb_pot.create_edge_circuit_fixed_node(theta)
        return qc
