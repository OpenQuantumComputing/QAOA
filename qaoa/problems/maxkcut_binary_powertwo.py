import networkx as nx
import numpy as np
import itertools
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import PhaseGate

from qiskit.circuit.library import PauliEvolutionGate

from .maxkcut_binary_PauliBasis import getPauliOperator

from .graph_problem import GraphProblem


class MaxKCutBinaryPowerOfTwo(GraphProblem):
    def __init__(
        self,
        G: nx.Graph,
        k_cuts: int,
        method: str = "Diffusion",
        fix_one_node: bool = False,  # this fixes the last node to color 1, i.e., one qubit gets removed
    ) -> None:
        MaxKCutBinaryPowerOfTwo.validate_parameters(k_cuts, method)

        self.k_cuts = k_cuts
        self.method = method

        N_qubits_per_node = int(np.ceil(np.log2(self.k_cuts)))
        super().__init__(G, N_qubits_per_node, fix_one_node)

        if self.method == "PauliBasis":
            self.op, self.ophalf = getPauliOperator(self.k_cuts, "all")

        self.construct_colors()

    @staticmethod
    def validate_parameters(k, method) -> None:
        ### 1) k_cuts must be a power of 2
        if not GraphProblem.is_power_of_two(k):
            raise ValueError("k_cuts must be a power of two")

        ### 2) k_cuts needs to be between 2 and 8
        if (k < 2) or (k > 8):
            raise ValueError(
                "k_cuts must be 2 or more, and is not implemented for k_cuts > 8"
            )

        ### 3) method
        valid_methods = ["PauliBasis", "Diffusion"]
        if method not in valid_methods:
            raise ValueError("method must be in " + str(valid_methods))

    def construct_colors(self):
        if self.k_cuts == 2:
            self.colors = {"color1": ["0"], "color2": ["1"]}
        elif self.k_cuts == 4:
            self.colors = {
                "color1": ["00"],
                "color2": ["01"],
                "color3": ["10"],
                "color4": ["11"],
            }
        elif self.k_cuts == 8:
            self.colors = {
                "color1": ["000"],
                "color2": ["001"],
                "color3": ["010"],
                "color4": ["011"],
                "color5": ["100"],
                "color6": ["101"],
                "color7": ["110"],
                "color8": ["111"],
            }
        # Create a dictionary to map each index to its corresponding set
        self.bitstring_to_color = {}
        for key, indices in self.colors.items():
            for index in indices:
                self.bitstring_to_color[index] = key

    def create_edge_circuit(self, theta):
        qc = QuantumCircuit(2 * self.N_qubits_per_node)
        if self.method == "PauliBasis":
            qc.append(PauliEvolutionGate(self.op, time=theta), qc.qubits)
        else:
            for k in range(self.N_qubits_per_node):
                qc.cx(k, self.N_qubits_per_node + k)
                qc.x(self.N_qubits_per_node + k)
            # C^{n-1}Phase
            if self.N_qubits_per_node == 1:
                phase_gate = PhaseGate(-theta)
            else:
                phase_gate = PhaseGate(-theta).control(self.N_qubits_per_node - 1)
            qc.append(
                phase_gate,
                [
                    self.N_qubits_per_node - 1 + ind
                    for ind in range(1, self.N_qubits_per_node + 1)
                ],
            )
            for k in reversed(range(self.N_qubits_per_node)):
                qc.x(self.N_qubits_per_node + k)
                qc.cx(k, self.N_qubits_per_node + k)
        return qc

    def create_edge_circuit_fixed_node(self, theta):
        qc = QuantumCircuit(self.N_qubits_per_node)
        if self.method == "PauliBasis":
            qc.append(PauliEvolutionGate(self.ophalf, time=-theta), qc.qubits)
        else:
            qc.x(qc.qubits)
            # C^{n-1}Phase
            if self.N_qubits_per_node == 1:
                phase_gate = PhaseGate(-theta)
            else:
                phase_gate = PhaseGate(-theta).control(self.N_qubits_per_node - 1)
            qc.append(phase_gate, qc.qubits)
            qc.x(qc.qubits)
        return qc
