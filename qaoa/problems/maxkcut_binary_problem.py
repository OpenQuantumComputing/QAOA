import networkx as nx
import numpy as np
import itertools
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import PhaseGate

from qiskit.circuit.library import PauliEvolutionGate

from .maxkcut_binary_direct import getDirectPauliOpertor

from .base_problem import Problem
from qaoa.util import *


class MaxKCutBinaryOptions:
    def __init__(self):
        self.option_table = {}
        self.option_table[2] = {
            "color_encoding": [
                "all",
            ]
        }
        self.option_table[4] = {
            "color_encoding": [
                "all",
            ]
        }
        self.option_table[8] = {
            "color_encoding": [
                "all",
            ]
        }

        self.option_table[3] = {
            "color_encoding": [
                "LessThanK",
            ]
        }
        self.option_table[7] = {
            "color_encoding": [
                "LessThanK",
            ]
        }

        self.option_table[5] = {"color_encoding": ["LessThanK", "max_balanced"]}
        # for k=6, max_balanced == Dicke1_2
        self.option_table[6] = {"color_encoding": ["LessThanK", "max_balanced"]}


class MaxKCutBinary(Problem):
    def __init__(
        self,
        G: nx.Graph,
        k_cuts: int,
        color_encoding: str = "all",
        force_power_of_two_Hamiltonian: bool = False,
        direct: bool = False,
        fix_one_node: bool = False,
    ) -> None:
        super().__init__()

        # fixes the last node to "color1"
        self.fix_one_node = fix_one_node

        self.num_V = G.number_of_nodes()
        self.k_cuts = k_cuts
        self.k_bits = int(np.ceil(np.log2(self.k_cuts)))
        self.N_qubits = (self.num_V - self.fix_one_node) * self.k_bits
        self.cost_param = Parameter("x_gamma")
        self.direct = direct
        self.color_encoding = color_encoding
        self.force_power_of_two_Hamiltonian = force_power_of_two_Hamiltonian

        self.validate_parameters()

        # ensure graph has labels 0, 1, ..., num_V-1
        self.graph_handler = GraphHandler(G)

        if self.direct:
            if self.force_power_of_two_Hamiltonian:
                if self.k_cuts == 2:
                    k = 2
                elif self.k_cuts in [3, 4]:
                    k = 4
                elif self.k_cuts in [5, 6, 7, 8]:
                    k = 8
            else:
                k = self.k_cuts
            self.op, self.ophalf = getDirectPauliOpertor(k, self.color_encoding)
        else:
            if self.force_power_of_two_Hamiltonian:
                self.construct_infeasible_states()
            elif self.k_cuts in [3, 5, 6, 7]:
                self.N_qubits_auxillary = 2

        self.construct_colors()

    def validate_parameters(self):
        ### 1) k_cuts needs to be between 2 and 8
        if (self.k_cuts < 2) or (self.k_cuts > 8):
            raise ValueError(
                "k_cuts must be 2 or more, and is not implemented for k_cuts > 8"
            )

        ### 2) if k_cuts is a power of 2, we do not need to force it
        if self.force_power_of_two_Hamiltonian and MaxKCutBinary.is_power_of_two(
            self.k_cuts
        ):
            raise ValueError(
                "k_cuts is a power of two, do not need to force_power_of_two_Hamiltonian"
            )

        #### 3) direct method implements H_P directly, not power of two
        # if self.direct and self.force_power_of_two_Hamiltonian:
        #    raise ValueError("direct method can not force_power_of_two_Hamiltonian")

    def construct_infeasible_states(self):
        if self.k_cuts == 3:
            self.infeasible = ["11"]
        elif self.k_cuts == 5:
            if self.color_encoding == "max_balanced":
                self.infeasible = ["100", "111", "101"]
            else:
                self.infeasible = ["101", "110", "111"]
        elif self.k_cuts == 6:
            if self.color_encoding in ["Dicke1_2", "max_balanced"]:
                self.infeasible = ["000", "111"]
            else:
                self.infeasible = ["110", "111"]
        elif self.k_cuts == 7:
            self.infeasible = ["111"]

    def construct_colors(self):
        if self.k_cuts == 2:
            if self.color_encoding == "all":
                self.colors = {"color1": ["0"], "color2": ["1"]}
            else:
                raise ValueError("invalid or unspecified color_encoding")
        elif self.k_cuts == 3:
            if self.color_encoding == "LessThanK":
                self.colors = {
                    "color1": ["00"],
                    "color2": ["01"],
                    "color3": ["10", "11"],
                }
            else:
                raise ValueError("invalid or unspecified color_encoding")
        elif self.k_cuts == 4:
            if self.color_encoding == "all":
                self.colors = {
                    "color1": ["00"],
                    "color2": ["01"],
                    "color3": ["10"],
                    "color4": ["11"],
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
                # ((1,), (2, 5), (3, 8), (4, 6), (7,))
                self.colors = {
                    "color1": ["000"],
                    "color2": ["001", "100"],
                    "color3": ["010", "111"],
                    "color4": ["011", "101"],
                    "color7": ["110"],
                }
            else:
                raise ValueError("invalid or unspecified color_encoding")
        elif self.k_cuts == 6:
            if self.color_encoding in ["Dicke1_2", "max_balanced"]:
                self.colors = {
                    "color1": ["000", "001"],
                    "color2": ["010"],
                    "color3": ["011"],
                    "color4": ["100"],
                    "color5": ["101"],
                    "color6": ["110", "111"],
                }
            elif self.color_encoding == "LessThanK":
                self.colors = {
                    "color1": ["000"],
                    "color2": ["001"],
                    "color3": ["010"],
                    "color4": ["011"],
                    "color5": ["100"],
                    "color6": ["101", "110", "111"],
                }
            else:
                raise ValueError("invalid or unspecified color_encoding")
        elif self.k_cuts == 7:
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
        elif self.k_cuts == 8:
            if self.color_encoding == "all":
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
            else:
                raise ValueError("invalid or unspecified color_encoding")
        # Create a dictionary to map each index to its corresponding set
        self.bitstring_to_color = {}
        for key, indices in self.colors.items():
            for index in indices:
                self.bitstring_to_color[index] = key

    @staticmethod
    def is_power_of_two(k) -> bool:
        """
        Return True if k is a power of two, False otherwise.
        """
        if k > 0 and (k & (k - 1)) == 0:
            return True
        return False

    def same_color(self, str1, str2) -> bool:
        return self.bitstring_to_color.get(str1) == self.bitstring_to_color.get(str2)

    def is_feasible(self, string: str) -> bool:
        """
        Check if string is feasible.

        Args:
            string (str): Binary string

        Returns:
            (bool) True if string is feasible, False otherwise.
        """
        if not len(string) == self.N_qubits:
            raise ValueError(
                "Lenght of strings should be "
                + str(self.N_qubits)
                + ", but we have "
                + str(len(string))
                + "."
            )
        if self.force_power_of_two_Hamiltonian and (not self.direct):
            for j in range(self.num_V - self.fix_one_node):
                segment = string[j * self.k_bits : (j + 1) * self.k_bits]
                if segment in self.infeasible:
                    return False
            return True
        else:
            return True

    def sliceString(self, string: str) -> list:
        """
        Convert a binary string to string of labels

        Args:
            string (str): Binary string

        Returns:
            (list): list of strings for each node
        """

        k = self.k_bits
        labels = []
        for v in range(self.num_V - self.fix_one_node):
            # Calculate the slice
            label = string[v * k : (v + 1) * k]
            labels.append(label)
        # If we have fixed the last node to "color1"
        if self.fix_one_node:
            labels.append(self.colors["color1"][0])
        return labels

    def cost(self, string: str) -> float | int:
        """
        Compute the cost for a given solution

        Args:
            string (str): Binary string

        Returns:
            (float | int): The cost of the given solution
        """
        if len(string) != self.N_qubits:
            raise ValueError(
                "We expect a string of length"
                + str(self.N_qubits)
                + ", but received length"
                + str(len(string))
            )
        labels = self.sliceString(string)
        C = 0
        for edge in self.graph_handler.G.edges():
            i = edge[0]
            j = edge[1]
            if not self.same_color(labels[i], labels[j]):
                w = self.graph_handler.G[edge[0]][edge[1]]["weight"]
                C += w
        return C

    def N_layer(self, str1: str, str2: str, I: int, J: int):
        """
        Return list of indices of NOT-gates for the N-layer.

        Args:
            c_binary (np.ndarray): Array of c in binary.
            d_binary (np.ndarray): Array of d in binary.
            I (int): Index of first qubit in vertex i.
            J (int): Index of first qubit in vertex j.

        Return:
            (list): List of indices
        """
        N1 = [int(I + index) for index, char in enumerate(str1) if char == "0"]
        N2 = [int(J + index) for index, char in enumerate(str2) if char == "0"]
        return N1 + N2

    def add_equalize_color(self, bs1, bs2, I, J, wg):
        # self.circuit.reset(self.anc[:])
        N = self.N_layer(bs1, bs2, I, J)
        if N:
            self.circuit.x(N)
        control_I = list(range(I, I + self.k_bits))
        control_J = list(range(J, J + self.k_bits))
        self.circuit.mcx(control_I, self.N_qubits)
        self.circuit.mcx(control_J, self.N_qubits + 1)

        # C^{n-1}Phase
        phase_gate = PhaseGate(wg).control(2)
        self.circuit.append(
            phase_gate, [self.N_qubits, self.N_qubits + 1, J + self.k_bits - 1]
        )

        self.circuit.mcx(control_J, self.N_qubits + 1)
        self.circuit.mcx(control_I, self.N_qubits)
        if N:
            self.circuit.x(N)

    def create_circuit(self) -> None:
        """
        Creates the circuit for the problem Hamiltonian
        """
        if self.direct:
            q = QuantumRegister(self.N_qubits)
            c = ClassicalRegister(self.N_qubits)
            self.circuit = QuantumCircuit(q, c)

            ## to avoid a deep circuit, we partition the edges into sets which can be executed in parallel
            for _, edges in self.graph_handler.parallel_edges.items():
                for edge in edges:
                    i = edge[0]
                    j = edge[1]
                    w = self.graph_handler.G[edge[0]][edge[1]]["weight"]
                    wg = w * self.cost_param
                    I = i * self.k_bits
                    J = j * self.k_bits

                    if self.num_V - self.fix_one_node not in [i, j]:
                        self.circuit.append(
                            PauliEvolutionGate(self.op, time=wg),
                            q[
                                list(range(I, I + self.k_bits))
                                + list(range(J, J + self.k_bits))
                            ],
                        )
                    else:
                        # if self.fix_one_node is False, this branch does not exist
                        minIJ = min(I, J)
                        self.circuit.append(
                            PauliEvolutionGate(self.ophalf, time=wg),
                            q[list(range(minIJ, minIJ + self.k_bits))],
                        )
                self.circuit.barrier()

        else:
            if not self.force_power_of_two_Hamiltonian:
                self.anc = AncillaRegister(self.N_qubits_auxillary)
                q = QuantumRegister(self.N_qubits)
                c = ClassicalRegister(self.N_qubits)
                self.circuit = QuantumCircuit(q, c, self.anc)
            else:
                q = QuantumRegister(self.N_qubits)
                c = ClassicalRegister(self.N_qubits)
                self.circuit = QuantumCircuit(q, c)

            self.cost_param = Parameter("x_gamma")

            ## to avoid a deep circuit, we partition the edges into sets which can be executed in parallel
            for _, edges in self.graph_handler.parallel_edges.items():
                for edge in edges:
                    i = edge[0]
                    j = edge[1]
                    w = self.graph_handler.G[edge[0]][edge[1]]["weight"]
                    wg = w * self.cost_param
                    I = i * self.k_bits
                    J = j * self.k_bits

                    ## ordinary MaxCut circuit for k = 2
                    # if self.k_cuts == 2:
                    #    if self.num_V - self.fix_one_node not in [i, j]:
                    #        self.circuit.cx(q[I], q[J])
                    #        self.circuit.rz(wg, q[J])
                    #        self.circuit.cx(q[I], q[J])
                    #    else:
                    #        # if self.fix_one_node is False, this branch does not exist
                    #        minIJ = min(I, J)
                    #        self.circuit.rz(wg, q[minIJ])

                    ## circuit for k >= 3
                    # else:

                    if self.num_V - self.fix_one_node not in [i, j]:
                        for k in range(self.k_bits):
                            self.circuit.cx(I + k, J + k)
                            self.circuit.x(J + k)
                        # C^{n-1}Phase
                        if self.k_bits == 1:
                            phase_gate = PhaseGate(-wg)
                        else:
                            phase_gate = PhaseGate(-wg).control(self.k_bits - 1)
                        self.circuit.append(
                            phase_gate,
                            [J - 1 + ind for ind in range(1, self.k_bits + 1)],
                        )
                        for k in reversed(range(self.k_bits)):
                            self.circuit.x(J + k)
                            self.circuit.cx(I + k, J + k)
                        if not self.force_power_of_two_Hamiltonian:
                            # for all pairs of colors with length > 1
                            for _, bitstrings in self.colors.items():
                                if len(bitstrings) > 1:
                                    pairs = list(itertools.combinations(bitstrings, 2))
                                    for bs1, bs2 in pairs:
                                        self.add_equalize_color(bs1, bs2, I, J, -wg)
                                        self.add_equalize_color(bs2, bs1, I, J, -wg)
                    else:
                        # if self.fix_one_node is False, this branch does not exist
                        minIJ = min(I, J)
                        for k in range(self.k_bits):
                            self.circuit.x(minIJ + k)
                        # C^{n-1}Phase
                        if self.k_bits == 1:
                            phase_gate = PhaseGate(-wg)
                        else:
                            phase_gate = PhaseGate(-wg).control(self.k_bits - 1)
                        self.circuit.append(
                            phase_gate,
                            [minIJ - 1 + ind for ind in range(1, self.k_bits + 1)],
                        )
                        for k in reversed(range(self.k_bits)):
                            self.circuit.x(minIJ + k)
                self.circuit.barrier()
