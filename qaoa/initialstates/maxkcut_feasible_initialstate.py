import numpy as np


from .base_initialstate import InitialState
from .dicke_initialstate import Dicke
from .dicke1_2_initialstate import Dicke1_2
from .lessthank_initialstate import LessThanK
from .tensor_initialstate import Tensor


class MaxKCutFeasible(InitialState):
    """
    Class that determines the feasible states for the type of MAX k-CUT problem with specified number of cuts, number of qubits per vertex,
    and 

    Attributes: 

    Methods:
        create_circuit():
    """
    def __init__(
        self, k_cuts: int, problem_encoding: str, color_encoding: str = "LessThanK"
    ) -> None:
        """
        Args:
            k_cuts (int):
            problem_encoding (str): description of the type of problem, either "onehot" (which corresponds to ...) or "binary" (which corresponds to ...)
            color_encoding (str): determines the approach to solving the MAX k-cut problem by following one of two methods, 
            either "Dicke1_2" (which corresponds to creating an initial state that is a superposition of the valid states that represents a color(only 6/8 possible states)) 
            or "LessThanK" (which corresponds to grouping states together and make the group represent one color)

        """
        self.k_cuts = k_cuts
        self.problem_encoding = problem_encoding
        self.color_encoding = color_encoding

        if not problem_encoding in ["onehot", "binary"]:
            raise ValueError('case must be in ["onehot", "binary"]')
        if problem_encoding == "binary":
            if k_cuts == 6 and (color_encoding not in ["Dicke1_2", "LessThanK"]):
                raise ValueError('color_encoding must be in ["LessThanK", "Dicke1_2"]')
            self.color_encoding = color_encoding

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

    def create_circuit(self) -> None:
        """

        """
        if self.problem_encoding == "binary":
            self.k_bits = int(np.ceil(np.log2(self.k_cuts)))
            self.num_V = self.N_qubits / self.k_bits

            if not self.num_V.is_integer():
                raise ValueError(
                    "Total qubits="
                    + str(self.N_qubits)
                    + " is not a multiple of "
                    + str(self.k_bits)
                )
            if self.k_cuts == 6 and self.color_encoding == "Dicke1_2":
                circ_one_node = Dicke1_2()
            else:
                circ_one_node = LessThanK(self.k_cuts)

        elif self.problem_encoding == "onehot":
            self.num_V = self.N_qubits / self.k_cuts

            if not self.num_V.is_integer():
                raise ValueError(
                    "Total qubits="
                    + str(self.N_qubits)
                    + " is not a multiple of "
                    + str(self.k_cuts)
                )
            self.num_V = int(self.num_V)

            circ_one_node = Dicke(1)
            circ_one_node.setNumQubits(self.k_cuts)

        self.num_V = int(self.num_V)
        self.tensor = Tensor(circ_one_node, self.num_V)

        self.tensor.create_circuit()
        self.circuit = self.tensor.circuit
