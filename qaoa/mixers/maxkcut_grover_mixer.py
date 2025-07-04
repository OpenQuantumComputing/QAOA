import numpy as np

from qaoa.mixers import Mixer, Grover
from qaoa.initialstates import Dicke, Plus

from qaoa.initialstates.dicke1_2_initialstate import Dicke1_2
from qaoa.initialstates.lessthank_initialstate import LessThanK
from qaoa.initialstates.tensor_initialstate import Tensor


class MaxKCutGrover(Mixer):
    """
    Grover mixer for the Max K-Cut problem.

    Subclass of the `Mixer` subclass that implements the Grover mixing operation for the Max k-cut problem.

    Attributes:
        k_cuts (int): The number of cuts in the Max k-Cut problem.
        problem_encoding (str): The encoding of the problem, either "onehot" or "binary".
        color_encoding (str): The encoding of colors, can be "max_balanced", "Dicke1_2", or "LessThanK".
        tensorized (bool): Whether to use tensorization for the mixer.

    Methods:
        is_power_of_two(): Returns True if `k_cuts` is a power of two, False otherwise.
        set_numV(k): Sets the number of vertices based on the number of cuts.
        create_circuit(): Constructs the Grover mixer circuit for the Max k-Cut problem.
    """

    def __init__(
        self, k_cuts: int, problem_encoding: str, color_encoding: str, tensorized: bool
    ) -> None:
        """
        Initializes the MaxKCutGrover mixer.

        Args:
            k_cuts (int): The number of cuts in the Max k-Cut problem.
            problem_encoding (str): The encoding of the problem, either "onehot" or "binary".
            color_encoding (str): The encoding of colors, can be "max_balanced", "Dicke1_2", or "LessThanK".
            tensorized (bool): Whether to use tensorization for the mixer.

        Raises:
            ValueError: If `k_cuts` is less than 2 or greater than 8, or if `problem_encoding` is not valid.
            ValueError: If `color_encoding` is not valid for the given `k_cuts`.
        """
        if (k_cuts < 2) or (k_cuts > 8):
            raise ValueError(
                "k_cuts must be 2 or more, and is not implemented for k_cuts > 8"
            )
        if not problem_encoding in ["onehot", "binary"]:
            raise ValueError('problem_encoding must be in ["onehot", "binary"]')
        self.k_cuts = k_cuts
        self.problem_encoding = problem_encoding
        self.color_encoding = color_encoding
        self.tensorized = tensorized

        if (self.problem_encoding == "binary") and self.is_power_of_two():
            print(
                "k_cuts is a power of two. You might want to use the X-mixer instead."
            )

        # for k=6, max_balanced == Dicke1_2
        if k_cuts == 6 and (
            color_encoding not in ["max_balanced", "Dicke1_2", "LessThanK"]
        ):
            raise ValueError(
                'color_encoding must be in ["LessThanK", "Dicke1_2", max_balanced]'
            )

    def is_power_of_two(self) -> bool:
        """
        Return True if self.k_cuts is a power of two, False otherwise.
        """
        if self.k_cuts > 0 and (self.k_cuts & (self.k_cuts - 1)) == 0:
            return True
        return False

    def set_numV(self, k):
        """
        Set the number of vertices based on the number of cuts.

        Args:
            k (int): The number of cuts in the Max k-Cut problem.

        Raises:
            ValueError: If the total number of qubits is not a multiple of k.
        """
        num_V = self.N_qubits / k

        if not num_V.is_integer():
            raise ValueError(
                "Total qubits=" + str(self.N_qubits) + " is not a multiple of " + str(k)
            )

        self.num_V = int(num_V)

    def create_circuit(self) -> None:
        """
        Constructs the Grover mixer circuit for the Max k-Cut problem.
        """

        if self.problem_encoding == "binary":
            self.k_bits = int(np.ceil(np.log2(self.k_cuts)))
            self.set_numV(self.k_bits)

            if self.is_power_of_two():
                circ_one_node = Plus()
                circ_one_node.N_qubits = self.k_bits
            elif self.k_cuts == 6 and self.color_encoding in [
                "max_balanced",
                "Dicke1_2",
            ]:
                circ_one_node = Dicke1_2()
            else:
                circ_one_node = LessThanK(self.k_cuts)

        elif self.problem_encoding == "onehot":
            self.set_numV(self.k_cuts)

            circ_one_node = Dicke(1)
            circ_one_node.setNumQubits(self.k_cuts)

        if self.tensorized:
            gm = Grover(circ_one_node)

            tensor_gm = Tensor(gm, self.num_V)

            tensor_gm.create_circuit()
            self.circuit = tensor_gm.circuit
        else:
            tensor_feas = Tensor(circ_one_node, self.num_V)

            gm = Grover(tensor_feas)

            gm.create_circuit()
            self.circuit = gm.circuit
