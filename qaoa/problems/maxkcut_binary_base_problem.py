import numpy as np
import itertools

from abc import abstractmethod
from .graph_problem import GraphProblem


class MaxKCutBinaryBase(GraphProblem):
    def __init__(
        self,
        G,
        N_qubits_per_node=1,
        fix_one_node: bool = False,  # this fixes the last node to color 1, i.e., one qubit gets removed
    ) -> None:
        super().__init__(G, N_qubits_per_node, fix_one_node)

    @abstractmethod
    def construct_colors(self) -> dict:
        """
        Abstract method to define a mapping of color names to binary string sequences
        This method must be implemented in subclasses.
        Returns:
            dict: A dictionary with color names as keys and binary string sequences as values.
        """

    def set_colors(self, colors):
        self.colors = colors
        # Create a dictionary to map each index to its corresponding set
        self.bitstring_to_color = {}
        for key, indices in self.colors.items():
            for index in indices:
                self.bitstring_to_color[index] = key

    def same_color(self, str1: str, str2: str) -> bool:
        """Check if two strings map to the same color."""
        return self.bitstring_to_color.get(str1) == self.bitstring_to_color.get(str2)

    def __slice_string__(self, string: str) -> list:
        """
        Convert a binary string to a list of labels for each node.

        Args:
            string (str): Binary string.

        Returns:
            list: List of labels for each node.
        """
        k = self.N_qubits_per_node
        labels = [
            string[v * k : (v + 1) * k] for v in range(self.num_V - self.fix_one_node)
        ]
        # Add fixed node label if applicable
        if self.fix_one_node:
            labels.append(self.colors["color1"][0])
        return labels

    def cost(self, string: str) -> float | int:
        """
        Compute the cost for a given solution.

        Args:
            string (str): Binary string.

        Returns:
            float | int: The cost of the given solution.
        """
        if len(string) != self.N_qubits:
            raise ValueError(
                f"Expected a string of length {self.N_qubits}, "
                f"but received length {len(string)}."
            )

        labels = self.__slice_string__(string)
        return sum(
            self.graph_handler.G[edge[0]][edge[1]].get("weight", 1)
            for edge in self.graph_handler.G.edges()
            if not self.same_color(labels[edge[0]], labels[edge[1]])
        )
