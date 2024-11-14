import unittest
import sys
import numpy as np
import networkx as nx

sys.path.append("../")

from qaoa.problems import MaxKCutOneHot


class TestMaxKCutOneHotProblem(unittest.TestCase):
    def __init__(self, methodname):
        super().__init__(methodname)
        V = np.arange(0, 2, 1)
        E = [(0, 1, 1.0)]
        self.barbell = nx.Graph()
        self.barbell.add_nodes_from(V)
        self.barbell.add_weighted_edges_from(E)

        V = np.arange(0, 3, 1)
        E = [(0, 1, 1.0), (1, 2, 1.0)]
        self.three_node_graph = nx.Graph()
        self.three_node_graph.add_nodes_from(V)
        self.three_node_graph.add_weighted_edges_from(E)

    def test_binstringToLabels_k2(self):
        """
        Test that MaxKCutOneHot.binstringToLabels() outputs correct labels for k = 2.
        """
        prob = MaxKCutOneHot(self.barbell, 2)
        labels = {"0101": "00", "0110": "01", "1010": "11", "1001": "10"}
        for binstring, expected in labels.items():
            computed = prob.binstringToLabels(binstring)
            msg = f"string: {binstring}, expected: {expected}, computed: {computed}"
            self.assertEqual(expected, computed, msg)

    def test_binstringToLabels_k3(self):
        """
        Test that MaxKCutOneHot.binstringToLabels() outputs correct labels for k = 3.
        """
        prob = MaxKCutOneHot(self.barbell, 3)
        labels = {
            "001001": "00",
            "001010": "01",
            "001100": "02",
            "010001": "10",
            "010010": "11",
            "010100": "12",
            "100001": "20",
            "100010": "21",
            "100100": "22",
        }
        for binstring, expected in labels.items():
            computed = prob.binstringToLabels(binstring)
            msg = f"string: {binstring}, expected: {expected}, computed: {computed}"
            self.assertEqual(expected, computed, msg)

    def test_cost_k2_barbell(self):
        """
        Test that the cost funciton in MaxKCutBinaryOneHot is correct for k = 2 with the barbell graph.
        """
        prob = MaxKCutOneHot(self.barbell, 2)
        strings = {"0101": 0, "1001": 1, "0110": 1, "1010": 0}
        for string, expected in strings.items():
            computed = prob.cost(string[::-1])
            msg = f"string: {string}, expected: {expected}, computed: {computed}"
            self.assertEqual(expected, computed, msg)

    def test_cost_k2_three_node_graph(self):
        """
        Test that the cost funciton in MaxKCutBinaryOntHot is correct for k = 2 with three-node graph.
        """
        prob = MaxKCutOneHot(self.three_node_graph, 2)
        strings = {"010101": 0, "100110": 2, "011010": 1, "101010": 0}
        for string, expected in strings.items():
            computed = prob.cost(string[::-1])
            msg = f"string: {string}, expected: {expected}, computed: {computed}"
            self.assertEqual(expected, computed, msg)

    def test_cost_k3_three_node_graph(self):
        """
        Test that the cost funciton in MaxKCutBinaryOntHot is correct for k = 3 with three-node graph.
        """
        prob = MaxKCutOneHot(self.three_node_graph, 3)
        strings = {
            "001010100": 2,
            "010100001": 2,
            "100100010": 1,
            "001001001": 0,
            "001100100": 1,
            "100100100": 0,
        }
        for string, expected in strings.items():
            computed = prob.cost(string[::-1])
            msg = f"string: {string}, expected: {expected}, computed: {computed}"
            self.assertEqual(expected, computed, msg)

    def test_cost_k6_three_node_graph(self):
        """
        Test that the cost funciton in MaxKCutBinaryOntHot is correct for k = 3 with three-node graph.
        """
        prob = MaxKCutOneHot(self.three_node_graph, 6)
        strings = {
            "000100100000000010": 2,
            "100000100000100000": 0,
            "010000100000100000": 1,
            "000001100000010000": 2,
        }
        for string, expected in strings.items():
            computed = prob.cost(string[::-1])
            msg = f"string: {string}, expected: {expected}, computed: {computed}"
            self.assertEqual(expected, computed, msg)


if __name__ == "__main__":
    unittest.main()
