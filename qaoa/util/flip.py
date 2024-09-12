import numpy as np
from qiskit import QuantumCircuit, QuantumRegister


class BitFlip:
    def __init__(self, n):
        self.circuit = None
        self.N_qubits = n

    def boost_samples(self, problem, string, K=5):
        """
        Random bitflips on string/list of strings to increase cost.

        input:
            - problem: BaseType Problem
            - string (str): string or list of strings
            - K (int): number of iteratations through string while flipping
        returns:
            - str: string after bitflips
        """
        string_arr = np.array([int(bit) for bit in string])
        old_string = string
        cost = problem.cost(string[::-1])

        for _ in range(K):
            shuffled_indices = np.arange(self.N_qubits)
            np.random.shuffle(shuffled_indices)

            for i in shuffled_indices:
                string_arr_altered = np.copy(string_arr)
                string_arr_altered[i] = not (string_arr[i])
                string_altered = "".join(map(str, string_arr_altered))
                new_cost = problem.cost(string_altered[::-1])

                if new_cost > cost:
                    cost = new_cost
                    string_arr = string_arr_altered
                    string = string_altered

        return string

    def xor(self, old_string, new_string):
        """
        Finds (old_string XOR new_string)

        input:
            - old_string (str): string before bitflips
            - new_string (str): string after bitflips
        returns:
            - list of qubits on which to apply X-gate
                if 1 at pos n - i, apply X-gate to qubit i
                if 0 at pos n - j, do nothing to qubit j
        """
        old = np.array([int(bit) for bit in old_string])
        new = np.array([int(bit) for bit in new_string])
        xor = []

        for a, b in zip(old, new):
            xor.append((a and (not b)) or ((not a) and b))

        return xor

    def create_circuit(self, xor: list[int | bool]) -> None:
        """
        Creates quantum circuit that performs bitflips

        input:
            - xor (list): list of qubits on which to apply X-gate
                if 1 at pos n - i, apply X-gate to qubit i
                if 0 at pos n - j, do nothing to qubit j
        returns:
            None
        """
        q = QuantumRegister(self.N_qubits)
        self.circuit = QuantumCircuit(q)
        indices_flip = np.where(xor[::-1])[0]
        if np.any(indices_flip):
            self.circuit.x(indices_flip)
