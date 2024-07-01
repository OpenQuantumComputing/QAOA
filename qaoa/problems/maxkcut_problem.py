from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import CU3Gate

import numpy as np

from .base_problem import Problem


class MaxKCut(Problem):
    def __init__(self, G, k_cuts):
        self.G = G
        self.num_V = self.G.number_of_nodes()
        self.k_cuts = k_cuts
        self.k_bits = int(np.ceil(np.log2(k_cuts)))
        self.circuit = None

    def binstringToLabels(self, binstring):
        n = self.k_bits
        label_list = [int(binstring[j*n:(j+1)*n], 2) for j in range(self.num_V)]
        label_string = "".join(map(str, label_list))
        return label_string

    def cost(self, string):
        labels = self.binstringToLabels(string)
        C = 0
        for edge in self.G.edges():
            i = edge[0]
            j = edge[1]
            li = min(self.k_cuts - 1, int(labels[i]))
            lj = min(self.k_cuts - 1, int(labels[j]))
            if li != lj:
                w = self.G[edge[0]][edge[1]]["weight"]
                C += w
        return C
    
    def create_circuit(self):
        q = QuantumRegister(self.num_V * self.k_bits)
        c = ClassicalRegister(self.num_V * self.k_bits)
        self.circuit = QuantumCircuit(q, c)
        cost_param = Parameter("x_gamma")
        for edge in self.G.edges():
            i = int(edge[0])
            j = int(edge[1])
            w = self.G[edge[0]][edge[1]]['weight']
            wg = w * cost_param
            I = i * self.k_bits
            J = j * self.k_bits
            self.circuit.barrier()
            for k in range(self.k_bits):
                self.circuit.cx(I + k, J + k)
                self.circuit.x(J + k)
            self.Cn_U3_0theta0([J-1+ind for ind in range(1, self.k_bits)], J+self.k_bits-1, -wg)
            for k in reversed(range(self.k_bits)):
                self.circuit.x(J + k)
                self.circuit.cx(I + k, J + k)
            self.circuit.barrier()

    def Cn_U3_0theta0(self, control_indices, target_index, theta):
            """
            Ref: https://arxiv.org/abs/0708.3274
            """
            n=len(control_indices)
            if n == 0:
                self.circuit.rz(theta, control_indices)
            elif n == 1:
                self.circuit.append(CU3Gate(0, theta, 0), [control_indices, target_index])
            elif n == 2:
                self.circuit.append(CU3Gate(0, theta/ 2, 0), [control_indices, target_index])  # V gate, V^2 = U
                self.circuit.cx(control_indices[0], control_indices[1])
                self.circuit.append(CU3Gate(0, -theta/ 2, 0), [control_indices, target_index])  # V dagger gate
                self.circuit.cx(control_indices[0], control_indices[1])
                self.circuit.append(CU3Gate(0, theta/ 2, 0), [control_indices, target_index]) #V gate
            else:
                raise Exception("C^nU_3(0,theta,0) not yet implemented for n="+str(n)+".")
        