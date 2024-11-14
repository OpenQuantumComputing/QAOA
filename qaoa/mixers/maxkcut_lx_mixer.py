import math

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.circuit.library import PauliEvolutionGate

from .base_mixer import Mixer


class MaxKCutLX(Mixer):
    def __init__(self, k_cuts: int, color_encoding: str, topology: str = "standard"):
        if (k_cuts < 2) or (k_cuts > 8):
            raise ValueError(
                "k_cuts must be 2 or more, and is not implemented for k_cuts > 8"
            )
        self.k_cuts = k_cuts
        self.k_bits = int(np.ceil(np.log2(k_cuts)))
        self.color_encoding = color_encoding
        self.topology = topology

        if self.is_power_of_two():
            raise ValueError("k_cuts is a power of two. Use e.g. X-mixer instead.")

        if not color_encoding:
            raise ValueError("please specify a color encoding")

        if k_cuts == 3 and (topology not in ["standard", "ring"]):
            raise ValueError('topology must be in ["standard", "ring"]')

        self.create_SparsePauliOp()

    def is_power_of_two(self) -> bool:
        """
        Return True if self.k_cuts is a power of two, False otherwise.
        """
        if self.k_cuts > 0 and (self.k_cuts & (self.k_cuts - 1)) == 0:
            return True
        return False

    def create_SparsePauliOp(self) -> None:
        """
        Create sparse Pauli operator for given k. Hard coded

        Returns:
            None
        """
        if self.k_cuts == 3:
            if self.color_encoding in ["LessThanK"]:
                LXM = {
                    Pauli("IX"): [Pauli("ZI")],
                    Pauli("XI"): [Pauli("IZ")],
                }
                if self.topology == "ring":
                    LXM[Pauli("XX")] = [Pauli("-ZZ")]
            else:
                raise ValueError("invalid or missing color_encoding")

        elif self.k_cuts == 5:
            if self.color_encoding in ["LessThanK"]:
                LXM = {
                    Pauli("IXX"): [Pauli("ZII")],
                    Pauli("IXI"): [Pauli("ZII")],
                    Pauli("XII"): [Pauli("IIZ"), Pauli("IZZ"), Pauli("IZI")],
                }
            else:
                raise ValueError("invalid or missing color_encoding")

        elif self.k_cuts == 6:
            if self.color_encoding == "LessThanK":
                LXM = {
                    Pauli("IIX"): [],
                    Pauli("IXI"): [Pauli("ZII")],
                    Pauli("XII"): [Pauli("IZI")],
                }
            elif self.color_encoding in ["Dicke1_2", "max_balanced"]:
                LXM = {
                    Pauli("IXX"): [-Pauli("IZZ")],
                    Pauli("XXI"): [-Pauli("ZZI")],
                    Pauli("IXI"): [-Pauli("ZIZ")],
                }
            else:
                raise ValueError("invalid or missing color_encoding")

        elif self.k_cuts == 7:
            if self.color_encoding == "LessThanK":
                LXM = {
                    Pauli("IIX"): [Pauli("ZII")],
                    Pauli("IXI"): [Pauli("IIZ")],
                    Pauli("XII"): [Pauli("IZI")],
                }
            else:
                raise ValueError("invalid or missing color_encoding")

        data = []
        coeffs = []

        # iterate through LXM dict,
        for PX, PZs in LXM.items():
            count = 1
            data.append(PX)
            for pz in PZs:
                composed = PX.compose(pz)
                data.append(composed)
                count += 1
            coeffs += [1 / (len(PZs) + 1)] * (len(PZs) + 1)
        self.op = SparsePauliOp(data, coeffs=coeffs)

    def create_circuit(self) -> None:
        self.num_V = int(self.N_qubits / self.k_bits)
        q = QuantumRegister(self.N_qubits)
        mixer_param = Parameter("x_beta")
        self.circuit = QuantumCircuit(q, name="Mixer")
        if math.log(self.k_cuts, 2).is_integer():
            self.circuit.rx(-2 * mixer_param, range(self.N_qubits))
        else:
            for v in range(self.num_V):
                self.circuit.append(
                    PauliEvolutionGate(self.op, time=mixer_param),
                    q[self.k_bits * v : self.k_bits * (v + 1)][::-1],
                )
