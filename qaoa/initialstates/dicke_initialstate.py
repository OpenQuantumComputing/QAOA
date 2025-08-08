import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import RYGate

from .base_initialstate import InitialState


class Dicke(InitialState):
    """
    Dicke initial state.

    Subclass of the `InitialState` class. Creates a circuit representing a Dicke state with Hamming weight k.

    Attributes: 
        k (int): The Hamming weight of the Dicke states.

    Methods:
        create_circuit(): Creates the circuit to prepare the Dicke states.
    """
    def __init__(self, k) -> None:
        """
        Args:
            k (int): The Hamming weight of the Dicke states.
        """
        super().__init__()
        self.k = k

    def create_circuit(self):
        """
        Circuit to prepare Dicke states, following the algorithm from https://arxiv.org/pdf/1904.07358.pdf.
        """

        q = QuantumRegister(self.N_qubits)
        self.circuit = QuantumCircuit(q)

        self.circuit.x(q[-self.k :])

        for l in range(self.k + 1, self.N_qubits + 1)[::-1]:
            self.circuit.append(
                Dicke.getBlock1(self.N_qubits, self.k, l), range(self.N_qubits)
            )

        for l in range(2, self.k + 1)[::-1]:
            self.circuit.append(
                Dicke.getBlock2(self.N_qubits, self.k, l), range(self.N_qubits)
            )

    @staticmethod
    def getRYi(n):
        """
        Returns gate (i) from section 2.2.

        Args:
            n (int): The integer parameter for gate (i).

        Returns:
            QuantumCircuit: Quantum circuit representing gate (i).
        """

        qc = QuantumCircuit(2)

        qc.cx(0, 1)
        theta = 2 * np.arccos(np.sqrt(1 / n))
        ry = RYGate(theta).control(ctrl_state="1")
        qc.append(ry, [1, 0])
        qc.cx(0, 1)

        return qc

    @staticmethod
    def getRYii(l, n):
        """
        Returns gate (ii)_l from section 2.2.

        Args:
            l (int): The integer parameter for gate (ii)_l.
            n (int): The integer parameter for gate (ii)_l.

        Returns:
            QuantumCircuit: Quantum circuit representing gate (ii)_l.
        """

        qc = QuantumCircuit(3)

        qc.cx(0, 2)
        theta = 2 * np.arccos(np.sqrt(l / n))
        ry = RYGate(theta).control(num_ctrl_qubits=2, ctrl_state="11")
        qc.append(ry, [2, 1, 0])
        qc.cx(0, 2)

        return qc

    @staticmethod
    def getSCS(n, k):
        """
        Returns SCS_{n,k} gate from definition 3.

        Args:
            n (int): The integer parameter for SCS_{n,k}.
            k (int): The integer parameter for SCS_{n,k}.

        Returns:
            QuantumCircuit: Quantum circuit representing SCS_{n,k}.
        """

        qc = QuantumCircuit(k + 1)

        qc.append(Dicke.getRYi(n), [k - 1, k])
        for l in range(2, k + 1):
            qc.append(Dicke.getRYii(l, n), [k - l, k - l + 1, k])

        return qc

    @staticmethod
    def getBlock1(n, k, l):
        """
        Returns the first block in Lemma 2.

        Args:
            n (int): The integer parameter for the quantum register size.
            k (int): The integer parameter for the Hamming weight.
            l (int): The integer parameter for the block.

        Returns:
            QuantumCircuit: Quantum circuit representing the first block in Lemma 2.
        """

        qr = QuantumRegister(n)
        qc = QuantumCircuit(qr)

        first = l - k - 1
        last = n - l

        index = list(range(n))

        if first != 0:
            index = index[first:]

        if last != 0:
            index = index[:-last]
            qc.append(Dicke.getSCS(l, k), index)
        else:
            qc.append(Dicke.getSCS(l, k), index)

        return qc

    @staticmethod
    def getBlock2(n, k, l):
        """
        Returns the second block from Lemma 2.

        Args:
            n (int): The integer parameter for the quantum register size.
            k (int): The integer parameter for the Hamming weight.
            l (int): The integer parameter for the block.

        Returns:
            QuantumCircuit: Quantum circuit representing the second block in Lemma 2.
        """

        qr = QuantumRegister(n)
        qc = QuantumCircuit(qr)

        last = n - l
        index = list(range(n))

        if last != 0:
            index = index[:-last]
            qc.append(Dicke.getSCS(l, l - 1), index)
        else:
            qc.append(Dicke.getSCS(l, l - 1), index)

        return qc
