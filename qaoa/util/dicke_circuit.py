import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import RYGate


def getRYi(n):
    """
    Returns gate (i) from section 2.2.

    Args:
        n (int): The integer parameter for gate (i).

    Returns:
        QuantumCircuit: Quantum circuit representing gate (i).
    """

    qc = QuantumCircuit(2)

    qc.cnot(0, 1)
    theta = 2 * np.arccos(np.sqrt(1 / n))
    ry = RYGate(theta).control(ctrl_state="1")
    qc.append(ry, [1, 0])
    qc.cnot(0, 1)

    return qc


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

    qc.cnot(0, 2)
    theta = 2 * np.arccos(np.sqrt(l / n))
    ry = RYGate(theta).control(num_ctrl_qubits=2, ctrl_state="11")
    qc.append(ry, [2, 1, 0])
    qc.cnot(0, 2)

    return qc


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

    qc.append(getRYi(n), [k - 1, k])
    for l in range(2, k + 1):
        qc.append(getRYii(l, n), [k - l, k - l + 1, k])

    return qc


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
        qc.i(qr[:first])

    if last != 0:
        index = index[:-last]
        qc.append(getSCS(l, k), index)
        qc.i(qr[-last:])
    else:
        qc.append(getSCS(l, k), index)

    return qc


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
        qc.append(getSCS(l, l - 1), index)
        qc.i(qr[-last:])
    else:
        qc.append(getSCS(l, l - 1), index)

    return qc


def DickeCircuit(n, k, barrier=False):
    """
    Circuit to prepare Dicke states, following the algorithm from https://arxiv.org/pdf/1904.07358.pdf.

    Args:
        n (int): The number of qubits in the quantum register.
        k (int): The Hamming weight of the Dicke states.
        barrier (bool, optional): Whether to insert barriers between gates for clarity (default is False).

    Returns:
        QuantumCircuit: Quantum circuit to prepare Dicke states.
    """

    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr)

    qc.x(qr[-k:])

    for l in range(k + 1, n + 1)[::-1]:
        qc.append(getBlock1(n, k, l), range(n))

    for l in range(2, k + 1)[::-1]:
        qc.append(getBlock2(n, k, l), range(n))

    if barrier:
        qc.barrier()

    return qc
