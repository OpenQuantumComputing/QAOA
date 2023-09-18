import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import RYGate

def getGate_i(n):
    '''
    returns gate (i) from section 2.2
    '''

    qc = QuantumCircuit(2)

    qc.cnot(0, 1)
    theta = 2*np.arccos(np.sqrt(1/n))
    ry=RYGate(theta).control(ctrl_state="1")
    qc.append(ry, [1, 0])
    qc.cnot(0, 1)

    return qc.to_gate()

def getGate_ii_l(l, n):
    '''
    returns gate (ii)_l from section 2.2
    '''

    qc = QuantumCircuit(3)

    qc.cnot(0, 2)
    theta = 2*np.arccos(np.sqrt(l/n))
    ry = RYGate(theta).control(num_ctrl_qubits = 2, ctrl_state="11")
    qc.append(ry, [2, 1, 0])
    qc.cnot(0, 2)

    return qc.to_gate()

def getGate_scs_nk(n, k):
    '''
    returns SCS_{n,k} gate from definition 3
    '''

    qc = QuantumCircuit(k+1)

    qc.append(getGate_i(n), [k-1, k])
    for l in range(2, k+1):
        qc.append(getGate_ii_l(l, n), [k-l, k-l+1, k])

    return qc.to_gate()

def getFirstBlock(n, k, l):
    '''
    returns the first block in Lemma 2
    '''

    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr)

    first = l-k-1
    last = n-l

    index = list(range(n))

    if first != 0:
        index = index[first:]
        qc.i(qr[:first])

    if last !=0:
        index = index[:-last]
        qc.append(getGate_scs_nk(l, k), index)
        qc.i(qr[-last:])

    else:
        qc.append(getGate_scs_nk(l, k), index)

    return qc.to_gate()

def getSecondBlock(n, k, l):
    '''
    returns second block from Lemma 2
    '''

    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr)

    last = n-l
    index = list(range(n))

    if last !=0:
        index = index[:-last]
        qc.append(getGate_scs_nk(l, l-1), index)
        qc.i(qr[-last:])
    else:
        qc.append(getGate_scs_nk(l, l-1), index)

    return qc.to_gate()

def dicke_state(n, k, barrier=False):
    '''
    circuit to prepare Dicke states, following algorithm from https://arxiv.org/pdf/1904.07358.pdf
    args:
        n: number of qubits
        k: Hamming weight of states
    '''

    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr)

    qc.x(qr[-k:])

    for l in range(k+1, n+1)[::-1]:
        qc.append(getFirstBlock(n, k, l), range(n))

    for l in range(2, k+1)[::-1]:
        qc.append(getSecondBlock(n, k, l), range(n))

    return qc
