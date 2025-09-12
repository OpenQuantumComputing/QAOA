from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

import numpy as np

def _bitstring(i, n, flip=False):
    if flip:
        return format(i, f'0{n}b')[::-1]
    else:
        return format(i, f'0{n}b')

 
def check_phase_separator_exact_qaoa(qaoa, *arg, **kwarg):
    return check_phase_separator_exact_problem(qaoa.problem, *arg, **kwarg)

def check_phase_separator_exact_problem(problem, t=1, flip=True, atol=1e-8, rtol=1e-8):
    """
    Exact check that the problem's circuit represents the problem's cost function.
    This tests checks that the unitary operator represented by the quantum circuit is
    equal to the excepted matrix with diagonal elements 
    exp(-j*t*cost(e)),
    where e is the corresponding binary state, up to a global phase.
    
    Suitable for <= 10 qubits as this check uses the full unitary matrix of size 2^n x 2^n).
    Returns: (ok: bool, report: dict)
    """

    paramed_circ = problem.circuit
    circ = paramed_circ.assign_parameters(
        {problem.circuit.parameters[0]: t},
        inplace = False
    )
    cost_fn = problem.cost

    U = Operator(circ).data  # complex ndarray
    n = circ.num_qubits
    d = 2**n
    # Compare diagonal phases to expected, modulo a global phase
    # expected diag entries
    costs = []
    for i in range(d):
        costs.append(cost_fn(_bitstring(i, n, flip=flip)))
    expected = np.exp(1j * t * np.asarray(costs, dtype=float))
    

    diag = np.diag(U)
    # if n < 4: 
    #     for i in range(d):
    #         print(expected[i]* diag[0], diag[i])
    # Remove global phase by aligning first nonzero expected
    ref_idx = 0
    g = diag[ref_idx] / expected[ref_idx]  # global phase factor
    ratios = diag / (expected * g)

    # Errors
    mag_err = np.max(np.abs(np.abs(diag) - 1.0))
    phase_err = np.max(np.abs(np.angle(ratios)))  # max residual phase after removing global
    ok = (mag_err <= rtol) and (phase_err <= atol)

    report = {
        "n_qubits": n,
        "max_magnitude_error": float(mag_err),
        "max_phase_error_rad_after_global": float(phase_err),
        "global_phase_rad": float(np.angle(g)),
    }
    if not ok:
        # include a few worst offenders
        idx_sorted = np.argsort(-np.abs(np.angle(ratios)))
        bad = []
        for k in idx_sorted[:8]:
            bad.append({
                "bitstring": list(_bitstring(k, n, flip=flip)),
                "diag_entry": complex(diag[k]),
                "expected": complex(expected[k]*g),
                "phase_residual_rad": float(np.angle(ratios[k])),
                "magnitude": float(np.abs(diag[k])),
            })
        report["examples"] = bad
    return ok, report
