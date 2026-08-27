from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

import numpy as np
import warnings

def _bitstring(i, n, flip=False):
    if flip:
        return format(i, f'0{n}b')[::-1]
    else:
        return format(i, f'0{n}b')

 
def check_phase_separator_exact_qaoa(qaoa, *arg, **kwarg):
    return check_phase_separator_exact_problem(qaoa.problem, *arg, **kwarg)

def _validation_energy(problem, bitstring, infeasible_energy, omit_infeasible_states):
    if problem.isFeasible(bitstring):
        return problem.energy(bitstring), True
    return infeasible_energy, not omit_infeasible_states


def check_phase_separator_exact_problem(
    problem,
    t=1,
    flip=True,
    atol=1e-8,
    rtol=1e-8,
    infeasible_energy=0.0,
    omit_infeasible_states=False,
    global_phase=None,
):
    """
    Exact check that the problem's circuit represents the problem's energy function.
    This test checks that the unitary operator represented by the quantum circuit is
    equal to the expected matrix with diagonal elements 
    exp(-j*t*energy(e)),
    where e is the corresponding binary state, up to a global phase.

    For infeasible states, a fixed placeholder energy can be used
    (infeasible_energy), and these states can optionally be omitted
    from the phase comparison by setting omit_infeasible_states=True.
    
    Suitable for <= 10 qubits as this check uses the full unitary matrix of size 2^n x 2^n).
    Returns: (ok: bool, report: dict)
    """
    if global_phase is not None:
        warnings.warn(
            "global_phase is deprecated, use infeasible_energy instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not np.isclose(infeasible_energy, 0.0):
            raise ValueError(
                "Use either infeasible_energy or global_phase, not both."
            )
        infeasible_energy = global_phase

    paramed_circ = problem.circuit
    circ = paramed_circ.assign_parameters(
        {problem.circuit.parameters[0]: t},
        inplace = False
    )
    energy_fn = problem.energy

    U = Operator(circ).data  # complex ndarray
    n = circ.num_qubits
    d = 2**n
    # Compare diagonal phases to expected, modulo a global phase
    # expected diag entries
    energies = np.zeros(d, dtype=float)
    mask = np.ones(d, dtype=bool)
    for i in range(d):
        energy, include_state = _validation_energy(
            problem,
            _bitstring(i, n, flip=flip),
            infeasible_energy=infeasible_energy,
            omit_infeasible_states=omit_infeasible_states,
        )
        energies[i] = energy
        mask[i] = include_state
    expected = np.exp(-1j * t * energies)
    

    diag = np.diag(U)
    if not np.any(mask):
        return False, {"n_qubits": n, "error": "No states selected for validation."}

    # Remove global phase by aligning first nonzero expected
    ref_idx = int(np.flatnonzero(mask)[0])
    g = diag[ref_idx] / expected[ref_idx]  # global phase factor
    ratios = diag / (expected * g)

    # Errors
    mag_err = np.max(np.abs(np.abs(diag[mask]) - 1.0))
    phase_err = np.max(
        np.abs(np.angle(ratios[mask]))
    )  # max residual phase after removing global
    ok = (mag_err <= rtol) and (phase_err <= atol)

    report = {
        "n_qubits": n,
        "max_magnitude_error": float(mag_err),
        "max_phase_error_rad_after_global": float(phase_err),
        "global_phase_rad": float(np.angle(g)),
    }
    if not ok:
        # include a few worst offenders
        idx_sorted = np.flatnonzero(mask)[np.argsort(-np.abs(np.angle(ratios[mask])))]
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
