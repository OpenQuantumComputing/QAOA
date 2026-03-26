import math
import itertools
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from .base_problem import Problem


class BucketExactCover(Problem):
    """
    Bucket Exact Cover (HUBO) problem.

    Subclass of the `Problem` class. Implements the binary-encoded exact cover
    problem where routes are grouped by boat (bucket), enabling a HUBO
    formulation that requires fewer qubits than the one-hot QUBO.

    Each bucket k corresponds to one boat and contains all routes for that boat.
    A b_k-bit binary string selects exactly one route from bucket k via modular
    wrapping (Strategy B): `idx = v % n_k` where `v` is the integer value of
    the binary string. This guarantees one route is always selected per bucket.

    Attributes:
        columns (np.ndarray): Full column matrix (N_rows × N_routes).
        num_buckets (int): Number of top rows used for bucket assignment.
        weights (np.ndarray): (Possibly scaled) weights for valid columns.
        penalty_factor (float): (Possibly scaled) penalty factor.
        scale_problem (bool): Whether the problem has been scaled.
        allow_infeasible (bool): Whether infeasible solutions are accepted.
        N_qubits (int): Total number of qubits.
        upper_bound_scaling (float): Upper bound used for scaling (defaults to 1.0)

    See Also:
        :meth:`get_encoding_degeneracy_stats` for modular wrap multiplicities per bucket.
    """

    def __init__(
        self,
        columns,
        num_buckets,
        weights=None,
        penalty_factor=None,
        allow_infeasible=False,
        scale_problem=False,
        upper_bound_scaling=1.0,
        penalty_factor_scaling=1.0
    ) -> None:
        super().__init__()

        self.columns = np.array(columns, dtype=float)
        self.num_buckets = num_buckets
        self.allow_infeasible = allow_infeasible
        self.scale_problem = scale_problem
        self.upper_bound_scaling = upper_bound_scaling
        self.penalty_factor_scaling = penalty_factor_scaling

        N_rows, N_routes = self.columns.shape
        n_orders = N_rows - num_buckets

        # --- Bucket structure ------------------------------------------------
        self._bucket_columns = []
        for k in range(num_buckets):
            bucket_k = [r for r in range(N_routes) if self.columns[k, r] == 1]
            self._bucket_columns.append(bucket_k)

        # Valid columns: those belonging to at least one bucket
        valid_set = set()
        for k in range(num_buckets):
            valid_set.update(self._bucket_columns[k])
        self._valid_columns = sorted(valid_set)
        self._valid_col_to_idx = {col: idx for idx, col in enumerate(self._valid_columns)}

        # Each bucket k has n_k columns; we need ceil(log2(n_k)) qubits to encode a choice among them.
        # Use at least 1 qubit (when n_k=1, log2(1)=0).
        self._bucket_sizes = [len(bc) for bc in self._bucket_columns]
        self._bucket_qubits = [
            max(1, math.ceil(math.log2(n_k))) for n_k in self._bucket_sizes
        ]
        self._bucket_qubit_offsets = []
        offset = 0
        for b_k in self._bucket_qubits:
            self._bucket_qubit_offsets.append(offset)
            offset += b_k
        self.N_qubits = offset

        # --- Weights ---------------------------------------------------------
        if weights is None:
            orig_weights = np.zeros(len(self._valid_columns))
        else:
            orig_weights = np.array(weights, dtype=float)[self._valid_columns]
        self._original_weights = orig_weights.copy()
        self.weights = orig_weights.copy()

        # --- Per-bucket weight statistics ------------------------------------
        bucket_mins = []
        bucket_maxs = []
        for k in range(num_buckets):
            if self._bucket_sizes[k] > 0:
                w_k = [
                    self._original_weights[self._valid_col_to_idx[g]]
                    for g in self._bucket_columns[k]
                ]
                bucket_mins.append(min(w_k))
                bucket_maxs.append(max(w_k))
            else:
                bucket_mins.append(0.0)
                bucket_maxs.append(0.0)

        C_min = sum(bucket_mins)
        C_max_shifted = sum(bmax - bmin for bmax, bmin in zip(bucket_maxs, bucket_mins))
        epsilon = max(1e-6, 1e-6 * C_max_shifted)

        # --- Penalty factor --------------------------------------------------
        if penalty_factor is None:
            if np.all(self._original_weights == 0):
                unscaled_penalty = 1.0
            else:
                unscaled_penalty = C_max_shifted + epsilon
        else:
            unscaled_penalty = float(penalty_factor)

        self._original_penalty_factor = unscaled_penalty
        self.penalty_factor = unscaled_penalty

        # --- Scaling ---------------------------------------------------------
        if scale_problem and C_max_shifted > 0:
            shift = C_min / num_buckets
            omega = self._original_weights - shift

            lam = C_max_shifted + epsilon
            denom = C_max_shifted * (1 + n_orders * (num_buckets - 1) ** 2)
            s = self.upper_bound_scaling / denom

            self.weights = s * omega
            self.penalty_factor = s * lam * self.penalty_factor_scaling

    # -------------------------------------------------------------------------
    # Decoding
    # -------------------------------------------------------------------------

    def _decode(self, string):
        """Decode a bitstring to a binary indicator vector over valid columns.

        Modular wrapping: idx = v % n_k always selects one route.
        """
        x = np.zeros(len(self._valid_columns))
        for k in range(self.num_buckets):
            offset = self._bucket_qubit_offsets[k]
            b_k = self._bucket_qubits[k]
            n_k = self._bucket_sizes[k]
            v = sum(int(string[offset + j]) * (1 << j) for j in range(b_k))
            idx = v % n_k
            global_col = self._bucket_columns[k][idx]
            valid_idx = self._valid_col_to_idx[global_col]
            x[valid_idx] = 1.0
        return x

    def get_encoding_degeneracy_stats(self):
        """
        Degeneracy from Strategy B encoding: ``n_k`` routes share ``2^{b_k}`` bit patterns.

        For bucket ``k``, integer values ``v ∈ {0, …, 2^{b_k}-1}`` map to local route index
        ``v % n_k``. Multiple ``v`` can select the same column (same decoded route).

        Returns:
            dict with keys:

            * ``per_bucket`` — list of ``dict`` (one per bucket) with:

              - ``bucket_index`` (int)
              - ``n_routes`` — ``n_k`` (distinct columns in the bucket)
              - ``num_qubits`` — ``b_k``
              - ``num_encoded_states`` — ``2^{b_k}``
              - ``multiplicities`` — list of length ``n_k``; ``multiplicities[j]`` is the
                number of encoded bit-patterns with ``v % n_k == j`` (same global column)
              - ``redundant_encodings`` — ``num_encoded_states - n_routes``; encodings beyond
                a single bit-pattern per route index (equals ``sum(m-1)`` over multiplicities)
              - ``global_column_indices`` — list of route column indices ``0 … N_routes-1`` in
                this bucket (same order as multiplicities / local indices)

            * ``total_encoded_bitstrings`` — product of ``2^{b_k}`` (full HUBO register size)
            * ``total_decoded_assignments`` — product of ``n_k`` (distinct decoded route tuples)
            * ``excess_encodings_over_assignments`` —
              ``total_encoded_bitstrings - total_decoded_assignments`` (aggregate “extra”
              bitstrings versus counting one representative per decoded choice)
            * ``mean_encoded_per_assignment`` —
              ``total_encoded_bitstrings / total_decoded_assignments``
        """
        per_bucket = []
        prod_encoded = 1
        prod_decoded = 1

        for k in range(self.num_buckets):
            n_k = self._bucket_sizes[k]
            b_k = self._bucket_qubits[k]
            if n_k <= 0:
                encoded = 1 << b_k
                entry = {
                    "bucket_index": k,
                    "n_routes": 0,
                    "num_qubits": b_k,
                    "num_encoded_states": encoded,
                    "multiplicities": [],
                    "redundant_encodings": encoded,
                    "global_column_indices": list(self._bucket_columns[k]),
                }
                per_bucket.append(entry)
                prod_encoded *= encoded
                continue

            encoded = 1 << b_k
            mult = []
            for idx in range(n_k):
                count = (encoded - 1 - idx) // n_k + 1
                mult.append(int(count))
            redundant = encoded - n_k

            prod_encoded *= encoded
            prod_decoded *= n_k

            per_bucket.append(
                {
                    "bucket_index": k,
                    "n_routes": n_k,
                    "num_qubits": b_k,
                    "num_encoded_states": encoded,
                    "multiplicities": mult,
                    "redundant_encodings": redundant,
                    "global_column_indices": list(self._bucket_columns[k]),
                }
            )

        excess = prod_encoded - prod_decoded
        return {
            "per_bucket": per_bucket,
            "total_encoded_bitstrings": int(prod_encoded),
            "total_decoded_assignments": int(prod_decoded),
            "excess_encodings_over_assignments": int(excess),
            "mean_encoded_per_assignment": float(prod_encoded) / float(prod_decoded),
        }

    # -------------------------------------------------------------------------
    # Cost functions
    # -------------------------------------------------------------------------

    def _compute_cost(self, x, w, pf):
        n_orders = self.columns.shape[0] - self.num_buckets
        penalty = 0.0
        if n_orders > 0:
            Y_orders = self.columns[self.num_buckets :, :][:, self._valid_columns]
            coverage = Y_orders @ x
            penalty = np.sum((coverage - 1) ** 2)
        return -(w @ x + pf * penalty)

    def cost(self, string):
        """
        Calculates the (possibly scaled) cost of a given solution.

        Args:
            string (str): Bitstring representing a candidate solution.

        Returns:
            float: Negated cost (QAOA maximizes).
        """
        x = self._decode(string)
        return self._compute_cost(x, self.weights, self.penalty_factor)

    def unscaled_cost(self, string):
        """
        Calculates the original unscaled cost of a given solution.

        Args:
            string (str): Bitstring representing a candidate solution.

        Returns:
            float: Negated unscaled cost.
        """
        x = self._decode(string)
        return self._compute_cost(x, self._original_weights, self._original_penalty_factor)

    def isFeasible(self, string):
        """
        Checks if a bitstring represents a feasible solution.

        Args:
            string (str): Bitstring representing a candidate solution.

        Returns:
            bool: True if feasible.
        """
        if self.allow_infeasible:
            return True
        n_orders = self.columns.shape[0] - self.num_buckets
        if n_orders == 0:
            return True
        x = self._decode(string)
        Y_orders = self.columns[self.num_buckets :, :][:, self._valid_columns]
        coverage = Y_orders @ x
        return np.allclose(coverage, np.ones(n_orders), atol=1e-7)

    # -------------------------------------------------------------------------
    # HUBO polynomial helpers
    # -------------------------------------------------------------------------

    def _mobius_inversion(self, f_vals, k):
        """
        Compute multilinear polynomial coefficients from function values via
        Möbius inversion on the Boolean lattice.

        f_vals[v] = f(z) where v = sum_j 2^j * z_j.

        Returns:
            dict mapping frozenset(local_qubit_indices) -> coefficient
        """
        result = {}
        for size in range(k + 1):
            for T in combinations(range(k), size):
                alpha = 0.0
                for u_size in range(size + 1):
                    for U in combinations(T, u_size):
                        v = sum(1 << j for j in U)
                        sign = (-1) ** (size - u_size)
                        alpha += sign * f_vals[v]
                if abs(alpha) > 1e-15:
                    result[frozenset(T)] = alpha
        return result

    # -------------------------------------------------------------------------
    # Circuit construction
    # -------------------------------------------------------------------------

    def create_circuit(self):
        """
        Creates the QAOA cost (phase separator) circuit for the HUBO problem.

        Builds the HUBO polynomial via multilinear interpolation, substitutes
        q_m = (1 - Z_m)/2, and implements multi-qubit Z-rotations using
        CNOT-ladder + RZ + reverse-CNOT-ladder.

        Returns:
            QuantumCircuit: Parameterized circuit with parameter "x_gamma".
        """
        n_orders = self.columns.shape[0] - self.num_buckets
        Y_orders = self.columns[self.num_buckets :, :] if n_orders > 0 else None

        # Accumulated HUBO polynomial: frozenset(global qubit indices) -> coeff
        hubo_poly = {}

        def add_to_poly(T_key, coeff):
            if abs(coeff) < 1e-15:
                return
            hubo_poly[T_key] = hubo_poly.get(T_key, 0.0) + coeff

        # Coverage polynomials per (order, bucket)
        cov_polys = {}

        for k in range(self.num_buckets):
            b_k = self._bucket_qubits[k]
            n_k = self._bucket_sizes[k]
            offset = self._bucket_qubit_offsets[k]
            num_states = 1 << b_k

            # Cost contribution values per state
            cost_vals = np.zeros(num_states)
            for v in range(num_states):
                idx = v % n_k
                global_col = self._bucket_columns[k][idx]
                valid_idx = self._valid_col_to_idx[global_col]
                cost_vals[v] = self.weights[valid_idx]

            # Möbius inversion for cost
            cost_alpha = self._mobius_inversion(cost_vals, b_k)
            for T_local, coeff in cost_alpha.items():
                T_global = frozenset(offset + j for j in T_local)
                add_to_poly(T_global, coeff)

            # Coverage values per (order, state)
            if n_orders > 0:
                for o in range(n_orders):
                    cov_vals = np.zeros(num_states)
                    for v in range(num_states):
                        idx = v % n_k
                        global_col = self._bucket_columns[k][idx]
                        cov_vals[v] = float(Y_orders[o, global_col])
                    cov_alpha = self._mobius_inversion(cov_vals, b_k)
                    cov_polys[(o, k)] = {}
                    for T_local, coeff in cov_alpha.items():
                        T_global = frozenset(offset + j for j in T_local)
                        cov_polys[(o, k)][T_global] = coeff

        # Penalty terms: penalty_factor * sum_o P_o
        # P_o = 1 - sum_k g_{o,k} + 2 * sum_{k<k'} g_{o,k} * g_{o,k'}
        if n_orders > 0:
            for o in range(n_orders):
                # Constant term: +1
                add_to_poly(frozenset(), self.penalty_factor)

                # -sum_k g_{o,k}
                for k in range(self.num_buckets):
                    for T, coeff in cov_polys.get((o, k), {}).items():
                        add_to_poly(T, -self.penalty_factor * coeff)

                # +2 * sum_{k<k'} g_{o,k} * g_{o,k'}
                for k in range(self.num_buckets):
                    for k2 in range(k + 1, self.num_buckets):
                        for T1, c1 in cov_polys.get((o, k), {}).items():
                            for T2, c2 in cov_polys.get((o, k2), {}).items():
                                T = T1 | T2  # qubit sets are disjoint
                                add_to_poly(T, 2.0 * self.penalty_factor * c1 * c2)

        # Substitute q_m = (1 - Z_m) / 2 to get Pauli-Z coefficients.
        # prod_{m in T} q_m = (1/2^|T|) * sum_{U ⊆ T} (-1)^|U| prod_{m in U} Z_m
        beta = {}  # frozenset(global qubit indices) -> Pauli-Z coefficient

        for T, alpha in hubo_poly.items():
            if abs(alpha) < 1e-15:
                continue
            T_list = sorted(T)
            d = len(T_list)
            factor = alpha / (2 ** d)
            for u_size in range(d + 1):
                for U in combinations(T_list, u_size):
                    U_key = frozenset(U)
                    sign = (-1) ** u_size
                    beta[U_key] = beta.get(U_key, 0.0) + sign * factor

        # Build Qiskit circuit
        gamma = Parameter("x_gamma")
        circuit = QuantumCircuit(self.N_qubits)

        for T_key, b_T in beta.items():
            if abs(b_T) < 1e-10:
                continue
            T_list = sorted(T_key)
            d = len(T_list)
            if d == 0:
                continue  # constant term — no gate needed
            elif d == 1:
                circuit.rz(2 * gamma * b_T, T_list[0])
            else:
                # CNOT ladder
                for i in range(d - 1):
                    circuit.cx(T_list[i], T_list[i + 1])
                circuit.rz(2 * gamma * b_T, T_list[d - 1])
                # Reverse CNOT ladder
                for i in range(d - 2, -1, -1):
                    circuit.cx(T_list[i], T_list[i + 1])

        self.circuit = circuit
        return circuit

    # -------------------------------------------------------------------------
    # Brute force
    # -------------------------------------------------------------------------

    def brute_force_solve(self, return_num_feasible=False):
        """
        Find the optimal solution by exhaustive enumeration over all bucket
        value combinations.

        Args:
            return_num_feasible (bool): If True, also return the count of
                feasible solutions encountered.

        Returns:
            str or (str, int): Optimal bitstring, and optionally the number of
                feasible solutions.
        """
        bucket_ranges = [range(1 << self._bucket_qubits[k]) for k in range(self.num_buckets)]

        best_val = -np.inf
        best_sol = None
        num_feasible = 0

        for combo in itertools.product(*bucket_ranges):
            parts = []
            for k, v in enumerate(combo):
                b_k = self._bucket_qubits[k]
                bits = "".join(str((v >> j) & 1) for j in range(b_k))
                parts.append(bits)
            bitstring = "".join(parts)

            val = self.cost(bitstring)
            if val > best_val:
                best_val = val
                best_sol = bitstring
            if self.isFeasible(bitstring):
                num_feasible += 1

        if return_num_feasible:
            return best_sol, num_feasible
        return best_sol


    # -------------------------------------------------------------------------
    # Decode histogram
    # -------------------------------------------------------------------------

    def decode_histogram(self, encoded_hist: dict) -> dict:
        """Collapse encoded histogram to decoded (ExactCover-format) keys.
        
        Multiple encoded bitstrings can map to the same column selection via modular
        wrapping; their counts are summed.
        
        Returns:
            dict: Keys are one-hot strings over all columns (ExactCover format),
                values are combined hit counts.
        """
        decoded = {}
        N_routes = self.columns.shape[1]
        for enc_bs, count in encoded_hist.items():
            x = self._decode(enc_bs)
            full = np.zeros(N_routes)
            for valid_idx in range(len(x)):
                if x[valid_idx]:
                    full[self._valid_columns[valid_idx]] = 1
            key = "".join(str(int(b)) for b in full)
            decoded[key] = decoded.get(key, 0) + count
        return decoded


    def preprocess_histogram(self, hist: dict) -> dict:
        """
        Decode histogram keys from HUBO bitstrings to one-hot column format.

        Multiple encoded bitstrings can map to the same column selection via
        modular wrapping; their counts are summed. The returned histogram uses
        keys compatible with ExactCover (one-hot over all columns), enabling
        direct comparison with QUBO-based exact cover results.

        Args:
            hist (dict): Raw histogram with encoded HUBO bitstrings as keys.

        Returns:
            dict: Histogram with one-hot keys and combined hit counts.
        """
        return self.decode_histogram(hist)