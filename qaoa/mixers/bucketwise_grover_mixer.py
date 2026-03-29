from __future__ import annotations

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import PhaseGate

from .base_mixer import Mixer


class BucketwiseGrover(Mixer):
    """
    Tensor product of local Grover diffusions on disjoint qubit registers ("buckets").

    On register ``k`` with ``b_k`` qubits, applies the same structure as
    :class:`~qaoa.mixers.grover_mixer.Grover` with uniform ``US = H^{⊗ b_k}``:

    ``H^{⊗ b_k} X^{⊗ b_k} C^{b_k-1}\\text{Phase}(-β) X^{⊗ b_k} H^{⊗ b_k}``,

    i.e. reflection ``2|u_k⟩⟨u_k| - I`` with ``|u_k⟩ = H^{⊗ b_k}|0⟩``, up to
    the usual Qiskit parameter convention shared with ``Grover``.

    Registers are fixed by ``bucket_qubit_offsets`` and ``bucket_qubits``; any
    problem whose cost circuit uses the same layout (e.g.
    :class:`~qaoa.problems.bucketexactcover_problem.BucketExactCover`) can use
    this mixer. For BucketExactCover, prefer :meth:`from_bucket_exact_cover`.
    """

    def __init__(
        self,
        bucket_qubits: list[int],
        bucket_qubit_offsets: list[int] | None = None,
        *,
        label: str | None = None,
    ) -> None:
        """
        Args:
            bucket_qubits: ``b_k`` — number of qubits per bucket, in order.
            bucket_qubit_offsets: Global index of the first qubit of each bucket.
                If *None*, registers are assumed contiguous starting at 0
                (same packing as :class:`~qaoa.problems.bucketexactcover_problem.BucketExactCover`).
            label: Optional circuit name; defaults to class name.
        """
        super().__init__(label=label)
        if not bucket_qubits:
            raise ValueError("bucket_qubits must be non-empty")
        if any(b < 1 for b in bucket_qubits):
            raise ValueError("each bucket must use at least one qubit")

        self._bucket_qubits = list(bucket_qubits)
        if bucket_qubit_offsets is None:
            self._bucket_qubit_offsets = _contiguous_offsets(self._bucket_qubits)
        else:
            self._bucket_qubit_offsets = list(bucket_qubit_offsets)

        _validate_bucket_layout(self._bucket_qubits, self._bucket_qubit_offsets)

        self.mixer_param = Parameter("x_beta")

    @classmethod
    def from_bucket_exact_cover(
        cls,
        problem: BucketExactCover,
        *,
        label: str | None = None,
    ) -> BucketwiseGrover:
        """
        Build from a :class:`~qaoa.problems.bucketexactcover_problem.BucketExactCover`
        instance (reads per-bucket widths and qubit offsets).
        """
        from qaoa.problems.bucketexactcover_problem import BucketExactCover as _BEC

        if not isinstance(problem, _BEC):
            raise TypeError(f"expected BucketExactCover, got {type(problem).__name__}")
        return cls(
            list(problem._bucket_qubits),
            list(problem._bucket_qubit_offsets),
            label=label,
        )

    def create_circuit(self) -> None:
        expected_n = sum(self._bucket_qubits)
        if self.N_qubits != expected_n:
            raise ValueError(
                f"mixer N_qubits ({self.N_qubits}) != sum(bucket_qubits) ({expected_n}); "
                "layout and QAOA problem size must agree."
            )

        widths = self._bucket_qubits
        offsets = self._bucket_qubit_offsets
        num_buckets = len(widths)

        qr = QuantumRegister(self.N_qubits, name="q")
        self.circuit = QuantumCircuit(qr)

        for k in range(num_buckets):
            b_k = widths[k]
            base = offsets[k]
            qs = [base + j for j in range(b_k)]

            self.circuit.h(qs)
            self.circuit.x(qs)
            if b_k == 1:
                self.circuit.append(PhaseGate(-self.mixer_param), [qs[0]])
            else:
                phase_gate = PhaseGate(-self.mixer_param).control(b_k - 1)
                self.circuit.append(phase_gate, qs)
            self.circuit.x(qs)
            self.circuit.h(qs)


def _contiguous_offsets(bucket_qubits: list[int]) -> list[int]:
    offsets: list[int] = []
    acc = 0
    for b in bucket_qubits:
        offsets.append(acc)
        acc += b
    return offsets


def _validate_bucket_layout(widths: list[int], offsets: list[int]) -> None:
    if len(widths) != len(offsets):
        raise ValueError(
            "bucket_qubits and bucket_qubit_offsets must have the same length"
        )
    if offsets[0] != 0:
        raise ValueError("bucket_qubit_offsets[0] must be 0")
    for k in range(len(widths) - 1):
        if offsets[k + 1] != offsets[k] + widths[k]:
            raise ValueError(
                "bucket_qubit_offsets must be contiguous: "
                f"expected offsets[{k+1}] == offsets[{k}] + bucket_qubits[{k}] "
                f"(got {offsets[k + 1]} vs {offsets[k] + widths[k]})"
            )
    last = offsets[-1] + widths[-1]
    if last != sum(widths):
        raise ValueError(
            "bucket layout is inconsistent with total qubit count "
            f"(last index end {last} != sum(bucket_qubits) {sum(widths)})"
        )
