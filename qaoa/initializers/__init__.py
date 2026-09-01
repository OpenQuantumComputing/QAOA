"""
QAOA parameter initializers.

Each initializer implements the :class:`Initializer` protocol: given a
``QAOA`` instance and the target depth it returns one or more candidate
angle arrays that are then evaluated and ranked by :meth:`QAOA.optimize`
before the local optimiser starts.

Public classes
--------------
Initializer   – abstract base (protocol)
LayerGrid     – 2-D grid search over the new layer; monotonic guarantee
Interp        – INTERP heuristic (linear interpolation across layers)
LinearRamp    – linearly-spaced angles from 0 to π/4
TQA           – Trotterised Quantum Annealing schedule
Random        – uniformly random angles
FixedAngles   – user-supplied / transferred angles
Fourier       – Fourier (u,v) parameterisation

Documentation table
-------------------
| Initializer  | Purpose                  | Requires prev depth | Monotonic | Extra evals | Ansatz        |
|--------------|--------------------------|---------------------|-----------|-------------|---------------|
| LayerGrid    | Grid over new layer      | yes (depth > 1)     | yes       | O(N²)       | any           |
| Interp       | Interpolate prev angles  | yes                 | no        | 0           | any           |
| LinearRamp   | Linear schedule          | no                  | no        | 0           | any           |
| TQA          | TQA schedule             | no                  | no        | 0           | any           |
| Random       | Random start             | no                  | no        | 0           | any           |
| FixedAngles  | User-supplied angles     | no                  | no        | 0           | any           |
| Fourier      | Fourier u,v params       | no                  | no        | 0           | standard only |

Citations
---------
INTERP / Fourier: Zhou et al., PRX 10, 021067 (2020)
    https://doi.org/10.1103/PhysRevX.10.021067
TQA: Farhi et al., arXiv:2012.06523 (2020)
    https://arxiv.org/abs/2012.06523
General survey: Guo et al., arXiv:2606.05311 (2025)
    https://arxiv.org/abs/2606.05311

SPIQ assessment
---------------
SPIQ (Bharadwaj & Wocjan, arXiv:2602.14327) performs a classical Clifford
search over a relaxed multi-angle QAOA ansatz and returns promising initial
angle candidates.  Its reference implementation
(github.com/d-bharadwaj/SPIQ) does not carry a permissive open-source
license, so code cannot be copied.  A clean integration would require:
  * a Clifford/stabilizer simulator (e.g. stim or qiskit.quantum_info);
  * mapping SPIQ's relaxed-ansatz parameters back to this package's flat
    [init | gamma_0 ... | beta_0 ...] ordering;
  * lazy import of the simulator so it remains optional.
The :class:`Initializer` interface already supports multiple candidates
(list return), so SPIQ would fit without interface changes.  Deferred to a
follow-up PR.
"""

from .base import Initializer
from .layer_grid import LayerGrid
from .interp import Interp
from .linear_ramp import LinearRamp
from .tqa import TQA
from .random import Random
from .fixed_angles import FixedAngles
from .fourier import Fourier

__all__ = [
    "Initializer",
    "LayerGrid",
    "Interp",
    "LinearRamp",
    "TQA",
    "Random",
    "FixedAngles",
    "Fourier",
]
