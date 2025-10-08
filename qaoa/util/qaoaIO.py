from qaoa import QAOA

import json
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Type
from enum import Enum
import os
import subprocess
from datetime import datetime
import platform

# ---------- Utility functions ----------

def _numpy_to_list(obj):
    """Recursively convert numpy arrays and Enums to JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Enum):
        return obj.name
    if isinstance(obj, dict):
        return {k: _numpy_to_list(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_numpy_to_list(v) for v in obj]
    return obj


def _list_to_numpy(obj):
    """Convert lists to numpy arrays recursively (for problem reconstruction)."""
    if isinstance(obj, dict):
        return {k: _list_to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list) and all(isinstance(v, (int, float)) for v in obj):
        return np.array(obj)
    if isinstance(obj, list):
        return [_list_to_numpy(v) for v in obj]
    return obj


# ---------- Problem Data Base Class ----------

@dataclass
class ProblemData:
    """Base class for all problem types."""
    problem_type: str = "Base"

    def to_dict(self):
        d = asdict(self)
        d["problem_type"] = self.problem_type
        return _numpy_to_list(d)

    @classmethod
    def from_dict(cls, data: dict) -> "ProblemData":
        """Factory method that picks the right subclass based on problem_type."""
        problem_type = data.get("problem_type", "Base")
        if problem_type not in problem_registry:
            raise ValueError(f"Unknown problem type: {problem_type}")
        subclass = problem_registry[problem_type]
        data = _list_to_numpy(data)
        data.pop("problem_type", None)
        return subclass(**data)


# ---------- Problem Data Subclasses ----------

@dataclass
class ExactCoverProblemData(ProblemData):
    columns: np.ndarray = None
    weights: np.ndarray = None
    solution: np.ndarray = None
    hamming_weight: int = None
    problem_type: str = "ExactCover"


@dataclass
class PortfolioOptimizationProblemData(ProblemData):
    risk: float = 0.0
    exp_returns: np.ndarray = None
    cov_matrix: np.ndarray = None
    budget: int = 0
    problem_type: str = "PortfolioOptimization"


# Register available problem types
problem_registry: Dict[str, Type[ProblemData]] = {
    "ExactCover": ExactCoverProblemData,
    "PortfolioOptimization": PortfolioOptimizationProblemData,
}


# ---------- QAOA-related classes ----------

class InitMethod(Enum):
    PLUS = "PLUS"
    DICKE = "DICKE"


class MixerMethod(Enum):
    X = "X"
    GROVER = "GROVER"
    XYCHAIN = "XYCHAIN"
    XYRING = "XYRING"


@dataclass
class DepthResult:
    optimal_angles: List[float]
    histogram: Dict[str, int]
    opt_time: float # runtime in seconds


@dataclass
class QAOAParameters:
    cvar: float
    init_method: InitMethod
    mixer_method: MixerMethod
    backend: str
    optimizer: str
    N_qubits: int
    #depths: List[DepthResult] = field(default_factory=list)
    depths: Dict[int, DepthResult] = field(default_factory=dict)

@dataclass
class QAOAResult:
    problem: ProblemData
    qaoa_params: QAOAParameters
    metadata: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Automatically populate metadata if not provided."""
        if not self.metadata:
            self.metadata = self._generate_metadata()



    def save(self, filename: str):
        """Save result (including problem type info) to JSON file."""
        data = {
            "problem": self.problem.to_dict(),
            "qaoa_params": _numpy_to_list(asdict(self.qaoa_params)),
            "metadata": self.metadata
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filename: str) -> "QAOAResult":
        """Load result and reconstruct correct ProblemData subclass."""
        with open(filename, "r") as f:
            data = json.load(f)

        # Rebuild problem instance from its dict
        problem = ProblemData.from_dict(data["problem"])

        #depths = [DepthResult(**d) for d in data["qaoa_params"]["depths"]]
        depths_data = data["qaoa_params"]["depths"]
        depths = {int(k): DepthResult(**v) for k, v in depths_data.items()}

        qaoa_params = QAOAParameters(
            cvar=data["qaoa_params"]["cvar"],
            init_method=InitMethod[data["qaoa_params"]["init_method"]],
            mixer_method=MixerMethod[data["qaoa_params"]["mixer_method"]],
            backend=data["qaoa_params"]["backend"],
            optimizer=data["qaoa_params"]["optimizer"],
            N_qubits=data["qaoa_params"]["N_qubits"],
            depths=depths,
        )

        return cls(problem=problem, qaoa_params=qaoa_params, metadata=data.get("metadata", {}))

    def _generate_metadata(self) -> dict:
        meta = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "system": platform.system(),
            "release": platform.release(),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        }

        # ---- Determine QAOA repo root from the location of this file ----
        qaoaIO_dir = os.path.dirname(__file__)
        meta["qaoa_repo_dir"] = qaoaIO_dir

        try:            
            commit_hash = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(qaoaIO_dir),
                    stderr=subprocess.DEVNULL
                )
                .decode("utf-8")
                .strip()
            )
            meta["qaoa_git_commit"] = commit_hash
        except Exception:
            meta["qaoa_git_commit"] = "unknown"

        return meta
    
    @classmethod
    def from_qaoa(cls, qaoa: QAOA, 
                  hist_shots: int = 1024,
                  solution: str = None) -> "QAOAResult":
        
        depths = {}
        for k in qaoa.optimization_results:
            depths[k] = DepthResult(
                qaoa.optimization_results[k].get_best_angles(),
                qaoa.optimization_results[k].opt_time, 
                qaoa.hist(qaoa.optimization_results[k].get_best_angles(), hist_shots)
            )

        problem_data = ExactCoverProblemData(
            columns = qaoa.problem.columns,
            weights = qaoa.problem.weights,
            solution = solution,
            hamming_weight = qaoa.problem.hamming_weight
        )

        init_method = InitMethod(str(qaoa.initialstate).split(" ")[0].split(".")[-1].upper())
        mixer_str = str(qaoa.mixer).split(" ")[0].split(".")[-1].upper()
        mixer_method = None
        if mixer_str == "XY":
            if qaoa.mixer.case == "ring":
                mixer_method = MixerMethod.XYRING
            elif qaoa.mixer.case == "chain":
                mixer_method = MixerMethod.XYCHAIN
        else:
            mixer_method = MixerMethod(mixer_str)

        assert("COBYLA" in str(qaoa.optimizer)), "unsupported optimizer "+str(qaoa.optimizer)+" in qaoaIO "
        optimizer = "COBYLA"

        qaoa_params = QAOAParameters(
            cvar = qaoa.cvar,
            init_method = init_method,
            mixer_method = mixer_method,
            backend = qaoa.backend.name,
            optimizer = optimizer,
            N_qubits = qaoa.problem.N_qubits,
            depths = depths
        )

        return cls(problem=problem_data, qaoa_params=qaoa_params)
    
    # TODO: Implement
    # def generate_qaoa_object(self) -> "QAOA"
