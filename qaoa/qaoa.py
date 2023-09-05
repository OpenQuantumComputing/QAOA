import numpy as np

from qaoa.mixers import Mixer
from qaoa.problems import Problem

class QAOA:
    """ Main class
    """
    def __init__(self, problem, mixer, params=None) -> None:
        """
        A QAO-Ansatz consist of two parts:

            :param problem of Basetype Problem,
            implementing the phase circuit and the cost.

            :param mixer of Basetype Mixer,
            specifying the mixer circuit and the initial state.

        :params additional parameters

        :param backend: backend
        :param precision: precision to reach for expectation value based on error=variance/sqrt(shots)
        :param shots: if precision=None, the number of samples taken
                      if precision!=None, the minimum number of samples taken

        """

        assert issubclass(problem, Problem)
        assert issubclass(mixer, Mixer)

        self.params = params
        self.problem = problem(params)
        self.mixer = problem(params)

