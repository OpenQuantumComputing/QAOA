import math
import pickle
import sys
import os

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from qaoa import QAOA, mixers, initialstates  # type: ignore
from qaoa.initialstates import MaxKCutFeasible
from qaoa.mixers import MaxKCutGrover, MaxKCutLX, XYTensor
from qaoa.problems import MaxKCutBinaryPowerOfTwo, MaxKCutBinaryFullH

from qiskit_algorithms.optimizers import SPSA, COBYLA, ADAM, NFT, NELDER_MEAD

from qiskit_aer import AerSimulator


def main(
        method,
        k,
        clf,
        mixerstr,
        casename,
        maxdepth,
        shots,
        ):

    angles = {"gamma": [0, 2 * np.pi, 20], "beta": [0, 2 * np.pi, 20]}
    optimizer = [COBYLA, {"maxiter": 100, "tol": 1e-3, "rhobeg": 0.05}]
    problem_encoding = "binary"

    if casename == "Barbell":
        V = np.arange(0, 2, 1)
        E = [(0, 1, 1.0)]
        G = nx.Graph()
        G.add_nodes_from(V)
        G.add_weighted_edges_from(E)
    elif casename == "BarabasiAlbert":
        G = nx.read_gml("data/w_ba_n10_k4_0.gml")
        # max_val = np.array([8.657714089848158, 10.87975400338161, 11.059417685176726, 11.059417685176726, 11.059417685176726, 11.059417685176726, 11.059417685176726])
    elif casename == "ErdosRenyi":
        G = nx.read_gml("data/er_n10_k4_0.gml")
        # max_val = np.array([12, 16, 16, 16, 16, 16, 16])

    string_identifier = (
        "method"
        + str(method)
        + "_"
        + "k"
        + str(k)
        + "_"
        + "clf"
        + str(clf)
        + "_"
        + "mixer"
        + str(mixerstr)
        + "_"
        "casename"
        + str(casename)
        + "_"
        + "shots"
        + str(shots)
    )
    print("Now running", string_identifier)

    if k == 3:
        kf = 4
    elif k in [5,6,7]:
        kf = 8

    if method == "fullH":
        problem = MaxKCutBinaryFullH(
            G,
            k,
            color_encoding=clf,
        )

        if mixerstr == "X":
            mixer = mixers.X()
        else:
            mixer = MaxKCutGrover(
                kf,
                problem_encoding=problem_encoding,
                color_encoding="all",
                tensorized=True,
            )

        initialstate = initialstates.Plus()

    else:
        problem = MaxKCutBinaryPowerOfTwo(
            G,
            kf,
        )

        if mixerstr == "LX":
            mixer = MaxKCutLX(k, color_encoding="LessThanK")
        elif mixerstr == "Grover":
            mixer = MaxKCutGrover(
                k,
                problem_encoding=problem_encoding,
                color_encoding="LessThanK",
                tensorized=False,
            )
        else:
            mixer = MaxKCutGrover(
                k,
                problem_encoding=problem_encoding,
                color_encoding="LessThanK",
                tensorized=True,
            )
        initialstate = MaxKCutFeasible(
            k, problem_encoding=problem_encoding, color_encoding="LessThanK"
        )

    fn = string_identifier + ".pickle"
    if os.path.exists(fn):
        try:
            with open(fn, "rb") as f:
                qaoa = pickle.load(f)
        except ValueError:
            print("file exists, but can not open it", fn)
    else:
        qaoa = QAOA(
            problem=problem,
            initialstate=initialstate,
            mixer=mixer,
            backend=AerSimulator(method="automatic", device="GPU"),
            shots=shots,
            optimizer=optimizer,
            sequential=True,
        )

    qaoa.optimize(maxdepth, angles=angles)

    with open(fn, "wb") as f:
        pickle.dump(qaoa, f)


if __name__ == "__main__":

    method = str(sys.argv[1])
    k = int(sys.argv[2])
    clf = str(sys.argv[3])
    mixerstr = str(sys.argv[4])
    casename = str(sys.argv[5])
    maxdepth = int(sys.argv[6])
    shots = int(sys.argv[7])

    main(
        method,
        k,
        clf,
        mixerstr,
        casename,
        maxdepth,
        shots,
    )
