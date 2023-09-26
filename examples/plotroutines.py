import matplotlib.pyplot as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator

import numpy as np

from qaoa import QAOA
from qaoa.mixers.constrained_mixer import Constrained


def __plot_landscape(A, extent, fig):
    if not fig:
        fig = pl.figure(figsize=(6, 6), dpi=80, facecolor="w", edgecolor="k")
    _ = pl.xlabel(r"$\gamma$")
    _ = pl.ylabel(r"$\beta$")
    ax = fig.gca()
    _ = pl.title("Expectation value")
    im = ax.imshow(A, interpolation="bicubic", origin="lower", extent=extent)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    _ = pl.colorbar(im, cax=cax)


def plot_E(qaoa_instance, fig=None):
    angles = qaoa_instance.landscape_p1_angles
    extent = [
        angles["gamma"][0],
        angles["gamma"][1],
        angles["beta"][0],
        angles["beta"][1],
    ]
    return __plot_landscape(qaoa_instance.exp_landscape(), extent, fig=fig)


def plot_Var(qaoa_instance, fig=None):
    angles = qaoa_instance.landscape_p1_angles
    extent = [
        angles["gamma"][0],
        angles["gamma"][1],
        angles["beta"][0],
        angles["beta"][1],
    ]
    return __plot_landscape(qaoa_instance.var_landscape(), extent, fig=fig)


def plot_Exp(
    qaoa_instance, maxdepth, mincost, maxcost, label, style="", fig=None, shots=None
):
    if not shots:
        exp = np.array(qaoa_instance.get_Exp())
    else:
        exp = []
        for p in range(1, qaoa_instance.current_depth + 1):
            alpha, sp = __apprpostproc_successprob(qaoa_instance, p, shots=shots)
            exp.append(alpha)
        exp = np.array(exp)

    if not fig:
        ax = pl.figure().gca()
    else:
        ax = fig.gca()
    pl.hlines(1, 1, maxdepth, linestyles="solid", colors="black")
    pl.plot(
        np.arange(1, maxdepth + 1),
        (maxcost - exp) / (maxcost - mincost),
        style,
        label=label,
    )
    pl.ylim(0, 1.01)
    pl.xlim(1 - 0.25, maxdepth + 0.25)
    _ = pl.ylabel("appr. ratio")
    _ = pl.xlabel("depth")
    _ = pl.legend(loc="lower right", framealpha=1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_successprob(
    qaoa_instance, maxdepth, mincost, maxcost, label, style="", fig=None, shots=10**4
):
    successp = []
    for p in range(1, qaoa_instance.current_depth + 1):
        alpha, sp = __apprpostproc_successprob(qaoa_instance, p, shots=shots)
        successp.append(sp)
    successp = np.array(successp)

    if not fig:
        ax = pl.figure().gca()
    else:
        ax = fig.gca()
    pl.hlines(1, 1, maxdepth, linestyles="solid", colors="black")
    pl.plot(
        np.arange(1, maxdepth + 1),
        successp,
        style,
        label=label,
    )
    pl.ylim(0, 1.01)
    pl.xlim(1 - 0.25, maxdepth + 0.25)
    _ = pl.ylabel("appr. ratio")
    _ = pl.xlabel("depth")
    _ = pl.legend(loc="lower right", framealpha=1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def __apprpostproc_successprob(qaoa_instance, depth, shots=10**4):
    """
    approximation ratio post processed with feasibility and success probability
    """
    hist = qaoa_instance.hist(
        qaoa_instance.optimization_results[depth].get_best_angles(), shots=shots
    )

    ratio = 0
    counts = 0

    for key in hist:
        # Qiskit uses big endian encoding, cost function uses litle endian encoding.
        # Therefore the string is reversed before passing it to the cost function.
        string = key[::-1]
        if qaoa_instance.problem.isFeasible(string):
            ratio -= qaoa_instance.problem.cost(string) * hist[key]
            counts += hist[key]
    return ratio / counts, counts / shots
