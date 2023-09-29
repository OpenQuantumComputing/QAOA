import matplotlib.pyplot as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator

import numpy as np

from qaoa import QAOA
from qaoa.mixers.constrained_mixer import Constrained

from qaoa.util import Statistic

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


def plot_ApproximationRatio(
    qaoa_instance, maxdepth, mincost, maxcost, label, style="", fig=None, shots=None
):
    if not shots:
        exp = np.array(qaoa_instance.get_Exp())
    else:
        exp = []
        for p in range(1, qaoa_instance.current_depth + 1):
            ar, sp = __apprrat_successprob(qaoa_instance, p, shots=shots)
            exp.append(ar)
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
    qaoa_instance, maxdepth, label, style="", fig=None, shots=10**4
):
    successp = []
    for p in range(1, qaoa_instance.current_depth + 1):
        ar, sp = __apprrat_successprob(qaoa_instance, p, shots=shots)
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
    _ = pl.ylabel("success prob")
    _ = pl.xlabel("depth")
    _ = pl.legend(loc="lower right", framealpha=1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def __apprrat_successprob(qaoa_instance, depth, shots=10**4):
    """
    approximation ratio post processed with feasibility and success probability
    """
    hist = qaoa_instance.hist(
        qaoa_instance.optimization_results[depth].get_best_angles(), shots=shots
    )

    counts = 0

    stat = Statistic(cvar=qaoa_instance.cvar)

    for key in hist:
        # Qiskit uses big endian encoding, cost function uses litle endian encoding.
        # Therefore the string is reversed before passing it to the cost function.
        string = key[::-1]
        if qaoa_instance.problem.isFeasible(string):
            cost = qaoa_instance.problem.cost(string)
            counts += hist[key]
            stat.add_sample(cost, hist[key])

    return -stat.get_CVaR(), counts / shots


def plot_angles(qaoa_instance, depth, label, style="", fig=None):
    angles=qaoa_instance.optimization_results[depth].get_best_angles()

    if not fig:
        ax = pl.figure().gca()
    else:
        ax = fig.gca()

    pl.plot(
        np.arange(1, depth + 1),
        angles[::2],
        "--"+style,
        label=r"$\gamma$ " + label,
    )
    pl.plot(
        np.arange(1, depth + 1),
        angles[1::2],
        "-"+style,
        label=r"$\beta$ " + label,
    )
    pl.xlim(1 - 0.25, depth + 0.25)
    _ = pl.ylabel("parameter")
    _ = pl.xlabel("depth")
    _ = pl.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
