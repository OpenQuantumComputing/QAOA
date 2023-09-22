import matplotlib.pyplot as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator

import numpy as np

from qaoa import QAOA


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


def plot_Exp(qaoa_instance, maxdepth, mincost, maxcost, label, style="", fig=None):
    if not fig:
        ax = pl.figure().gca()
    else:
        ax = fig.gca()
    pl.hlines(1, 1, maxdepth, linestyles="solid", colors="black")
    pl.plot(
        np.arange(1, maxdepth + 1),
        (maxcost - np.array(qaoa_instance.get_Exp())) / (maxcost - mincost),
        style,
        label=label,
    )
    pl.ylim(0, 1.01)
    pl.xlim(1 - 0.25, maxdepth + 0.25)
    _ = pl.ylabel("appr. ratio")
    _ = pl.xlabel("depth")
    _ = pl.legend(loc="lower right", framealpha=1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
