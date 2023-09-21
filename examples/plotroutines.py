import matplotlib.pyplot as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable

from qaoa import QAOA


def __plot_landscape(A, extent):
    f = pl.figure(figsize=(6, 6), dpi=80, facecolor="w", edgecolor="k")
    _ = pl.xlabel(r"$\gamma$")
    _ = pl.ylabel(r"$\beta$")
    ax = pl.gca()
    _ = pl.title("Expectation value")
    im = ax.imshow(A, interpolation="bicubic", origin="lower", extent=extent)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    _ = pl.colorbar(im, cax=cax)


def plot_E(qaoa_instance, figsize=(6, 6)):
    angles = qaoa_instance.landscape_p1_angles
    extent = [
        angles["gamma"][0],
        angles["gamma"][1],
        angles["beta"][0],
        angles["beta"][1],
    ]
    __plot_landscape(qaoa_instance.exp_landscape(), extent)


def plot_Var(qaoa_instance, figsize=(6, 6)):
    angles = qaoa_instance.landscape_p1_angles
    extent = [
        angles["gamma"][0],
        angles["gamma"][1],
        angles["beta"][0],
        angles["beta"][1],
    ]
    __plot_landscape(qaoa_instance.var_landscape(), extent)
