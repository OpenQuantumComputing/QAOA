import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator

import numpy as np

from qaoa import QAOA

from qaoa.util import Statistic


def __plot_landscape(A, extent, fig):
    if not fig:
        fig = plt.figure(figsize=(6, 6), dpi=80, facecolor="w", edgecolor="k")
    _ = plt.xlabel(r"$\gamma$")
    _ = plt.ylabel(r"$\beta$")
    ax = fig.gca()
    _ = plt.title("Expectation value")
    im = ax.imshow(A, interpolation="bicubic", origin="lower", extent=extent)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    _ = plt.colorbar(im, cax=cax)


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
        ax = plt.figure().gca()
    else:
        ax = fig.gca()
    plt.hlines(1, 1, maxdepth, linestyles="solid", colors="black")
    plt.plot(
        np.arange(1, maxdepth + 1),
        (maxcost - exp) / (maxcost - mincost),
        style,
        label=label,
    )
    plt.ylim(0, 1.01)
    plt.xlim(1 - 0.25, maxdepth + 0.25)
    _ = plt.ylabel("appr. ratio")
    _ = plt.xlabel("depth")
    _ = plt.legend(loc="lower right", framealpha=1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_successprob(qaoa_instance, maxdepth, label, style="", fig=None, shots=10**4):
    successp = []
    for p in range(1, qaoa_instance.current_depth + 1):
        ar, sp = __apprrat_successprob(qaoa_instance, p, shots=shots)
        successp.append(sp)
    successp = np.array(successp)

    if not fig:
        ax = plt.figure().gca()
    else:
        ax = fig.gca()
    plt.hlines(1, 1, maxdepth, linestyles="solid", colors="black")
    plt.plot(
        np.arange(1, maxdepth + 1),
        successp,
        style,
        label=label,
    )
    plt.ylim(0, 1.01)
    plt.xlim(1 - 0.25, maxdepth + 0.25)
    _ = plt.ylabel("success prob")
    _ = plt.xlabel("depth")
    _ = plt.legend(loc="lower right", framealpha=1)
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

    for string in hist:
        if qaoa_instance.problem.isFeasible(string):
            cost = qaoa_instance.problem.cost(string)
            counts += hist[string]
            stat.add_sample(cost, hist[string], string)

    return -stat.get_CVaR(), counts / shots


def plot_angles(qaoa_instance, depth, label, style="", fig=None):
    angles = qaoa_instance.optimization_results[depth].get_best_angles()

    if not fig:
        ax = plt.figure().gca()
    else:
        ax = fig.gca()

    plt.plot(
        np.arange(1, depth + 1),
        angles[::2],
        "--" + style,
        label=r"$\gamma$ " + label,
    )
    plt.plot(
        np.arange(1, depth + 1),
        angles[1::2],
        "-" + style,
        label=r"$\beta$ " + label,
    )
    plt.xlim(1 - 0.25, depth + 0.25)
    _ = plt.ylabel("parameter")
    _ = plt.xlabel("depth")
    _ = plt.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def draw_colored_graph(G, edge_colors):
    # Draw the graph with colored edges
    # extend the color_map if necessary
    color_map = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "pink",
        "brown",
        "gray",
        "yellow",
        "cyan",
    ]
    pos = nx.spring_layout(G)  # Positions for all nodes

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="lightgray")

    # Draw edges with colors
    for color_idx, edges in edge_colors.items():
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edges,
            width=2,
            edge_color=color_map[color_idx % len(color_map)],
        )

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")

    # Show the graph
    plt.axis("off")
    plt.show()
