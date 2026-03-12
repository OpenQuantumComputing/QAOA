import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
import networkx as nx

import numpy as np
from math import comb

from .statistic import Statistic


def _np2str(npBitString):
    """Cast binary numpy arrays to bitstrings.

    Safe to call both with a standard list and bitstring as long as the
    entries are in fact integers.
    """
    s = ""
    for i in npBitString:
        s += str(int(i))
    return s

# Keep the old private name as an alias for internal backward compatibility.
__np2str = _np2str


def _get_fig_ax(fig=None, **fig_kwargs):
    """Return a (fig, ax) pair, creating a new figure when *fig* is ``None``."""
    if fig is None:
        fig_kwargs.setdefault("figsize", (6, 5))
        fig, ax = plt.subplots(1, 1, **fig_kwargs)
    else:
        ax = fig.gca()
    return fig, ax


def _plot_landscape(A, extent, fig=None, title=None):
    """Plot a 2-D landscape (expectation value or variance) with a colour bar.

    Args:
        A (np.ndarray): 2-D array of values to display.
        extent (list): ``[gamma_min, gamma_max, beta_min, beta_max]``.
        fig (matplotlib.figure.Figure, optional): Existing figure. A new one
            is created when ``None``.
        title (str, optional): Plot title; defaults to *"Expectation value"*.

    Returns:
        tuple: ``(fig, ax)`` – the figure and axes objects.
    """
    fig, ax = _get_fig_ax(fig, figsize=(6, 6), dpi=80, facecolor="w", edgecolor="k")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(title if title else "Expectation value")
    im = ax.imshow(A, interpolation="bicubic", origin="lower", extent=extent)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    return fig, ax


def plot_E(qaoa_instance, fig=None, title=None):
    """Plot the sampled expectation-value landscape at depth *p = 1*.

    Args:
        qaoa_instance (QAOA): A QAOA instance whose cost landscape has been sampled.
        fig (matplotlib.figure.Figure, optional): Existing figure to draw on.
        title (str, optional): Plot title.

    Returns:
        tuple: ``(fig, ax)``.
    """
    angles = qaoa_instance.landscape_p1_angles
    extent = [
        angles["gamma"][0],
        angles["gamma"][1],
        angles["beta"][0],
        angles["beta"][1],
    ]
    return _plot_landscape(qaoa_instance.exp_landscape(), extent, fig=fig, title=title)


def plot_Var(qaoa_instance, fig=None, title=None):
    """Plot the sampled variance landscape at depth *p = 1*.

    Args:
        qaoa_instance (QAOA): A QAOA instance whose cost landscape has been sampled.
        fig (matplotlib.figure.Figure, optional): Existing figure to draw on.
        title (str, optional): Plot title.

    Returns:
        tuple: ``(fig, ax)``.
    """
    angles = qaoa_instance.landscape_p1_angles
    extent = [
        angles["gamma"][0],
        angles["gamma"][1],
        angles["beta"][0],
        angles["beta"][1],
    ]
    return _plot_landscape(qaoa_instance.var_landscape(), extent, fig=fig, title=title)


def plot_ApproximationRatio(
    qaoa_instance, maxdepth, mincost, maxcost, label, style="", fig=None, shots=None
):
    """Plot the approximation ratio as a function of circuit depth.

    Args:
        qaoa_instance (QAOA): A QAOA instance that has been optimized.
        maxdepth (int): Maximum depth to plot.
        mincost (float): Known minimum cost (for normalization).
        maxcost (float): Known maximum cost (for normalization).
        label (str): Legend label.
        style (str): Matplotlib line-style string.
        fig (matplotlib.figure.Figure, optional): Existing figure to draw on.
        shots (int, optional): If given, re-sample to compute the ratio
            with feasibility post-processing.

    Returns:
        tuple: ``(fig, ax)``.
    """
    if not shots:
        exp = np.array(qaoa_instance.get_Exp())
    else:
        exp = []
        for p in range(1, qaoa_instance.current_depth + 1):
            ar, _sp = _apprrat_successprob(qaoa_instance, p, shots=shots)
            exp.append(ar)
        exp = np.array(exp)

    fig, ax = _get_fig_ax(fig)
    ax.hlines(1, 1, maxdepth, linestyles="solid", colors="black")
    ax.plot(
        np.arange(1, maxdepth + 1),
        (maxcost - exp) / (maxcost - mincost),
        style,
        label=label,
    )
    ax.set_ylim(0, 1.01)
    ax.set_xlim(1 - 0.25, maxdepth + 0.25)
    ax.set_ylabel("appr. ratio")
    ax.set_xlabel("depth")
    ax.legend(loc="lower right", framealpha=1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return fig, ax


def plot_successprob(qaoa_instance, maxdepth, label, style="", fig=None, shots=10**4):
    """Plot the success probability as a function of circuit depth.

    Args:
        qaoa_instance (QAOA): A QAOA instance that has been optimized.
        maxdepth (int): Maximum depth to plot.
        label (str): Legend label.
        style (str): Matplotlib line-style string.
        fig (matplotlib.figure.Figure, optional): Existing figure to draw on.
        shots (int): Number of shots for sampling.

    Returns:
        tuple: ``(fig, ax)``.
    """
    successp = []
    for p in range(1, qaoa_instance.current_depth + 1):
        _ar, sp = _apprrat_successprob(qaoa_instance, p, shots=shots)
        successp.append(sp)
    successp = np.array(successp)

    fig, ax = _get_fig_ax(fig)
    ax.hlines(1, 1, maxdepth, linestyles="solid", colors="black")
    ax.plot(
        np.arange(1, maxdepth + 1),
        successp,
        style,
        label=label,
    )
    ax.set_ylim(0, 1.01)
    ax.set_xlim(1 - 0.25, maxdepth + 0.25)
    ax.set_ylabel("success prob")
    ax.set_xlabel("depth")
    ax.legend(loc="lower right", framealpha=1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return fig, ax


def _apprrat_successprob(qaoa_instance, depth, shots=10**4):
    """Compute approximation ratio and success probability with feasibility post-processing."""
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

# Keep the old private name as an alias for internal backward compatibility.
__apprrat_successprob = _apprrat_successprob


def plot_angles(qaoa_instance, depth, label, style="", fig=None):
    """Plot optimal gamma and beta angles at a given depth.

    Args:
        qaoa_instance (QAOA): A QAOA instance that has been optimized.
        depth (int): Circuit depth whose angles to display.
        label (str): Legend label suffix.
        style (str): Matplotlib line-style string.
        fig (matplotlib.figure.Figure, optional): Existing figure to draw on.

    Returns:
        tuple: ``(fig, ax)``.
    """
    angles = qaoa_instance.optimization_results[depth].get_best_angles()

    fig, ax = _get_fig_ax(fig)

    ax.plot(
        np.arange(1, depth + 1),
        angles[::2],
        "--" + style,
        label=r"$\gamma$ " + label,
    )
    ax.plot(
        np.arange(1, depth + 1),
        angles[1::2],
        "-" + style,
        label=r"$\beta$ " + label,
    )
    ax.set_xlim(1 - 0.25, depth + 0.25)
    ax.set_ylabel("parameter")
    ax.set_xlabel("depth")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return fig, ax


def draw_colored_graph(G, edge_colors):
    """Draw a graph with edges coloured according to *edge_colors*.

    Args:
        G (networkx.Graph): The graph to draw.
        edge_colors (dict): Mapping ``{colour_index: [(u, v), ...], ...}``.

    Returns:
        tuple: ``(fig, ax)``.
    """
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
    fig, ax = plt.subplots()
    pos = nx.circular_layout(G)

    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="lightgray", ax=ax)

    for color_idx, edges in edge_colors.items():
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edges,
            width=2,
            edge_color=color_map[color_idx % len(color_map)],
            ax=ax,
        )

    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif", ax=ax)

    ax.set_axis_off()
    return fig, ax


def plot_AllOptimalParameters(qaoa, figsize=(14, 6), title=None):
    """Plot optimal parameter trajectories across depths (2×2 panel).

    Top row shows gamma parameters, bottom row shows beta parameters.
    Left column: parameter values at each depth.
    Right column: each parameter index tracked across depths.

    Concept taken from Figure 1 a) and b) in https://arxiv.org/pdf/2209.11348.pdf

    Args:
        qaoa (QAOA): A QAOA instance that has been optimized.
        figsize (tuple): Figure size.
        title (str, optional): Super-title for the figure.

    Returns:
        tuple: ``(fig, axs)`` where *axs* is a flat list of four axes.
    """
    maxdepth = qaoa.current_depth
    fig, axs = plt.subplots(2, 2, figsize=figsize, constrained_layout=False, sharex=True)
    axs = [None] * 4
    axs[0] = plt.subplot(2, 2, 1)
    labelsLeft = []
    for p in range(1, maxdepth + 1):
        labelsLeft.append("p = " + str(p))
        axs[0].plot(np.arange(1, p + 1), qaoa.get_gamma(p), label=labelsLeft[-1], linestyle="-", marker="o")
    axs[0].set_ylabel(r"$\gamma_j$")
    axs[0].grid()

    axs[1] = plt.subplot(2, 2, 3)
    for p in range(1, maxdepth + 1):
        axs[1].plot(np.arange(1, p + 1), qaoa.get_beta(p), linestyle="-", marker="o")
    axs[1].grid()
    axs[1].set_ylabel(r"$\beta_j$")
    axs[1].set_xlabel("Parameter index, $j$")

    plt.xticks(np.arange(1, maxdepth + 1))
    fig.subplots_adjust(bottom=0.15)

    fig.legend(labels=labelsLeft, loc="upper left", bbox_to_anchor=(0.08, 1.08), ncol=int(np.ceil(maxdepth / 2)))

    axs[2] = plt.subplot(2, 2, 2)
    labelsRight = []
    allGammas = [qaoa.get_gamma(p) for p in range(1, maxdepth + 1)]
    allBetas = [qaoa.get_beta(p) for p in range(1, maxdepth + 1)]

    for p in range(1, maxdepth + 1):
        indexDevelopment = [allGammas[i][p - 1] for i in range(p - 1, maxdepth)]
        labelsRight.append("j = " + str(p))
        axs[2].plot(np.arange(p, maxdepth + 1), indexDevelopment, label=labelsRight[-1], linestyle="-", marker="o")
    axs[2].set_ylabel(r"$\gamma_j$")
    axs[2].grid()

    axs[3] = plt.subplot(2, 2, 4)
    for p in range(1, maxdepth + 1):
        indexDevelopment = [allBetas[i][p - 1] for i in range(p - 1, maxdepth)]
        axs[3].plot(np.arange(p, maxdepth + 1), indexDevelopment, label=labelsRight[-1], linestyle="-", marker="o")
    axs[3].grid()
    axs[3].set_ylabel(r"$\beta_j$")
    axs[3].set_xlabel("Circuit depth, $p$")

    plt.xticks(np.arange(1, maxdepth + 1))
    fig.subplots_adjust(bottom=0.15)

    fig.legend(labels=labelsRight, loc="upper right", bbox_to_anchor=(0.95, 1.08), ncol=int(np.ceil(maxdepth / 2)))
    fig.tight_layout()

    if title:
        fig.suptitle(title, y=1.12, fontsize=15)

    return fig, axs


def plot_optimalHitRatios(qaoa, optimal_solution, shots=1024, fig=None, label=None, style="", title=None, **kwargs):
    """Plot the hit ratio for the optimal solution as a function of depth.

    Args:
        qaoa (QAOA): A QAOA instance that has been optimized.
        optimal_solution (array-like): Binary array representing the optimal
            solution (will be converted to a bitstring).
        shots (int): Number of shots for sampling.
        fig (matplotlib.figure.Figure, optional): Existing figure to draw on.
        label (str, optional): Legend label.
        style (str): Matplotlib line-style string.
        title (str, optional): Plot title.

    Returns:
        tuple: ``(fig, ax)``.
    """
    optimal_sol = _np2str(optimal_solution)
    hit_rates = np.zeros(qaoa.current_depth)
    for d in range(qaoa.current_depth):
        hist = qaoa.hist(qaoa.optimization_results[d + 1].get_best_angles(), shots)
        if optimal_sol in hist:
            num_hits = hist[optimal_sol]
            hit_rates[d] = num_hits / shots

    fig, ax = _get_fig_ax(fig)
    ax.plot(np.arange(1, qaoa.current_depth + 1), hit_rates, style, label=label, **kwargs)
    ax.legend()
    ax.set_xlabel("depth")
    ax.set_ylabel("Hit ratio for optimal solution")
    if title:
        ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return fig, ax


def plot_feasibleHitRatios(qaoa, shots=1024, fig=None, label=None, style="", title=None, **kwargs):
    """Plot the feasible-solution hit ratio as a function of depth.

    Args:
        qaoa (QAOA): A QAOA instance that has been optimized.
        shots (int): Number of shots for sampling.
        fig (matplotlib.figure.Figure, optional): Existing figure to draw on.
        label (str, optional): Legend label.
        style (str): Matplotlib line-style string.
        title (str, optional): Plot title.

    Returns:
        tuple: ``(fig, ax)``.
    """
    hit_rates = np.zeros(qaoa.current_depth)
    for d in range(qaoa.current_depth):
        hist = qaoa.hist(qaoa.optimization_results[d + 1].get_best_angles(), shots)
        num_hits = 0
        for bitstring, hits in hist.items():
            if qaoa.problem.isFeasible(bitstring):
                num_hits += hits
        hit_rates[d] = num_hits / shots

    fig, ax = _get_fig_ax(fig)
    ax.plot(np.arange(1, qaoa.current_depth + 1), hit_rates, style, label=label, **kwargs)
    ax.legend()
    ax.set_xlabel("depth")
    ax.set_ylabel("Hit ratio for feasible solutions")
    if title:
        ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return fig, ax


def printBestHistogramEntries(qaoa, classical_solution=None, num_solutions=10, shots=1024):
    """Print the most frequent measurement outcomes at each depth.

    Args:
        qaoa (QAOA): A QAOA instance that has been optimized.
        classical_solution (array-like, optional): Known optimal solution
            (binary array); highlighted in the output with ``-->``.
        num_solutions (int): Number of top solutions to display per depth.
        shots (int): Number of shots for sampling.
    """
    best_classical_sol = None
    if classical_solution is not None:
        best_classical_sol = _np2str(classical_solution)
        print("Classical best result: ", (best_classical_sol, qaoa.problem.cost(best_classical_sol)))
        print(" --> points to the classical solution ")
    print("   * marks feasible solutions ")
    for p in range(1, qaoa.current_depth + 1):
        hist = qaoa.hist(qaoa.optimization_results[p].get_best_angles(), shots=shots)

        sorted_hist = dict(sorted(hist.items(), key=lambda item: item[1], reverse=True))

        i = 1
        best_classical_sol_freq = None
        best_classical_sol_i = None
        print("Results for depth " + str(p) + " using best angles:")
        for s, freq in sorted_hist.items():
            cost = qaoa.problem.cost(s)
            if i == 1:
                best_sol = s
                best_cost = cost
                best_freq = freq
                best_i = i
            elif cost > best_cost:
                best_sol = s
                best_cost = cost
                best_freq = freq
                best_i = i
            if s == best_classical_sol:
                best_classical_sol_freq = freq
                best_classical_sol_i = i
            if i <= num_solutions:
                toprint = "    "
                if s == best_classical_sol:
                    toprint = "--> "
                elif qaoa.problem.isFeasible(s):
                    toprint = "  * "
                print(str(i) + "\t" + toprint + str(s) + ", " + str(cost) + ",   " + str(freq))
            i = i + 1
        if best_classical_sol_freq is not None:
            print("Found best classical solution with " + str(best_classical_sol_freq) + " shots (rank: " + str(best_classical_sol_i) + ")")
        else:
            print("Did not find best classical solution")
            print(str(best_i) + "\t" + toprint + str(best_sol) + ", " + str(best_cost) + ",   " + str(best_freq) + "<-- Best obtained solution")


def plotHitProbabilities(qaoa, opt_sol, depth=None, hist_shots=2**13, **kwargs):
    """Plot hit probability for the optimal solution (delegates to :func:`plotHitProbabilities_fromHist`).

    Args:
        qaoa (QAOA): A QAOA instance that has been optimized.
        opt_sol (str): Optimal solution bitstring.
        depth (int, optional): Depth to use; defaults to ``qaoa.current_depth``.
        hist_shots (int): Number of shots for sampling.
        **kwargs: Forwarded to :func:`plotHitProbabilities_fromHist`.

    Returns:
        tuple: ``(fig, ax)``.
    """
    if depth is None:
        depth = qaoa.current_depth
    hist = qaoa.hist(qaoa.get_angles(depth), hist_shots)

    return plotHitProbabilities_fromHist(hist, opt_sol, **kwargs)


def plotHitProbabilities_fromHist(hist, opt_sol,
                         hamming_weight=None, plot_random=True, fig=None,
                         label="QAOA", style="o-", title=None, title_add_on="",
                         max_shots_base2=20):
    """Plot the probability of finding the optimal solution vs. number of shots.

    Args:
        hist (dict): Histogram of measurement outcomes.
        opt_sol (str): Optimal solution bitstring.
        hamming_weight (int, optional): If given, also plot the random
            baseline restricted to the subspace with this Hamming weight.
        plot_random (bool): Whether to plot random-guess baselines.
        fig (matplotlib.figure.Figure, optional): Existing figure to draw on.
        label (str): Legend label for the QAOA curve.
        style (str): Matplotlib line-style string.
        title (str, optional): Plot title.
        title_add_on (str): Additional text appended to the title.
        max_shots_base2 (int): Log₂ of the maximum number of shots on the x-axis.

    Returns:
        tuple: ``(fig, ax)``.
    """

    def prob_hit_ones(p, n):
        return 1 - (1 - p)**n

    fig, ax = _get_fig_ax(fig)
    ns = 2**np.arange(max_shots_base2)

    if plot_random:
        state_space_size = 2**len(opt_sol)
        p_random = 1 / state_space_size
        hit_prob_random = [prob_hit_ones(p_random, n) for n in ns]
        ax.plot(ns, hit_prob_random, ".:k", label="random all")

        if hamming_weight is not None:
            subspace_size = comb(len(opt_sol), hamming_weight)
            p_subspace = 1 / subspace_size
            hit_prob_subspace = [prob_hit_ones(p_subspace, n) for n in ns]
            ax.plot(ns, hit_prob_subspace, ".-.k", label="random subspace")

    hist_shots = sum(hist.values())
    p_qaoa = 0.0
    if opt_sol in hist.keys():
        p_qaoa = hist[opt_sol] / hist_shots
    hit_prob_qaoa = [prob_hit_ones(p_qaoa, n) for n in ns]
    ax.plot(ns, hit_prob_qaoa, style, label=label)

    ax.set_xscale("log")
    ax.set_ylabel("probability")
    ax.set_xlabel("num shots")
    if title is None:
        ax.set_title("Probability for finding optimal solution " + title_add_on)
    else:
        ax.set_title(title + title_add_on)
    ax.legend()
    ax.grid(True, which="both", ls="--")
    return fig, ax


