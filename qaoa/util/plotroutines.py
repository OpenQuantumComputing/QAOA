import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator

import numpy as np

from qaoa import QAOA

from qaoa.util import Statistic

def np2str(npBitString):
    """
    Cast binary numpy arrays to bitstrings.
    Safe to call both with a standard list and bitstring as long as the entries are in fact integers
    """
    s = ""
    for i in npBitString:
        s += str(int(i))
    return s

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
    return fig, ax


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

def plot_AllOptimalParameters(qaoa, figsize=(14, 6), title=None):
    """
    2x2 plot showing the development of the best parameter values found for each parameter and each depth.
    Top row:         Gamma parameters
    Lower row:       Beta parameters
    Left-hand side:  Relationship between parameters values plotted for each depth
    Right-hand side: Development of each parameter's optimal value w.r.t. depth

    Consept taken from Figure 1 a) and b) in https://arxiv.org/pdf/2209.11348.pdf 
    """
    maxdepth = qaoa.current_depth
    fig, axs = plt.subplots(2, 2, figsize=figsize, constrained_layout=False, sharex=True)
    axs = [None]*4
    axs[0] = plt.subplot(2, 2, 1)
    labelsLeft = []
    for p in range(1, maxdepth+1):
        labelsLeft.append("p = "+str(p))
        axs[0].plot(np.arange(1, p+1), qaoa.get_gamma(p), label=labelsLeft[-1], linestyle='-', marker='o')
    axs[0].set_ylabel(r'$\gamma_j$')
    axs[0].grid()

    axs[1] = plt.subplot(2, 2, 3)
    for p in range(1, maxdepth+1):
        axs[1].plot(np.arange(1, p+1), qaoa.get_beta(p), linestyle='-', marker='o')
    axs[1].grid()
    axs[1].set_ylabel(r'$\beta_j$')
    axs[1].set_xlabel('Parameter index, $j$')



    plt.xticks(np.arange(1,maxdepth+1))
    fig.subplots_adjust(bottom=0.15)   ##  Need to play with this number.

    fig.legend(labels=labelsLeft, loc='upper left', bbox_to_anchor=(0.08, 1.08), ncol=int(np.ceil(maxdepth/2)))



    axs[2] = plt.subplot(2, 2, 2)
    labelsRight = []
    allGammas = [qaoa.get_gamma(p) for p in range(1, maxdepth+1)]
    allBetas = [qaoa.get_beta(p) for p in range(1, maxdepth+1)]
    
    for p in range(1, maxdepth+1):
        indexDevelopment = [allGammas[i][p-1] for i in range(p-1, maxdepth)]
        labelsRight.append("j = "+str(p))
        axs[2].plot(np.arange(p, maxdepth+1), indexDevelopment, label=labelsRight[-1], linestyle='-', marker='o')
    axs[2].set_ylabel(r'$\gamma_j$')
    axs[2].grid()

    axs[3] = plt.subplot(2, 2, 4)
    for p in range(1, maxdepth+1):
        indexDevelopment = [allBetas[i][p-1] for i in range(p-1, maxdepth)]
        axs[3].plot(np.arange(p, maxdepth+1), indexDevelopment, label=labelsRight[-1], linestyle='-', marker='o')
    axs[3].grid()
    axs[3].set_ylabel(r'$\beta_j$')
    axs[3].set_xlabel('Circuit depth, $p$')


    plt.xticks(np.arange(1,maxdepth+1))
    fig.subplots_adjust(bottom=0.15)   ##  Need to play with this number.

    fig.legend(labels=labelsRight, loc='upper right', bbox_to_anchor=(0.95, 1.08), ncol=int(np.ceil(maxdepth/2)))
    fig.tight_layout() 

    if title:
        fig.suptitle(title, y = 1.12, fontsize=15)
    

def plot_optimalHitRatios(qaoa, optimal_solution, shots=1024, fig=None, label=None, style="", title=None, **kwargs):

    # Compute the rate at which the best angles gives us the optimal solution
    optimal_sol = np2str(optimal_solution)
    hit_rates = np.zeros(qaoa.current_depth)
    for d in range(qaoa.current_depth):
        hist = qaoa.hist(qaoa.optimization_results[d+1].get_best_angles(), shots)
        if optimal_sol in hist:
            num_hits = hist[optimal_sol]
            hit_rates[d] = num_hits/shots
    
    plt.plot(np.arange(1, qaoa.current_depth+1), hit_rates, style, label=label, **kwargs)
    plt.legend()
    plt.xlabel("depth")
    plt.ylabel("Hit ratio for optimal solution")
    if title:
        plt.title(title)


def plot_feasibleHitRatios(qaoa, shots=1024, fig=None, label=None, style="", title=None, **kwargs):

    # Compute the rate at which the best angles gives us feasible solutions
    hit_rates = np.zeros(qaoa.current_depth)
    for d in range(qaoa.current_depth):
        hist = qaoa.hist(qaoa.optimization_results[d+1].get_best_angles(), shots)
        num_hits = 0
        for bitstring, hits in hist.items():
            if qaoa.problem.isFeasible(bitstring):
                num_hits += hits        
        hit_rates[d] = num_hits/shots
    
    plt.plot(np.arange(1, qaoa.current_depth+1), hit_rates, style, label=label, **kwargs)
    plt.legend()
    plt.xlabel("depth")
    plt.ylabel("Hit ratio for feasible solutions")
    if title:
        plt.title(title)



def printBestHistogramEntries( qaoa, classical_solution = None, num_solutions=10, shots=1024):

    best_classical_sol = None
    if classical_solution is not None:
        best_classical_sol = np2str(classical_solution)
        print("Classical best result: ", (best_classical_sol, qaoa.problem.cost(best_classical_sol)))
        print(" --> points to the classical solution ")
    print("   * marks feasible solutions ")
    for p in range(1, qaoa.current_depth+1):
        hist = qaoa.hist(qaoa.optimization_results[p].get_best_angles(), shots=shots)

        sorted_hist = dict(sorted(hist.items(), key=lambda item: item[1], reverse=True))

        i = 1
        best_classical_sol_freq = None
        best_classical_sol_i = None
        print("Results for depth "+str(p)+" using best angles:")
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
                print(str(i) +"\t" + toprint + str(s) +", " + str(cost) + ",   " +str(freq))
            i = i + 1
        if best_classical_sol_freq is not None:
            print("Found best classical solution with " + str(best_classical_sol_freq) + " shots (rank: "+str(best_classical_sol_i)+")")
        else:
            print("Did not find best classical solution")
            print(str(best_i) +"\t" + toprint + str(best_sol) +", " + str(best_cost) + ",   " +str(best_freq) +"<-- Best obtained solution")
