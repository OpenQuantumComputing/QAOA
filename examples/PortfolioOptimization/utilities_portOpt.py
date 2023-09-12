import math
import itertools


def approxRatio(cost, max_feasible, min_feasible):
    # Approximation ratio for feasible solutions of the portfolio optimization problem. Unfeasible solutions
    # have approximation ratio zero.
    # https://link.springer.com/article/10.1007/s11128-022-03766-5
    approx_ratio = (cost - max_feasible) / (min_feasible - max_feasible)
    return approx_ratio


def computeMinMaxCosts(N_assets, minusCostFunction, isFeasible):
    best_sol = None
    min_cost = math.inf
    costs_feasible = []
    costs = []
    for s in ["".join(i) for i in itertools.product("01", repeat=N_assets)]:
        c_penalty = -minusCostFunction(s)  # function returns -cost
        costs.append(c_penalty)
        if isFeasible(s):
            costs_feasible.append(c_penalty)
        if c_penalty < min_cost and isFeasible(s):
            best_sol = s[
                ::-1
            ]  # Qiskit uses big endian encoding, cost function uses litle endian encoding.
            # Therefore the string is reversed before passing it to the cost function.
            min_cost = c_penalty
        else:
            pass
    return min_cost, max(costs_feasible), best_sol


def computeAverageApproxRatio(
    hist, max_feasible, min_feasible, minusCostFunction, isFeasible
):
    # Takes in histogram and computes the average approximation ratio
    tot_shots = 0
    avg_approx_ratio = 0

    for key in hist:
        shots = hist[key]
        tot_shots = tot_shots + shots

        if isFeasible(key):
            cost = -minusCostFunction(
                key[::-1]
            )  # Qiskit uses big endian encoding, cost function uses litle endian encoding.
            # Therefore the string is reversed before passing it to the cost function.
            approx_for_key = approxRatio(cost, max_feasible, min_feasible) * shots
            avg_approx_ratio += approx_for_key

    avg_approx_ratio = avg_approx_ratio / tot_shots
    return avg_approx_ratio
