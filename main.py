# Assignment 3: Dynamic Portfolio Allocation
# Authors: Logan Geffroy (11269464), Victor Liu (11272290)
#
# INSTRUCTIONS:
# Run this file ('python main.py') to reproduce the Base Case results (Section 3).
# This will compute the grid, solve the DP, and run the Monte Carlo simulation.

import numpy as np

from grille import grille
from dp_solver import build_transition_matrices, solve_dp
from simulation import simulate_paths
from build_portfolios import build_portfolios


def interpolate_value_at_w0(V_t, w_grid, w0):
    """
    Simple linear interpolation of V_t over the wealth grid at wealth w0.
    Works with row/column vectors or flattened inputs.
    """
    w_flat = np.array(w_grid).reshape(-1)
    V_flat = np.array(V_t).reshape(-1)

    idx_high = np.searchsorted(w_flat, w0, side="left")

    if idx_high == 0:
        return V_flat[0]
    if idx_high >= len(w_flat):
        return V_flat[-1]

    idx_low = idx_high - 1
    w_low = w_flat[idx_low]
    w_high = w_flat[idx_high]
    V_low = V_flat[idx_low]
    V_high = V_flat[idx_high]

    alpha = (w0 - w_low) / (w_high - w_low)
    return V_low + alpha * (V_high - V_low)


def run_base_case(n_paths=50000, rng_seed=123):
    """
    Run the base case and return all the objects needed for plots and analysis.
    This version ALSO prints results (for use inside a notebook).
    """
    # Base case parameters
    w0 = 100.0
    G = 200.0
    T = 10
    nw = 200
    ns = 3

    # Efficient portfolios
    vmu, vsi, weights, index_means, cov_matrix = build_portfolios()

    # 1. Wealth grid
    w_grid, lw_grid = grille(w0, T, nw, ns, vmu, vsi, G)

    # 2. Transition matrices
    P = build_transition_matrices(w_grid, vmu, vsi)

    # 3. Dynamic programming
    V, policy = solve_dp(w_grid, vmu, vsi, G, T, P=P)

    # 4. In-sample probability
    p_in = interpolate_value_at_w0(V[0, :], w_grid, w0)

    # 5. Out-of-sample Monte Carlo simulation
    rng = np.random.default_rng(rng_seed)
    W_paths = simulate_paths(
        w0, vmu, vsi, policy, w_grid, T,
        n_paths=n_paths, rng=rng
    )
    p_oos = np.mean(W_paths[:, -1] >= G)

    # *************   PRINT RESULTS HERE   *************
    print("------ Base Case Results ------")
    print(f"In-sample success probability  : {p_in:.4f}")
    print(f"Out-of-sample success probability: {p_oos:.4f}")
    print("--------------------------------")

    return w0, G, T, w_grid, V, policy, W_paths, p_in, p_oos


def main():
    # Pure script mode: prints occur inside run_base_case()
    run_base_case()


if __name__ == "__main__":
    main()
