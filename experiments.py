import numpy as np
from time import perf_counter

from grille import grille
from dp_solver import build_transition_matrices, solve_dp
from simulation import simulate_paths


def run_reference_case(vmu, vsi,
                       w0=100.0, G=200.0, T=10,
                       nw=200, ns=3,
                       n_paths_oos=50000, rng=None):
    """
    Run the base case similar to section 4.1 (no cash flows).

    Returns a dict with grids, value function, policy, and probabilities.
    """
    start = perf_counter()

    w_grid, lw_grid = grille(w0, T, nw, ns, vmu, vsi, G)
    P = build_transition_matrices(w_grid, vmu, vsi)
    V, policy = solve_dp(w_grid, vmu, vsi, G, T, P=P)

    dp_time = perf_counter() - start

    # interpolate in-sample success probability at initial wealth w0
    w_flat = w_grid.reshape(-1)
    idx_high = np.searchsorted(w_flat, w0, side="left")
    if idx_high == 0:
        V0 = V[0, 0]
    elif idx_high >= len(w_flat):
        V0 = V[0, -1]
    else:
        idx_low = idx_high - 1
        w_low = w_flat[idx_low]
        w_high = w_flat[idx_high]
        V_low = V[0, idx_low]
        V_high = V[0, idx_high]
        alpha = (w0 - w_low) / (w_high - w_low)
        V0 = V_low + alpha * (V_high - V_low)

    # out-of-sample Monte Carlo
    sim_start = perf_counter()
    W_paths = simulate_paths(w0, vmu, vsi, policy, w_grid, T,
                             n_paths=n_paths_oos, rng=rng)
    sim_time = perf_counter() - sim_start

    p_oos = np.mean(W_paths[:, -1] >= G)

    return {
        "w_grid": w_grid,
        "lw_grid": lw_grid,
        "V": V,
        "policy": policy,
        "P": P,
        "p_success_in_sample": V0,
        "p_success_oos": p_oos,
        "dp_time_sec": dp_time,
        "sim_time_sec": sim_time,
    }


def experiment_vary_nw(vmu, vsi, nw_list,
                       w0=100.0, G=200.0, T=10,
                       ns=3, n_paths_oos=20000, rng=None):
    """
    Run the model for different wealth grid sizes nw.
    Returns a list of result dicts, one per nw.
    """
    results = []
    for nw in nw_list:
        res = run_reference_case(vmu, vsi, w0=w0, G=G, T=T,
                                 nw=nw, ns=ns,
                                 n_paths_oos=n_paths_oos, rng=rng)
        res["nw"] = nw
        results.append(res)
    return results


def experiment_vary_ns(vmu, vsi, ns_list,
                       w0=100.0, G=200.0, T=10,
                       nw=200, n_paths_oos=20000, rng=None):
    """
    Run the model for different values of ns (grid range in standard deviations).
    Returns a list of result dicts, one per ns.
    """
    results = []
    for ns in ns_list:
        res = run_reference_case(vmu, vsi, w0=w0, G=G, T=T,
                                 nw=nw, ns=ns,
                                 n_paths_oos=n_paths_oos, rng=rng)
        res["ns"] = ns
        results.append(res)
    return results
