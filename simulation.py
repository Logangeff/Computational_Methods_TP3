import numpy as np


def _nearest_grid_indices(w_grid, w_values):
    """
    Map continuous wealth values to nearest indices on the wealth grid,
    using distance in log-wealth.
    """
    w_grid = np.asarray(w_grid, dtype=float).reshape(-1)
    w_values = np.asarray(w_values, dtype=float)

    log_grid = np.log(w_grid)
    log_vals = np.log(w_values)

    # searchsorted function gives insertion index on the sorted log-grid
    idx = np.searchsorted(log_grid, log_vals, side="left")

    idx0 = np.clip(idx - 1, 0, len(log_grid) - 1)
    idx1 = np.clip(idx, 0, len(log_grid) - 1)

    # choose nearer of idx0 and idx1 in log-space
    choose_right = np.abs(log_vals - log_grid[idx1]) < np.abs(log_vals - log_grid[idx0])
    indices = np.where(choose_right, idx1, idx0)

    return indices


def simulate_paths(w0, vmu, vsi, policy, w_grid, T, n_paths, rng=None):
    """
    Simulate Monte Carlo wealth paths under a fixed optimal policy.

    Parameters
    ----------
    w0 : float
        Initial wealth.
    vmu : array_like, shape (m,)
        Portfolio expected log-returns.
    vsi : array_like, shape (m,)
        Portfolio volatilities.
    policy : ndarray, shape (T, nw)
        Optimal portfolio indices from DP.
    w_grid : array_like, shape (nw,) or (nw,1)
        Wealth grid used in the DP.
    T : int
        Horizon in periods.
    n_paths : int
        Number of simulated scenarios.
    rng : np.random.Generator, optional
        Random number generator. If None, a default one is created.

    Returns
    -------
    W_paths : ndarray, shape (n_paths, T+1)
        Simulated wealth paths.
    """
    vmu = np.asarray(vmu, dtype=float)
    vsi = np.asarray(vsi, dtype=float)
    w_grid = np.asarray(w_grid, dtype=float).reshape(-1)
    policy = np.asarray(policy, dtype=int)

    if rng is None:
        rng = np.random.default_rng()

    W_paths = np.empty((n_paths, T + 1), dtype=float)
    W_paths[:, 0] = w0

    for t in range(T):
        # determine grid indices and corresponding portfolio for all paths
        idx = _nearest_grid_indices(w_grid, W_paths[:, t])
        k_t = policy[t, idx]     
        mu_t = vmu[k_t]
        sigma_t = vsi[k_t]

        z = rng.normal(size=n_paths)
        # geometric Brownian update
        W_paths[:, t + 1] = W_paths[:, t] * np.exp(
            (mu_t - 0.5 * sigma_t**2) + sigma_t * z
        )

    return W_paths
