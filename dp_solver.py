import numpy as np

def build_transition_matrices(w, vmu, vsi):
    """
    Build transition probability matrices for each portfolio.

    Parameters
    ----------
    w : array_like, shape (nw,) or (nw,1)
        Wealth grid.
    vmu : array_like, shape (m,)
        Expected log-returns of the portfolios.
    vsi : array_like, shape (m,)
        Volatilities of the portfolios.

    Returns
    -------
    P : ndarray, shape (m, nw, nw)
        Transition probabilities. P[k, i, j] is the probability of
        moving from wealth node i to node j using portfolio k over one period.
    """
    w = np.asarray(w, dtype=float).reshape(-1)
    vmu = np.asarray(vmu, dtype=float)
    vsi = np.asarray(vsi, dtype=float)

    nw = w.shape[0]
    m = vmu.shape[0]
    log_w = np.log(w)

    P = np.empty((m, nw, nw), dtype=float)
    sqrt2pi = np.sqrt(2.0 * np.pi)

    for k in range(m):
        mu = vmu[k]
        sigma = vsi[k]
        drift = mu - 0.5 * sigma**2

        for i in range(nw):
            mean = log_w[i] + drift
            z = (log_w - mean) / sigma  # standardized log-return
            # standard normal pdf φ(z); proportional to transition weight
            weights = np.exp(-0.5 * z**2) / sqrt2pi

            s = weights.sum()
            if s == 0.0:
                # extremely unlikely numerically, but guard anyway
                P[k, i, :] = 1.0 / nw
            else:
                P[k, i, :] = weights / s

    return P


def solve_dp(w, vmu, vsi, G, T, P=None):
    """
    Backward dynamic programming to maximize probability of reaching G at time T.

    Parameters
    ----------
    w : array_like, shape (nw,) or (nw,1)
        Wealth grid.
    vmu : array_like, shape (m,)
        Expected log-returns of the portfolios.
    vsi : array_like, shape (m,)
        Volatilities of the portfolios.
    G : float
        Goal wealth.
    T : int
        Horizon in periods.
    P : ndarray, shape (m, nw, nw), optional
        Precomputed transition matrices. If None, they are built inside.

    Returns
    -------
    V : ndarray, shape (T+1, nw)
        Value function. V[t, i] = maximal probability of success starting
        at time t in wealth node i.
    policy : ndarray, shape (T, nw)
        Optimal portfolio index. policy[t, i] in {0, ..., m-1}.
    """
    w = np.asarray(w, dtype=float).reshape(-1)
    vmu = np.asarray(vmu, dtype=float)
    vsi = np.asarray(vsi, dtype=float)

    nw = w.shape[0]
    m = vmu.shape[0]

    if P is None:
        P = build_transition_matrices(w, vmu, vsi)

    V = np.zeros((T + 1, nw), dtype=float)
    policy = np.zeros((T, nw), dtype=int)

    # Terminal condition
    V[T, :] = (w >= G).astype(float)

    # Backward induction
    for t in range(T - 1, -1, -1):
        for i in range(nw):
            # expected continuation value under each portfolio
            # P[:, i, :] has shape (m, nw)
            EV = P[:, i, :] @ V[t + 1, :]  # shape (m,)
            best_k = int(np.argmax(EV))
            V[t, i] = EV[best_k]
            policy[t, i] = best_k

    return V, policy
