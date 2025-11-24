import numpy as np

def grille(w0, T, nw, ns, vmu, vsi, G):
    """
    Python translation of the MATLAB grille.m function.

    Parameters
    ----------
    w0 : float
        Initial wealth.
    T : int
        Horizon in periods.
    nw : int
        Number of wealth grid points.
    ns : float
        Number of standard deviations for the grid range.
    vmu : array_like
        Portfolio expected log-returns (shape (m,)).
    vsi : array_like
        Portfolio volatilities (shape (m,)).
    G : float
        Goal wealth level.

    Returns
    -------
    w : ndarray, shape (nw, 1)
        Wealth grid in dollars.
    lw : ndarray, shape (nw, 1)
        Natural logarithm of the wealth grid.
    """
    vmu = np.asarray(vmu, dtype=float)
    vsi = np.asarray(vsi, dtype=float)

    # min and max of means and std devs
    mumin = np.min(vmu)
    mumax = np.max(vmu)
    simax = np.max(vsi)

    # min and max wealth
    wmin = w0 * np.exp((mumin - 0.5 * simax * simax) * T - ns * simax * np.sqrt(T))
    wmax = w0 * np.exp((mumax - 0.5 * simax * simax) * T + ns * simax * np.sqrt(T))

    # grid of ln(wealth) – column vector like MATLAB
    lw = np.linspace(np.log(wmin), np.log(wmax), nw).reshape(-1, 1)

    # adjustment to have ln(G) between two points
    lG = np.log(G)
    # MATLAB: I = find(lw > lG, 1)
    I = np.where(lw > lG)[0][0]

    lw_lower = lw[I - 1, 0]
    lw_upper = lw[I, 0]
    lw_mid = lw_lower + (lw_upper - lw_lower) / 2.0

    # difference between midpoint and ln(G)
    dif = lG - lw_mid

    # adjust log-wealth grid
    lw = lw + dif

    # wealth grid in dollars
    w = np.exp(lw)

    return w, lw
