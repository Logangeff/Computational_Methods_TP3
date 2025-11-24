import numpy as np

def build_portfolios():
    """
    Construct the 15 efficient portfolios used in the base case (section 4.1)
    of Das et al. (2020).

    The construction follows the article:
    - Table 1: mean vector and covariance matrix of the 3 index funds
      (U.S. Bonds, International Stocks, U.S. Stocks).
    - Figure 1 (bottom): portfolio weights for 15 consecutive portfolios
      on the efficient frontier.

    Returns
    -------
    vmu : ndarray, shape (15,)
        Annual continuously-compounded expected returns of the 15 portfolios.
    vsi : ndarray, shape (15,)
        Annual standard deviations of the 15 portfolios.
    weights : ndarray, shape (15, 3)
        Portfolio weights in the three index funds:
        columns = [U.S. Bonds, International Stocks, U.S. Stocks].
    m : ndarray, shape (3,)
        Mean vector of the three index funds.
    cov : ndarray, shape (3, 3)
        Covariance matrix of the three index funds.
    """

    # Means and covariance of the three index funds (Table 1)
    m = np.array([
        0.0493,   # U.S. Bonds
        0.0770,   # International Stocks
        0.0886    # U.S. Stocks
    ], dtype=float)

    # Covariance matrix (annual), taken from Table 1.
    # Rows/cols in same order as m.
    cov = np.array([
        [ 0.0017 , -0.0017 , -0.0021 ],
        [-0.0017 ,  0.0396 ,  0.03086],
        [-0.0021 ,  0.03086,  0.0392 ]
    ], dtype=float)

    # Ensure symmetry (small rounding differences in the PDF)
    cov[1, 2] = cov[2, 1] = 0.5 * (cov[1, 2] + cov[2, 1])

    # Weights of the 15 portfolios on the efficient frontier (Fig. 1 bottom) 
    # Columns: [U.S. Bonds, International Stocks, U.S. Stocks]
    weights = np.array([
        [0.9098,  0.0225,  0.0677],
        [0.8500,  0.0033,  0.1467],
        [0.7903, -0.0160,  0.2257],
        [0.7305, -0.0352,  0.3047],
        [0.6707, -0.0545,  0.3837],
        [0.6110, -0.0737,  0.4628],
        [0.5512, -0.0930,  0.5418],
        [0.4915, -0.1122,  0.6208],
        [0.4317, -0.1315,  0.6998],
        [0.3719, -0.1507,  0.7788],
        [0.3122, -0.1700,  0.8578],
        [0.2524, -0.1892,  0.9368],
        [0.1927, -0.2085,  1.0158],
        [0.1329, -0.2277,  1.0948],
        [0.0731, -0.2470,  1.1738],
    ], dtype=float)

    # Compute portfolio μ and σ for each of the 15 portfolios 

    # Expected continuously-compounded return
    vmu = weights @ m

    # Variance: w_k^T Σ w_k, then σ_k = sqrt(variance)
    variances = np.einsum("ik,kj,ij->i", weights, cov, weights)
    vsi = np.sqrt(variances)

    return vmu, vsi, weights, m, cov
