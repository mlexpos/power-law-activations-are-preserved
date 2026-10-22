"""Power-law fits of spectra, used by every experiment that reports an exponent.

A fit is ordinary least squares of log10 lambda_j on log10 j over an explicit index window
[j_min, j_max] (1-indexed, inclusive).  Every reported slope states its window.
"""

import numpy as np

DEFAULT_WINDOW = (1, 100)
ROBUSTNESS_WINDOWS = [(1, 100), (10, 100), (1, 30), (10, 300)]


def powerlaw_fit(lam, window=DEFAULT_WINDOW):
    """(slope, intercept, R^2) of the log-log OLS fit of lam_j, j in window; lam in decreasing order."""
    j_min, j_max = window
    j_max = min(j_max, len(lam))
    j = np.arange(j_min, j_max + 1)
    y = np.asarray(lam[j_min - 1:j_max], dtype=float)
    if (y <= 0).any():
        raise ValueError(f"non-positive eigenvalue in window {window}")
    x, y = np.log10(j), np.log10(y)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    r2 = 1 - (resid**2).sum() / ((y - y.mean()) ** 2).sum()
    return float(slope), float(intercept), float(r2)


def powerlaw_fit_or_nan(lam, window=DEFAULT_WINDOW):
    """As powerlaw_fit, but (nan, nan, nan) when the window holds a non-positive eigenvalue (a
    numerically rank-deficient covariance)."""
    try:
        return powerlaw_fit(lam, window)
    except ValueError:
        return (np.nan, np.nan, np.nan)
