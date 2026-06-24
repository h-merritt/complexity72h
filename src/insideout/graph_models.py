"""Graphical model / covariance computation functions.

Functions
---------
compute_covariance
    Sample covariance matrix.
compute_correlation
    Pearson correlation matrix.
compute_precision
    Precision matrix via pseudo-inverse.
fit_glasso
    Graphical Lasso with optional cross-validated regularisation.
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
from sklearn.preprocessing import StandardScaler


def compute_covariance(X: np.ndarray) -> np.ndarray:
    """Compute the sample covariance matrix.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Data matrix with observations as rows and variables as columns.

    Returns
    -------
    np.ndarray of shape (n_features, n_features)
        Sample covariance matrix.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(50, 4)
    >>> cov = compute_covariance(X)
    >>> cov.shape
    (4, 4)
    """
    return np.cov(X.T)


def compute_correlation(X: np.ndarray) -> np.ndarray:
    """Compute the Pearson correlation matrix.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Data matrix with observations as rows and variables as columns.

    Returns
    -------
    np.ndarray of shape (n_features, n_features)
        Pearson correlation matrix with values in [-1, 1].

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(50, 4)
    >>> corr = compute_correlation(X)
    >>> corr.shape
    (4, 4)
    """
    return np.corrcoef(X.T)


def compute_precision(X: np.ndarray) -> np.ndarray:
    """Compute the precision matrix via pseudo-inverse of the covariance.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Data matrix with observations as rows and variables as columns.

    Returns
    -------
    np.ndarray of shape (n_features, n_features)
        Precision matrix (Moore-Penrose pseudo-inverse of the covariance).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(50, 4)
    >>> prec = compute_precision(X)
    >>> prec.shape
    (4, 4)
    """
    return np.linalg.pinv(np.cov(X.T))


def fit_glasso(
    X: np.ndarray,
    alpha: float | None = None,
) -> tuple[np.ndarray, float]:
    """Fit a Graphical Lasso model and return the sparse precision matrix.

    Solves the L1-penalised log-likelihood::

        argmax_{Theta > 0} [log det Theta - tr(S Theta) - alpha * ||Theta||_1]

    If *alpha* is ``None``, the regularisation strength is selected
    automatically via cross-validation (``GraphicalLassoCV``).

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Data matrix. Columns are standardised internally before fitting.
    alpha : float or None, optional
        L1 regularisation strength. ``None`` triggers cross-validated
        selection (default: ``None``).

    Returns
    -------
    precision : np.ndarray of shape (n_features, n_features)
        Estimated sparse precision matrix.
    alpha_used : float
        Regularisation strength actually used (equals *alpha* when provided,
        otherwise the CV-selected value).
    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(50, 4)
    >>> prec, alpha = fit_glasso(X, alpha=0.1)
    >>> prec.shape
    (4, 4)
    """
    X_scaled = StandardScaler().fit_transform(X)
    model = (
        GraphicalLassoCV()
        if alpha is None
        else GraphicalLasso(alpha=alpha, max_iter=500)
    )
    model.fit(X_scaled)
    return model.precision_, float(getattr(model, "alpha_", alpha))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
