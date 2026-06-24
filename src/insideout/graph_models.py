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
    """
    X_scaled = StandardScaler().fit_transform(X)
    model = (
        GraphicalLassoCV()
        if alpha is None
        else GraphicalLasso(alpha=alpha, max_iter=500)
    )
    model.fit(X_scaled)
    return model.precision_, float(getattr(model, "alpha_", alpha))
