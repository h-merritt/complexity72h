"""Plotting utility helpers.

Functions
---------
save_fig
    Save a matplotlib figure as PNG and PDF.
configure_plot_style
    Load matplotlib rcParams from a YAML config file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml

logger = logging.getLogger(__name__)


def _flatten(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Recursively flatten a nested dict into dot-separated keys."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(
                _flatten(v, new_key, sep=sep).items()
            )  # recurse into nested block
        else:
            items.append((new_key, v))  # leaf value: emit as rcParams key
    return dict(items)


def configure_plot_style(path: str | Path) -> None:
    """Load matplotlib rcParams from a YAML config file.

    Parameters
    ----------
    path : str | Path
        Path to the YAML file containing nested rcParam definitions.

    Examples
    --------
    >>> configure_plot_style("configs/plotting/default.yaml")
    """
    with open(path) as f:
        config: dict[str, Any] = yaml.safe_load(f)

    params = _flatten(config)
    for key, value in params.items():
        try:
            plt.rcParams[key] = value
        except Exception:
            logger.warning(
                f"Failed to set rcParam '{key}'; skipping."
            )  # skip unknown or invalid params gracefully


def save_fig(fig: plt.Figure, name: str, plots_dir: str | Path) -> None:
    """Save a matplotlib figure as both PNG and PDF.

    Parameters
    ----------
    fig : plt.Figure
        The figure to save.
    name : str
        Base filename (without extension).
    plots_dir : str | Path
        Root output directory. ``{plots_dir}/png/`` and
        ``{plots_dir}/pdf/`` subdirectories are created as needed.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from insideout.viz.utils import save_fig
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3])
    >>> save_fig(fig, "my_plot", "results/plots")
    """
    plots_dir = Path(plots_dir)
    for fmt in ("png", "pdf"):
        out = plots_dir / fmt
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{name}.{fmt}", bbox_inches="tight")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
