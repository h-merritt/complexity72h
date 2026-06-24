from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_fig(fig: plt.Figure, name: str, plots_dir: str | Path) -> None:
    plots_dir = Path(plots_dir)
    for fmt in ("png", "pdf"):
        out = plots_dir / fmt
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{name}.{fmt}", bbox_inches="tight")
