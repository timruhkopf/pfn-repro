"""Plots shared by the notebooks: the PPD as a heatmap on the bar distribution's own intervals."""

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh


def _edges(centres: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    mid = (centres[1:] + centres[:-1]) / 2
    return np.concatenate(
        [[max(lo, centres[0] - (mid[0] - centres[0]))], mid, [min(hi, centres[-1] + (centres[-1] - mid[-1]))]]
    )


def density_matrix(density: pd.DataFrame, borders: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows (x, bin, density) of one panel + its (bin, lo, hi) -> (x edges, y edges, density [n_x, n_bins])."""
    xs = np.sort(density["x"].to_numpy(dtype=float))
    xs = np.unique(xs)
    m = density.pivot_table(index="x", columns="bin", values="density").reindex(index=xs)
    b = borders.sort_values("bin")
    y_edges = np.concatenate([b["lo"].to_numpy(), b["hi"].to_numpy()[-1:]])
    return _edges(xs), y_edges, m.to_numpy()


def ppd_heatmap(
    ax: Axes, density: pd.DataFrame, borders: pd.DataFrame, cmap: str = "Blues", vmax_quantile: float = 0.99
) -> QuadMesh:
    """Each cell is one bar interval at one x: its colour is the predictive density there. The colour
    scale is capped at a quantile, so a few very sharp intervals don't wash out the rest."""
    x_edges, y_edges, d = density_matrix(density, borders)
    vmax = float(np.nanquantile(d, vmax_quantile)) or None
    return ax.pcolormesh(x_edges, y_edges, d.T, shading="flat", cmap=cmap, vmin=0, vmax=vmax, rasterized=True)
