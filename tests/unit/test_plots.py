import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pfn_repro.evaluation.plots import density_matrix, ppd_heatmap


def test_density_matrix_and_heatmap() -> None:
    xs, bins = np.linspace(0.05, 0.95, 10), np.arange(4)
    xi, bi = np.meshgrid(xs, bins, indexing="ij")
    dens = pd.DataFrame({"x": xi.ravel(), "bin": bi.ravel(), "density": np.ones(40)})
    borders = pd.DataFrame({"bin": bins, "lo": [0, 0.1, 0.5, 0.6], "hi": [0.1, 0.5, 0.6, 1.0]})
    x_edges, y_edges, d = density_matrix(dens, borders)
    assert len(x_edges) == 11 and list(y_edges) == [0, 0.1, 0.5, 0.6, 1.0] and d.shape == (10, 4)
    assert x_edges[0] >= 0 and x_edges[-1] <= 1 and bool(np.all(np.diff(x_edges) > 0))
    fig, ax = plt.subplots()
    ppd_heatmap(ax, dens, borders)
    plt.close(fig)
