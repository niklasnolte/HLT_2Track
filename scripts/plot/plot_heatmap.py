import typing as t

import numpy as np
from matplotlib import pyplot as plt
from hlt2trk.utils.config import get_config, Locations, format_location

plt.style.use("seaborn")

cfg = get_config()

X, Y = np.load(format_location(Locations.gridXY, cfg)).values()

def plot_heatmap(
        x: np.ndarray,
        y: np.ndarray,
        params: t.Optional[dict] = None,
):
    x0 = x[:, 0]
    x1 = x[:, 1]
    fig = plt.figure()
    ax: plt.Axes = fig.add_subplot()
    sc = ax.scatter(x0, x1, c=y, cmap=plt.cm.RdBu, s=150, marker="s", **(params or {}))
    ax.set_xlabel(cfg.features[0])
    ax.set_ylabel(cfg.features[1])
    ax.set_title(cfg.model)
    plt.colorbar(sc)
    plt.savefig(format_location(Locations.heatmap, cfg))


plot_heatmap(X, Y)
