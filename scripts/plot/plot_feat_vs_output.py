import typing as t

import numpy as np
from matplotlib import pyplot as plt
from hlt2trk.utils.config import get_config, Locations, format_location

plt.style.use("seaborn")

cfg = get_config()

X, Y = np.load(format_location(Locations.gridXY, cfg)).values()

def plot_feat_vs_output(
        x: np.ndarray, y: np.ndarray, params: t.Optional[dict] = None):
    params = params or {}
    fig, axes = plt.subplots(
        int(np.ceil(len(cfg.features) / 2)),
        2, dpi=120, figsize=(16, 9), sharey=True)
    for i, (feature, ax) in enumerate(zip(cfg.features, axes.flatten())):
        xi = x[:, i]
        srt = np.argsort(xi)
        xi = xi[srt]
        xiuniq, idxs = np.unique(xi, return_index=True)
        y_by_xi_mean = [yi.mean() for yi in np.split(y[srt], idxs[1:])]
        y_by_xi_min = [yi.min() for yi in np.split(y[srt], idxs[1:])]
        y_by_xi_max = [yi.max() for yi in np.split(y[srt], idxs[1:])]
        ax.scatter(xiuniq, y_by_xi_mean, **params, label="mean")
        ax.scatter(xiuniq, y_by_xi_min, **params, label="min")
        ax.scatter(xiuniq, y_by_xi_max, **params, label="max")
        ax.set_xlabel(feature)
        ax.set_ylabel("output")
        ax.legend()
    fig.suptitle(cfg.model)
    plt.tight_layout()
    plt.savefig(format_location(Locations.feat_vs_output, cfg))
    # plt.show()


plot_feat_vs_output(X, Y)
