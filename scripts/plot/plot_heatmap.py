import typing as t
from os.path import join

import numpy as np
from hlt2trk.utils.config import Locations, dirs, format_location, get_config
from matplotlib import pyplot as plt

cfg = get_config()
if cfg.plot_style == "dark":
    plt.style.use(join(dirs.project_root, 'scripts/plot/paper-dark'))

X, Y = np.load(format_location(Locations.gridXY, cfg)).values()
with open(format_location(Locations.auc_acc, cfg), "r") as f:
    auc, acc, cut = map(float, f.read().split(","))


def plot_heatmap(
        x: np.ndarray,
        y: np.ndarray,
        params: t.Optional[dict] = None,
):
    x0 = x[:, 0]
    x1 = x[:, 1]
    fig = plt.figure()
    ax: plt.Axes = fig.add_subplot()

    c = plt.cm.RdBu(y)
    sc = ax.scatter(x0, x1, c=y, cmap=plt.cm.RdBu, s=150, marker="s", **(params or {}))
    eps = 1e-2
    mask = (y < cut + eps) & (y > cut - eps)
    ax.scatter(x0[mask], x1[mask], c="purple", s=10, marker=".")
    ax.set_xlabel(cfg.features[0])
    ax.set_ylabel(cfg.features[1])
    ax.set_title(cfg.model)
    ax.text(.96, .98, f"auc: {auc:.3f}\nacc: {acc:.3f}\ncut: {cut:.3f}")
    plt.colorbar(sc)
    plt.savefig(format_location(Locations.heatmap, cfg))


plot_heatmap(X, Y)
