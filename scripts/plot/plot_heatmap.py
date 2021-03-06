import itertools
import typing as t
from os.path import join
from re import M

import numpy as np
from hlt2trk.utils.config import (Locations, dirs, feature_repr,
                                  format_location, get_config)
import matplotlib
matplotlib.rcParams.update({'font.size': 15})
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

cfg = get_config()
if cfg.plot_style == "dark":
    plt.style.use(join(dirs.project_root, 'scripts/plot/paper-dark'))

X, Y = np.load(format_location(Locations.gridXY, cfg)).values()
with open(format_location(Locations.auc_acc, cfg), "r") as f:
    auc, _, _ = map(float, f.read().split(","))
with open(format_location(Locations.target_cut, cfg), "r") as f:
    target_cut = float(f.read())


def plot_heatmap(
        x: np.ndarray,
        y: np.ndarray,
        params: t.Optional[dict] = None,
):
    with PdfPages(format_location(Locations.heatmap, cfg)) as pdf:
        combinations = list(itertools.combinations(range(x.shape[1]), 2))
        if x.shape[1] == 4:
            # put minipchi2 vs sumpt in front
            combinations.pop(combinations.index((1, 3)))
            combinations.insert(0, (3, 1))
        for idxs in combinations:
            x0u = np.unique(x[:, idxs[0]])
            x1u = np.unique(x[:, idxs[1]])
            y_meaned = [
                y[(x[:, idxs[0]] == x0i) & (x[:, idxs[1]] == x1i)].mean()
                for x0i in x0u for x1i in x1u]
            y_meaned = np.array(y_meaned)

            xs = np.array([list(x) for x in itertools.product(x0u, x1u)])
            x0 = xs[:, 0]
            x1 = xs[:, 1]

            fig = plt.figure()
            ax: plt.Axes = fig.add_subplot()

            sc = ax.scatter(x0, x1, c=y_meaned, cmap=plt.cm.RdBu,
                            s=150, marker="s", **(params or {}))
            shape_ = int(np.sqrt(len(x0)))
            ax.contour(
                x0.reshape(-1, shape_),
                x1.reshape(-1, shape_),
                y_meaned.reshape(-1, shape_),
                levels=[target_cut])
            ax.set_xlabel(feature_repr(cfg.features[idxs[0]]))
            ax.set_ylabel(feature_repr(cfg.features[idxs[1]]))
            #ax.set_title(cfg.model)
            ax.text(0, 1.05, f"auc: {auc:.3f}",
                    transform=ax.transAxes,
                    horizontalalignment="left",
                    verticalalignment="top")
            plt.colorbar(sc)
            plt.tight_layout()
            pdf.savefig()
            plt.close()


plot_heatmap(X, Y)
