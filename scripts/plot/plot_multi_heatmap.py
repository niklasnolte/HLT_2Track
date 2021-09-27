import itertools
import typing as t
from os.path import join
from re import M
from copy import copy

import numpy as np
from hlt2trk.utils.config import (
    Locations,
    dirs,
    feature_repr,
    Configs,
    format_location,
    get_config,
)
import matplotlib

matplotlib.rcParams.update({"font.size": 15})
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

cfg = get_config()
if cfg.plot_style == "dark":
    plt.style.use(join(dirs.project_root, "scripts/plot/paper-dark"))


def add_model(model, cfg):
    cpy = copy(cfg)
    cpy.model = model
    return cpy


XYs = {}
aucs = {}
target_cuts = {}

for m in Configs.model:
    XYs[m] = np.load(format_location(Locations.gridXY, add_model(m, cfg))).values()
    with open(format_location(Locations.auc_acc, add_model(m, cfg)), "r") as f:
        auc, _, _ = map(float, f.read().split(","))
        aucs[m] = auc
    with open(format_location(Locations.target_cut, add_model(m, cfg)), "r") as f:
        target_cuts[m] = float(f.read())


with PdfPages(format_location(Locations.heatmap_agg, cfg)) as pdf:
    nfeatures = len(cfg.features)
    combinations = list(itertools.combinations(range(nfeatures), 2))
    if nfeatures == 4:
        # put minipchi2 vs sumpt in front
        combinations.pop(combinations.index((1, 3)))
        combinations.insert(0, (3, 1))
    for idxs in combinations:
        fig, axes = plt.subplots(
            1,
            len(Configs.model),
            sharex=True,
            sharey=True,
            figsize=(5 * len(Configs.model), 5),
            gridspec_kw={"width_ratios": [1] * (len(Configs.model) - 1) + [1.25]},
        )
        fig.supxlabel(feature_repr(cfg.features[idxs[0]]))
        fig.supylabel(feature_repr(cfg.features[idxs[1]]))
        for i, (m, ax) in enumerate(zip(Configs.model, axes)):
            x, y = XYs[m]
            auc = aucs[m]
            target_cut = target_cuts[m]
            x0u = np.unique(x[:, idxs[0]])
            x1u = np.unique(x[:, idxs[1]])
            y_meaned = [
                y[(x[:, idxs[0]] == x0i) & (x[:, idxs[1]] == x1i)].mean()
                for x0i in x0u
                for x1i in x1u
            ]
            y_meaned = np.array(y_meaned)

            xs = np.array([list(x) for x in itertools.product(x0u, x1u)])
            x0 = xs[:, 0]
            x1 = xs[:, 1]

            sc = ax.scatter(x0, x1, c=y_meaned, cmap=plt.cm.RdBu, s=150, marker="s")
            shape_ = int(np.sqrt(len(x0)))
            ax.contour(
                x0.reshape(-1, shape_),
                x1.reshape(-1, shape_),
                y_meaned.reshape(-1, shape_),
                levels=[target_cut],
            )
            # ax.set_title(cfg.model)
            # ax.text(0, 1.05, f"auc: {auc:.3f}",
            #         transform=ax.transAxes,
            #         horizontalalignment="left",
            #         verticalalignment="top")
            if i > 0:
                plt.tick_params(top=False, bottom=True, left=True, right=False)
        plt.colorbar(sc)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
