import pickle
import numpy as np
from copy import copy
from os.path import join
from matplotlib.backends.backend_pdf import PdfPages
import itertools
from collections import defaultdict

import matplotlib.pyplot as plt
from hlt2trk.utils.config import (Configs, Locations, format_location,
                                  get_config, dirs)

cfg = get_config()
if cfg.plot_style == "dark":
    plt.style.use(join(dirs.project_root, 'scripts/plot/paper-dark'))


def add_seed_and_model(seed, model, cfg):
    cpy = copy(cfg)
    cpy.model = model
    cpy.seed = seed
    return cpy


seed_model_configs = list(itertools.product(Configs.model, Configs.seed))

locations = [format_location(Locations.auc_acc, add_seed_and_model(seed, m, cfg))
                      for (m, seed) in seed_model_configs]


metrics = defaultdict(lambda :defaultdict(list))
for file, (model, seed) in zip(locations, seed_model_configs):
    with open(file, "r") as f:
        auc, acc, _ = map(float, f.read().split(","))
        metrics["auc"][model].append(auc)
        metrics["acc"][model].append(acc)


with PdfPages(format_location(Locations.seed_violins, cfg)) as pdf:
  for metric in ["auc", "acc"]:
    violins = {model: np.array(metrics[metric][model]) - np.array(metrics[metric]["nn-regular"])
                for model in Configs.model if model != "nn-regular"}
    inds = np.arange(1, len(violins) + 1)
    means = {name: np.mean(v) for name, v in violins.items()}
    mins = {name: np.min(v) for name, v in violins.items()}
    maxs = {name: np.max(v) for name, v in violins.items()}

    fig, ax = plt.subplots(1, 1)
    ax.axvline(0, ls=':', c='red', alpha=.8)
    parts = ax.violinplot(violins.values(), vert=False, showextrema=True,
                          showmedians=True, showmeans=False)
    for y, xs in enumerate(violins.values()):
        ax.scatter(xs, [y + 1] * len(xs))
    ax.scatter(means.values(), inds, marker='o', color='white', s=30, zorder=3)

    for pc in parts['bodies']:
        pc.set_facecolor('#7751ae')
        pc.set_edgecolor('black')
        pc.set_alpha(.8)
    ax.set_yticks(range(1, len(violins) + 1))
    ax.set_yticklabels([])
    # ax.set_title(" - ".join([eff_kind[j], mask_kind[k]]))
    x0, x1 = ax.get_xlim()
    for i, model in enumerate(violins.keys()):
        x0 = maxs[model]
        # ax.text(x0 + 0.05 * abs(x0), i + 1.1, model)
        ax.text(x0 + 0.05 * abs(x0), i + 1.08, model)
        xlabel = metric + r"$_{x} - $" + metric + r"$_{\mathrm{NN}}$"
    ax.set_xlabel(xlabel)
    pdf.savefig()
    plt.close()

    for model, violin in violins.items():
        fig, ax = plt.subplots(1, 1)
        x0 = mins[model]
        ax.text(x0 + 0.05 * abs(x0), 1.08, model)
        # ax.axvline(0, ls='--', c='k', alpha=.8)
        ax.violinplot(violin, vert=False, showextrema=False,
                      showmedians=True, showmeans=False)
        ax.scatter(violin, [1] * len(violin))
        ax.scatter(means[model], 1, marker='o', color='white', s=30, zorder=3)
        ax.set_yticks([1])
        ax.set_yticklabels([])
        # ax.set_title(" - ".join([eff_kind[j], mask_kind[k]]))
        # xlabel = r"$\frac{\epsilon_{" + model + r"} - \epsilon_{\mathrm{NN}}}"
        #     r"{\epsilon_{\mathrm{NN}}}$"
        xlabel = metric + r"$_{\mathrm{" + model + r"}} - $" + metric + r"$_{\mathrm{NN}}$"
        ax.set_xlabel(xlabel)
        pdf.savefig()
        plt.close()

# plt.savefig(join(dirs.plots, f"candle-plot_{'_'.join(strings)}.pdf"))
