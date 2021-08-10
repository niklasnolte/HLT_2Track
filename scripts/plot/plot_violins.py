import pickle
import numpy as np
from copy import copy
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.pyplot as plt
from hlt2trk.utils.config import (Configs, Locations, format_location,
                                  get_config)

cfg = get_config()


def add_model(model, cfg):
    cpy = copy(cfg)
    cpy.model = model
    return cpy


locations = [format_location(Locations.target_effs, add_model(m, cfg))
             for m in Configs.model]

models = {}
for file, model in zip(locations, Configs.model):
    with open(file, "rb") as f:
        models[model] = pickle.load(f)

tos_effs = {}
effs = {}
for model in models:
    effs[model] = models[model]["eff"].values
    tos_effs[model] = models[model]["tos_eff"].values


def frac(x):
    # return x
    return (x - effs["nn-regular"]) / effs["nn-regular"]


valid_models = [model for model in models.keys() if model != "nn-regular"]
violins = {model: frac(effs[model]) for model in valid_models}

inds = np.arange(1, len(violins) + 1)
means = [np.mean(v) for v in violins.values()]

fig, ax = plt.subplots(1, 1)
ax.violinplot(violins, vert=False, showextrema=True,
              showmedians=True, showmeans=False)

with PdfPages(format_location(Locations.violins, cfg)) as pdf:
    for y, xs in enumerate(violins):
        ax.scatter(xs, [y + 1] * len(xs))
    ax.scatter(means, inds, marker='o', color='white', s=30, zorder=3)
    ax.set_yticks(range(1, len(violins) + 1))
    ax.set_yticklabels(valid_models)
    ax.set_xlabel(
        r"$\frac{\epsilon_{x} - \epsilon_{\mathrm{NN}}}"
        r"{\epsilon_{\mathrm{NN}}}$")

plt.savefig(format_location(Locations.violins, cfg))
# plt.savefig(join(dirs.plots, f"candle-plot_{'_'.join(strings)}.pdf"))
