import pickle
from matplotlib import transforms
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
dec_effs = {}
modes = {}
for model in models:
    dec_effs[model] = models[model]["eff"].values
    tos_effs[model] = models[model]["tos_eff"].values
    modes[model] = models[model]["mode"].values


def frac(x):
    # return x
    return (x - effs["nn-regular"])  # / effs["nn-regular"]


mask_b = modes[model] < 20000000
mask_c = modes[model] > 20000000
with PdfPages(format_location(Locations.violins, cfg)) as pdf:
    eff_kind = ["dec", "tos"]
    mask_kind = ["b", "c"]
    for k, mask in enumerate([mask_b, mask_c]):
        for j, effs in enumerate([dec_effs, tos_effs]):
            violins = {model: frac(effs[model])[mask]
                       for model in Configs.model if model != "nn-regular"}

            inds = np.arange(1, len(violins) + 1)
            means = {name: np.mean(v) for name, v in violins.items()}
            mins = {name: np.min(v) for name, v in violins.items()}

            fig, ax = plt.subplots(1, 1)
            ax.axvline(0, ls='--', c='k', alpha=.8)
            ax.violinplot(violins.values(), vert=False, showextrema=True,
                          showmedians=True, showmeans=False)
            for y, xs in enumerate(violins.values()):
                ax.scatter(xs, [y + 1] * len(xs))
            ax.scatter(means.values(), inds, marker='o', color='white', s=30, zorder=3)
            ax.set_yticks(range(1, len(violins) + 1))
            ax.set_yticklabels([])
            ax.set_title(" - ".join([eff_kind[j], mask_kind[k]]))
            x0, x1 = ax.get_xlim()
            for i, model in enumerate(violins.keys()):
                x0 = mins[model]
                # ax.text(x0 + 0.05 * abs(x0), i + 1.1, model)
                ax.text(x0 + 0.05 * abs(x0), i + 1.08, model)
            xlabel = r"$\epsilon_{x} - \epsilon_{\mathrm{NN}}$"
            ax.set_xlabel(xlabel)
            pdf.savefig()

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
                ax.set_title(" - ".join([eff_kind[j], mask_kind[k]]))
                # xlabel = r"$\frac{\epsilon_{" + model + r"} - \epsilon_{\mathrm{NN}}}" +\
                #     r"{\epsilon_{\mathrm{NN}}}$"
                xlabel = r"$\epsilon_{" + model + r"} - \epsilon_{\mathrm{NN}}$"
                ax.set_xlabel(xlabel)
                pdf.savefig()

# plt.savefig(join(dirs.plots, f"candle-plot_{'_'.join(strings)}.pdf"))
