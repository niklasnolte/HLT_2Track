import pickle
from typing import OrderedDict
from matplotlib import transforms
import numpy as np
from copy import copy
from os.path import join
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.size": 20,
                            "text.usetex" : True,
                            "font.family": "serif",
                            "font.sans-serif": ["Computer Modern Roman"]})
from hlt2trk.utils.config import (Configs, Locations, format_location,
                                  get_config, dirs)

cfg = get_config()
if cfg.plot_style == "dark":
    plt.style.use(join(dirs.project_root, 'scripts/plot/paper-dark'))


def add_model(model, cfg):
    cpy = copy(cfg)
    cpy.model = model
    return cpy

models_of_interest = Configs.model

locations = [format_location(Locations.target_effs, add_model(m, cfg))
             for m in models_of_interest]

models = OrderedDict()
for file, model in zip(locations, models_of_interest):
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
    return (x - effs["nn-regular"])  # / effs["nn-regular"]


mask_b = modes[model] < 20000000
mask_c = modes[model] > 20000000
with PdfPages(format_location(Locations.violins, cfg)) as pdf:
    eff_kind = ["", "TOS"]
    mask_kind = ["beauty", "charm"]
    for j, effs in enumerate([dec_effs, tos_effs]):
        fig, axes = plt.subplots(1, 2, figsize=(16, 4))
        for k, mask in enumerate([mask_b, mask_c]):
            violins = OrderedDict((model, frac(effs[model])[mask])
                       for model in models_of_interest[::-1] if model != "nn-regular")

            inds = np.arange(1, len(violins) + 1)
            means = {name: np.mean(v) for name, v in violins.items()}
            mins = {name: np.min(v) for name, v in violins.items()}

            axes[k].axvline(0, ls=':', c='red', alpha=.8)
            parts = axes[k].violinplot(violins.values(), vert=False, showextrema=True,
                                  showmedians=True, showmeans=False)
            for y, xs in enumerate(violins.values()):
                axes[k].scatter(xs, [y + 1] * len(xs))
            axes[k].scatter(means.values(), inds, marker='o', color='white', s=30, zorder=3)

            for pc in parts['bodies']:
                pc.set_facecolor('#7751ae')
                pc.set_edgecolor('black')
                pc.set_alpha(.8)
            axes[k].set_yticks([])
            axes[k].set_yticklabels([])
            axes[k].set_title(" ".join([eff_kind[j], mask_kind[k]]))
            x0, x1 = axes[k].get_xlim()
            for i, model in enumerate(violins.keys()):
                x0 = mins[model]
                # axes[k].text(x0 + 0.05 * abs(x0), i + 1.1, model)
                axes[k].text(x0 + 0.05 * abs(x0), i + 1.08, model)
            xlabel = r"$\epsilon_{x} - \epsilon_{\mathrm{NN}}$"
            axes[k].set_xlabel(xlabel)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    for j, effs in enumerate([dec_effs, tos_effs]):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        for k, mask in enumerate([mask_b, mask_c]):
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
                ax.set_title(" ".join([eff_kind[j], mask_kind[k]]))
                # xlabel = r"$\frac{\epsilon_{" + model + r"} - \epsilon_{\mathrm{NN}}}"
                #     r"{\epsilon_{\mathrm{NN}}}$"
                xlabel = r"$\epsilon_{\mathrm{" + model + r"}} - \epsilon_{\mathrm{NN}}$"
                ax.set_xlabel(xlabel)
                pdf.savefig()
                plt.close()

# plt.savefig(join(dirs.plots, f"candle-plot_{'_'.join(strings)}.pdf"))
