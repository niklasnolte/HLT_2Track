import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from hlt2trk.utils.config import Locations, format_location, get_config, dirs, Configs
from hlt2trk.utils.data import get_data, is_signal
from os.path import join
from hlt2trk.models import load_model, get_evaluator
from copy import copy

def add_model(model, cfg):
    cpy = copy(cfg)
    cpy.model = model
    return cpy


models = Configs.model[:3] # exclude everything except nn-regular, bdt, nn-inf

# Load configuration
cfg = get_config()
if cfg.plot_style == "dark":
    plt.style.use(join(dirs.project_root, "scripts/plot/paper-dark"))

target_cuts = {}
for m in models:
  with open(format_location(Locations.target_cut, add_model(m, cfg)), "r") as f:
      target_cuts[m] = float(f.read())

# Load configuration
eval_funs = {m: get_evaluator(add_model(m, cfg)) for m in models}
models = {m: load_model(add_model(m, cfg)) for m in models}
data = get_data(cfg)

data = data[data.eventtype != 0]  # need TOS -> no minbias
data = data[data.validation]
data = data[is_signal(cfg, data)]  # only care about TOS

# directions of the signal
# trk1_signal.. should be the same as trk2_signal.. (this is enforced in is_signal)
data["DX"] = (
    data["trk1_signal_TRUEENDVERTEX_X"] - data["trk1_signal_TRUEORIGINVERTEX_X"]
)
data["DY"] = (
    data["trk1_signal_TRUEENDVERTEX_Y"] - data["trk1_signal_TRUEORIGINVERTEX_Y"]
)
data["DZ"] = (
    data["trk1_signal_TRUEENDVERTEX_Z"] - data["trk1_signal_TRUEORIGINVERTEX_Z"]
)
# calculate phi
data["PHI"] = np.arctan2(data["DY"], data["DX"])
# calculate radial flight distance
data["Radial FD"] = np.log(np.sqrt(data["DX"] ** 2 + data["DY"] ** 2))
# calculate normal flight distance
data["FD"] = np.log(np.sqrt(data["DX"] ** 2 + data["DY"] ** 2 + data["DZ"] ** 2))
# calculate eta
data["ETA"] = np.arctanh(
    data["DZ"] / np.sqrt(data["DX"] ** 2 + data["DZ"] ** 2 + data["DY"] ** 2)
)

# restrict eta to [2,5] (only cuts away very few)
data = data[data.ETA <= 5]
data = data[data.ETA >= 2]

# get lifetime with scale for plotting
np.random.seed(2)
scale = np.random.uniform(1.5, 2.5, len(data))
data["Lifetime"] = data["trk1_signal_TRUETAU"] * scale * 1000 # ps
data["fdchi2"] = data["fdchi2"] + np.log(scale**2)
data["minipchi2"] = data["minipchi2"] + np.log(scale**2)

# restrict lifetime
data = data[data["Lifetime"] > (.0002 * scale)]

# for plotting, restrict flight distances (only sub% inefficiency)
# data = data[data["Radial FD"] < 6]
# data = data[data["FD"] < 50]

# Evaluate model on data
for m in models:
  data[f"pred_{m}"] = eval_funs[m](models[m], data[cfg.features].to_numpy()) > target_cuts[m]
# remove duplicate entries for signals
# there might be two 2-body SVs that belong to the same beauty/charm,
# don't want to count them twice
data = (
    data.groupby(["eventtype", "EventInSequence", "trk1_signal_TRUEENDVERTEX_Z"])
    .max()
    .reset_index()
)

charm = data[data.signal_type == 1]
beauty = data[data.signal_type == 4]

heavy_flavors = dict(beauty=beauty, charm=charm, heavy_flavor=data,)


def get_bins(data, nbins=20):
    return np.quantile(data, np.linspace(0, 1, nbins + 1))


with PdfPages(format_location(Locations.multi_eff_vs_kinematics, cfg)) as pdf:
    for name, flavor in heavy_flavors.items():
        for variable in ["Radial FD", "ETA", "PHI", "FD", "Lifetime"]:
            fig, axes = plt.subplots(
                1,
                len(models),
                sharex=True,
                sharey=True,
                figsize=(5 * len(models), 5),
                gridspec_kw={"width_ratios": [1] * (len(models) - 1) + [1.25]},
            )
            if variable == "Lifetime":
              fig.supxlabel("Lifetime [ps]", y = .04)
            else:
              fig.supxlabel(variable, y=.04)
            fig.supylabel("arbitrary units", y = .55)
            for m,ax in zip(models, axes):
              efficiencies = []
              uncertainties = []
              bins = get_bins(flavor[variable], 110)
              for (l, h) in zip(bins[:-1], bins[1:]):
                  mask = l <= flavor[variable]
                  mask &= flavor[variable] < h
                  p = flavor[f"pred_{m}"][mask].sum() / mask.sum()
                  efficiencies.append(p)
                  uncertainties.append(np.sqrt(p*(1-p) / mask.sum()))

              centers = (bins[:-1] + bins[1:]) / 2
              ax.set_ylim(0,1)
              ax.stairs(efficiencies, bins, label="efficiency")
              ax.errorbar(centers, efficiencies, yerr=uncertainties, fmt="none")
              counts, bins, patches = ax.hist(flavor[variable], bins=50, label="data distribution", alpha=.5)
              for bar in patches:
                bar.set_height(bar.get_height() / counts.max())

              #ax.grid(linestyle="--")
              #ax.grid(linestyle=":", which="minor")
              #ax.set_title(f"{cfg.model} {name} TOS efficiency at 660 kHz")
              if ax is axes[-1]:
                  ax.legend(loc="lower right")
            plt.tight_layout()
            pdf.savefig()
            plt.close()
