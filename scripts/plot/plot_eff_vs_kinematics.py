import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from hlt2trk.utils.config import Locations, format_location, get_config, dirs
from hlt2trk.utils.data import get_data, is_signal
from os.path import join
from hlt2trk.models import load_model, get_evaluator

# Load configuration
cfg = get_config()
if cfg.plot_style == "dark":
    plt.style.use(join(dirs.project_root, "scripts/plot/paper-dark"))

with open(format_location(Locations.target_cut, cfg), "r") as f:
    target_cut = float(f.read())

# Load configuration
eval_fun = get_evaluator(cfg)
model = load_model(cfg)
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
data["Radial FD"] = np.sqrt(data["DX"] ** 2 + data["DY"] ** 2)
# calculate eta
data["ETA"] = np.arctanh(
    data["DZ"] / np.sqrt(data["DX"] ** 2 + data["DZ"] ** 2 + data["DY"] ** 2)
)
# restrict eta to [2,5] (only cuts away very few)
data = data[data.ETA <= 5]
data = data[data.ETA >= 2]

# for plotting, restrict radial flight distance to < 6
data = data[data["Radial FD"] < 6]

# Evaluate model on data
data["pred"] = eval_fun(model, data[cfg.features].to_numpy()) > target_cut
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


with PdfPages(format_location(Locations.eff_vs_kinematics, cfg)) as pdf:
    for name, flavor in heavy_flavors.items():
        for variable in ["Radial FD", "ETA", "PHI"]:
            efficiencies = []
            uncertainties = []
            bins = get_bins(flavor[variable])
            for (l, h) in zip(bins[:-1], bins[1:]):
                mask = l <= flavor[variable]
                mask &= flavor[variable] < h
                p = flavor.pred[mask].sum() / mask.sum()
                efficiencies.append(p)
                uncertainties.append(np.sqrt(p*(1-p) / mask.sum()))

            _, ax = plt.subplots()
            centers = (bins[:-1] + bins[1:]) / 2
            ax.stairs(efficiencies, bins, label="efficiency")
            ax.errorbar(centers, efficiencies, yerr=uncertainties, fmt="none")
            ax.hist(flavor[variable], alpha=.5, label="distribution", density=True, bins=50)
            ax.set_xlabel(variable)
            ax.set_ylabel("arbitrary units")
            ax.grid(linestyle="--")
            ax.grid(linestyle=":", which="minor")
            ax.set_title(f"{cfg.model} {name} TOS efficiency at 660 kHz")
            ax.legend(loc="lower right")
            pdf.savefig()
            plt.close()
