import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from hlt2trk.utils.config import Locations, format_location, get_config, dirs
from hlt2trk.utils.data import get_data
from os.path import join
from hlt2trk.models import load_model, get_evaluator

# Load configuration
cfg = get_config()
if cfg.plot_style == "dark":
    plt.style.use(join(dirs.project_root, 'scripts/plot/paper-dark'))

with open(format_location(Locations.target_cut, cfg), "w") as f:
    cut = int(f.read())

# Load configuration
eval_fun = get_evaluator(cfg)
model = load_model(cfg)
data = get_data(cfg)

data = data[data.eventtype != 0] # minbias is not needed here, only efficiencies
data = data[data.validation]

# Evaluate model on data
data["pred"] = eval_fun(model, data[cfg.features].to_numpy())
data["tos_pred"] = data.pred * (data.signal_type > 0)


# max aggregation is good for all things that we care about here
# which is predictions, and the per-event truth kinematics
evt_grp = ["eventtype", "EventInSequence"]
data = data.groupby(evt_grp).agg(max).reset_index()


def is_charm(evttype):
    return evttype > 20000000  # first digit 2 is charm

charm = data[is_charm(data.eventtype)]
beauty = data[~is_charm(data.eventtype)]

def get_bins(data, nbins=10):
    return np.quantile(data, np.linspace(0, 1, nbins, endpoint=False))


beauty_phi_bins = get_bins(beauty.signal_TRUEETA)


with PdfPages(format_location(Locations.eff_vs_kinematics, cfg)) as pdf:
    for tos in [False, True]:
        tos_kw = "tos_pred" if tos else "pred"

        _, ax = plt.subplots()

        ax.hist(beauty[tos_kw], bins=beauty_phi_bins, histtype="step", label="Beauty", color="C0", normed=True)
        ax.set_xlabel("Heavy Flavour ETA")
        ax.set_ylabel("efficiency")
        ax.grid(linestyle="--")
        ax.grid(linestyle=":", which="minor")
        ax.set_title(cfg.model + (" TOS" if tos else ""))
        ax.legend(loc="lower right", title=f"{' TOS' if tos else ''} eff at 660Hz")
        pdf.savefig()
        plt.close()
