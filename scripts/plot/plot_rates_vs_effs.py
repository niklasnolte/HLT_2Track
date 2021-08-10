import pickle

import matplotlib.pyplot as plt
from hlt2trk.utils.config import Locations, format_location, get_config

# Load configuration
cfg = get_config()


def plot_rates_vs_effs(efficiencies):
    rates = efficiencies.pop(0)
    target_rate = 660

    _, ax = plt.subplots()
    for mode, data in efficiencies.items():
        eff = data["eff"]
        tos_eff = data["tos_eff"]
        ax.plot(
            rates, eff, label=f"{mode:^4}",
            c=f"C{mode}")
        ax.plot(
            rates, tos_eff, label=f"{mode:^4}",
            ls="--", c=f"C{mode}")

    ax.set_xlabel("rate (kHz)")
    ax.set_ylabel("efficiency")
    # ax.set_ylim(0, max_eff)
    ax.axvline(x=target_rate, color="red")
    ax.grid(linestyle="--")
    ax.grid(linestyle=":", which="minor")
    ax.set_title(cfg.model)
    ax.legend(loc="lower right", title="mode / eff at 660Hz")
    plt.savefig(format_location(Locations.rate_vs_eff, cfg))


with open(format_location(Locations.full_effs, cfg), "rb") as f:
    efficiencies = pickle.load(f)
plot_rates_vs_effs(efficiencies)
