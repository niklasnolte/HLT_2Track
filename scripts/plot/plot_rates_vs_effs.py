import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from hlt2trk.utils.config import Locations, format_location, get_config, evttypes

# Load configuration
cfg = get_config()

with open(format_location(Locations.full_effs, cfg), "rb") as f:
    efficiencies = pickle.load(f)
with open(format_location(Locations.target_effs, cfg), "rb") as f:
    target_effs = pickle.load(f)

rates = efficiencies.pop(0)

def is_charm(evttype):
    return evttype > 20000000  # first digit 2 is charm

with PdfPages(format_location(Locations.rate_vs_eff, cfg)) as pdf:
  for tos in [False, True]:
    eff_keyword = "tos_eff" if tos else "eff"
    target_rate = 660

    _, ax = plt.subplots()
    
    for mode, data in efficiencies.items():
        if tos:
            eff = data[eff_keyword]
        else:
            eff = data[eff_keyword]
        ax.plot(
            rates, eff, c=f"C{int(is_charm(evttypes[mode]))}"
        )
    charm_eff = target_effs[eff_keyword][is_charm(target_effs["mode"])].mean()
    beauty_eff = target_effs[eff_keyword][~is_charm(target_effs["mode"])].mean()

    ax.plot([0],[0], c="C0", label=f"beauty: {beauty_eff:.4f}")
    ax.plot([0],[0], c="C1", label=f"charm: {charm_eff:.4f}")
    ax.set_xlabel("rate (kHz)")
    ax.set_ylabel("efficiency")
    ax.axvline(x=target_rate, color="red")
    ax.grid(linestyle="--")
    ax.grid(linestyle=":", which="minor")
    ax.set_title(cfg.model + (" TOS" if tos else ""))
    ax.legend(loc="lower right", title=f"{' TOS' if tos else ''} eff at 660Hz")
    pdf.savefig()
    plt.close()
