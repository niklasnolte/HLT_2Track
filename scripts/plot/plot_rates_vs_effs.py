import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from hlt2trk.utils.config import Locations, format_location, get_config, evttypes, dirs, onetrack_target_rate, twotrack_target_rate
from os.path import join

# Load configuration
cfg = get_config()
if cfg.plot_style == "dark":
    plt.style.use(join(dirs.project_root, 'scripts/plot/paper-dark'))

if cfg.onetrack:
  full_effs_loc = Locations.onetrack_full_effs
  target_effs_loc = Locations.onetrack_target_effs
  rate_vs_eff_loc = Locations.onetrack_rate_vs_eff
else:
  full_effs_loc = Locations.full_effs
  target_effs_loc = Locations.target_effs
  rate_vs_eff_loc = Locations.rate_vs_eff



with open(format_location(full_effs_loc, cfg), "rb") as f:
    efficiencies = pickle.load(f)
with open(format_location(target_effs_loc, cfg), "rb") as f:
    target_effs = pickle.load(f)

rates = efficiencies.pop(0)
if cfg.onetrack:
  rates = rates["eff"]
  # remove minbias
  target_effs = target_effs.iloc[1:]


def is_charm(evttype):
    return evttype > 20000000  # first digit 2 is charm


with PdfPages(format_location(rate_vs_eff_loc, cfg)) as pdf:
    for tos in [False, True]:
        eff_keyword = "tos_eff" if tos else "eff"

        if cfg.onetrack:
          target_rate = onetrack_target_rate
        else:
          target_rate = twotrack_target_rate

        _, ax = plt.subplots()

        for mode, data in efficiencies.items():
            eff = data[eff_keyword]
            ax.plot(
                rates, eff, c=f"C{int(is_charm(evttypes[mode]))}"
            )
        charm_eff = target_effs[eff_keyword][is_charm(target_effs["mode"])].mean()
        beauty_eff = target_effs[eff_keyword][~is_charm(target_effs["mode"])].mean()
        ax.plot([target_rate], [0], c="C0", label=f"beauty: {beauty_eff:.4f}")
        ax.plot([target_rate], [0], c="C1", label=f"charm: {charm_eff:.4f}")
        ax.set_xlabel("rate (kHz)")
        ax.set_ylabel("efficiency")
        ax.axvline(x=target_rate, color="red")
        ax.grid(linestyle="--")
        ax.grid(linestyle=":", which="minor")
        ax.set_title((cfg.model or "1-Track") + (" TOS" if tos else ""))
        ax.legend(loc="lower right", title=f"{' TOS' if tos else ''} eff at {target_rate}kHz")
        pdf.savefig()
        plt.close()
