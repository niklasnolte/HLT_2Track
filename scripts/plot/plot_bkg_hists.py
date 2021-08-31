import os
from os.path import join

import numpy as np
from hlt2trk.utils.config import get_config_from_file, dirs
from hlt2trk.utils.data import get_data, is_signal
from matplotlib import pyplot as plt

cfg = get_config_from_file(join(dirs.project_root, "config.yml"))

if cfg.plot_style == "dark":
    plt.style.use(join(dirs.project_root, 'scripts/plot/paper-dark'))

df = get_data(cfg)
bkg = df[df.signal_type == 0]
sig = df[is_signal(cfg, df)]

if cfg.data_type == "lhcb":
    # minbias + svs of which the tracks associated pvs are at least 10mm away from the signal pv
    bkg_from_mb = bkg[bkg.eventtype == 0]
    bkg_fromsignal = bkg[(bkg.eventtype != 0) & (bkg.trk1_fromHFPV + bkg.trk2_fromHFPV == 0)]
    sig = sig[sig.eventtype != 0]
    sig = sig.sample(frac = .1, random_state = 1)


fig, axes = plt.subplots(
    int(np.ceil(len(cfg.features) / 2)), 2, dpi=120, figsize=(16, 9)
)
for feature, ax in zip(cfg.features, axes.flatten()):
    l1 = bkg_from_mb[feature].quantile(0.01)
    l2 = bkg_fromsignal[feature].quantile(0.01)
    l3 = sig[feature].quantile(0.01)
    h1 = bkg_from_mb[feature].quantile(0.99)
    h2 = bkg_fromsignal[feature].quantile(0.99)
    h3 = sig[feature].quantile(0.99)
    range_ = (min(l1, l2), max(h1, h2))
    ax.hist(bkg_from_mb[feature], bins=20, density=True, alpha=0.5, range=range_, label="bkg_from_bkg")
    ax.hist(bkg_fromsignal[feature], bins=20, density=True, alpha=0.5, range=range_, label="bkg_from_sig")
    ax.hist(sig[feature], bins=20, density=True, alpha=0.5, range=range_, label="sig")
    ax.legend()
    ax.set_title(feature, y=0.9)

# savefile = "datahists_{signal_type}_{features}_{data_type}_{normalize}.pdf"
# fig.savefig(
#     os.path.join(dirs.plots, format_location(savefile, cfg))
# )

plt.show()
