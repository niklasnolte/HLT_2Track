import os
from os.path import join

import numpy as np
from hlt2trk.utils.config import (Configuration, Locations, dirs,
                                  format_location, get_config)
from hlt2trk.utils.data import get_data, is_signal
from matplotlib import pyplot as plt

cfg = Configuration(features=["fdchi2", "sumpt", "vchi2", "minipchi2"])  # get_config()
if cfg.plot_style == "dark":
    plt.style.use(join(dirs.project_root, 'scripts/plot/paper-dark'))

df = get_data()
bkg = df[df.signal_type == 0]
sig = df[is_signal(cfg, df)]

fig, axes = plt.subplots(
    int(np.ceil(len(cfg.features) / 2)), 2, dpi=120, figsize=(16, 9)
)
for feature, ax in zip(cfg.features, axes.flatten()):
    l1 = bkg[feature].quantile(0.01)
    l2 = sig[feature].quantile(0.01)
    h1 = bkg[feature].quantile(0.99)
    h2 = sig[feature].quantile(0.99)
    range_ = (min(l1, l2), max(h1, h2))
    ax.hist(bkg[feature], bins=20, density=True, alpha=0.5, range=range_, label="bkg")
    ax.hist(sig[feature], bins=20, density=True, alpha=0.5, range=range_, label="sig")
    ax.legend()
    ax.set_title(feature, y=0.9)

savefile = "datahists_{signal_type}_{features}_{data_type}_{normalize}.pdf"
fig.savefig(
    os.path.join(dirs.plots, format_location(savefile, cfg))
)

plt.show()
