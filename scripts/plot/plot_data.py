import numpy as np
from matplotlib import pyplot as plt
import hlt2trk.utils.meta_info as meta
import os

sig, bkg = meta.get_data()

fig, axes = plt.subplots(
    int(np.ceil(len(meta.features) / 2)), 2, dpi=120, figsize=(16, 9)
)
for feature, ax in zip(meta.features, axes.flatten()):
    l1 = bkg[feature].quantile(0.01)
    l2 = sig[feature].quantile(0.01)
    h1 = bkg[feature].quantile(0.99)
    h2 = sig[feature].quantile(0.99)
    range_ = (min(l1, l2), max(h1, h2))
    ax.hist(bkg[feature], bins=20, density=True, alpha=0.5, range=range_, label="bkg")
    ax.hist(sig[feature], bins=20, density=True, alpha=0.5, range=range_, label="sig")
    ax.legend()
    ax.set_title(feature, y=0.9)
fig.savefig(
    os.path.join(meta.locations.project_root, f"plots/data_{meta.path_suffix}.png")
)

plt.show()
