import numpy as np
from matplotlib import pyplot as plt
import hlt2trk.data.meta_info as meta

sig, bkg = meta.get_data()

fig, axes = plt.subplots(int(np.ceil(len(meta.features) / 2)), 2, dpi=120, figsize=(3, 2))
for feature, ax in zip(meta.features, axes.flatten()):
    ax.hist(bkg[feature], bins=20, density=True, alpha=0.5)
    ax.hist(sig[feature], bins=20, density=True, alpha=0.5)
    ax.set_title(feature, y=.9)
# fig.savefig(join(meta.locations.project_root, 'plots/data/LHCbHists.png'))

plt.show()
