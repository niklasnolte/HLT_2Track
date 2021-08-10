import numpy as np
import matplotlib.pyplot as plt

from os.path import join
import os
from hlt2trk.utils.config import dirs, format_location, get_config, Configs, format_location, Locations
import numpy as np
import pickle

# cfg = get_config()

# locations = [format_location(cfg, Locations.target_effs) + ]

def conditions(f):
    strings = ["fdchi2+sumpt+vchi2+minipchi2",
               "ipcuttrain:6",
               "unnormed",
               "svchi2:20",
               ]
    return all([string in f for string in strings])

files = os.listdir(dirs.results_eff)
subset = [f for f in files if f.startswith("target-eff") and conditions(f)]
order = ["qda", "lda", "gnb", "nn-inf", "nn-one", "nn-regular", "bdt"]
models = {}
if len(subset) > 0:
    for fname in subset:
        file = join(dirs.results_eff, fname)
        with open(file, "rb") as f:
            models[fname.split("_")[1]] = pickle.load(f)

tos_effs = {}
effs = {}
for model in models:
  effs[model] = models[model]["eff"].values
  tos_effs[model] = models[model]["tos_eff"].values


fig, ax = plt.subplots(1,1)

violins = [effs["bdt"] - effs["nn-regular"], effs["nn-inf"] - effs["nn-regular"], effs["nn-one"]-effs["nn-regular"]]

ax.violinplot(violins, vert=False, showextrema=True, showmedians=True)
for y, xs in enumerate(violins):
  ax.scatter(xs, [y+1]*len(xs))
ax.set_yticks(range(1,len(violins) + 1))
ax.set_yticklabels(["bdt", "nn-inf", "nn-one"])
ax.set_xlabel("efficiency difference ($\epsilon_{\mathrm{unconstrained}} - \epsilon_{x}$)")
plt.show()
