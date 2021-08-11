import matplotlib.pyplot as plt
from hlt2trk.utils.config import format_location, get_config, Configs, format_location, Locations
import pickle
from copy import copy

cfg = get_config()

def add_model(model, cfg):
  cpy = copy(cfg)
  cpy.model = model
  return cpy

locations = [format_location(Locations.target_effs, add_model(m, cfg)) for m in Configs.model]

models = {}
for file,model in zip(locations, Configs.model):
    with open(file, "rb") as f:
        models[model] = pickle.load(f)

tos_effs = {}
effs = {}
for model in models:
  effs[model] = models[model]["eff"].values
  tos_effs[model] = models[model]["tos_eff"].values


fig, ax = plt.subplots(1,1)

violins = {model:effs[model] - effs["nn-regular"] for model in models if model != "nn-regular"}

ax.violinplot(violins.values(), vert=False, showextrema=True, showmedians=True)
for y, xs in enumerate(violins.values()):
  ax.scatter(xs, [y+1]*len(xs))
ax.set_yticks(range(1,len(violins) + 1))
ax.set_yticklabels(violins.keys())
ax.set_xlabel("efficiency difference ($\epsilon_{\mathrm{unconstrained}} - \epsilon_{x}$)")
plt.savefig(format_location(Locations.violins, cfg))
