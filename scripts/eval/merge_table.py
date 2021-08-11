from hlt2trk.utils.config import dirs
import pandas as pd
import pickle

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

dfs = []
for model, df in models.items():
    df.set_index("mode", inplace=True)
    df.columns = pd.MultiIndex.from_product(
        ([model], ["dec", "tos"]), names=["model", "$\epsilon_{660}$"])
    dfs.append(df)
df = pd.concat(dfs, axis=1)


def make_table(df):
    df
    table = df.to_latex(
        column_format="c" * len(models) * 2 + "c",
        multicolumn=True,
        multicolumn_format="c",
        bold_rows=True,
        float_format="%.3f",
        index=True,
        escape=False,
        caption="efficiencies",
    )
    return table


with open(format_location(Locations.eff_table, cfg), "w") as f:
    f.writelines(make_table(df))
