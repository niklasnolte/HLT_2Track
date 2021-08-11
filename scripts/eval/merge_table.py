import pickle
from copy import copy

import pandas as pd
from hlt2trk.utils.config import (Configs, Locations, format_location,
                                  get_config)

cfg = get_config()


def add_model(model, cfg):
    cpy = copy(cfg)
    cpy.model = model
    return cpy


locations = [format_location(Locations.target_effs, add_model(m, cfg))
             for m in Configs.model]

models = {}
for file, model in zip(locations, Configs.model):
    with open(file, "rb") as f:
        models[model] = pickle.load(f)

dfs = []
for model, df in models.items():
    df.set_index("mode", inplace=True)
    df.columns = pd.MultiIndex.from_product(
        ([model], ["dec", "tos"]), names=["model", r"$\epsilon_{660}$"])
    dfs.append(df)
df = pd.concat(dfs, axis=1)
mean = df.mean(axis=0).rename("mean")
std = df.std(axis=0).rename("std")
df = df.sort_index(axis=0)
idxmax = pd.concat(
    [((df["nn-regular"] - df[model]) / df["nn-regular"]).idxmax()
     for model in df.columns.get_level_values(0)[:: 2]]).rename(r"max $\delta$ nn")
idxmin = pd.concat(
    [((df["nn-regular"] - df[model]) / df["nn-regular"]).idxmin()
     for model in df.columns.get_level_values(0)[:: 2]]).rename(r"min $\delta$ nn")
for series in [idxmin, idxmax]:
    series.index = df.columns
df = df.append(mean)
df = df.append(std)
df = df.append(idxmax)
df = df.append(idxmin)


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
