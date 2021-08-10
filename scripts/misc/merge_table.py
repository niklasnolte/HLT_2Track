
from os.path import join
import os
from hlt2trk.utils.config import dirs
import pandas as pd
import numpy as np
import pickle

strings = ["fdchi2+sumpt+vchi2+minipchi2",
           "ipcuttrain:6",
           #    "ipcuttrain:10",
           "unnormed",
           "svchi2:20",
           ]


def conditions(f):
    return all([string in f for string in strings])


files = os.listdir(dirs.results_eff)
subset = [f for f in files if f.startswith("target-eff") and conditions(f)]

order = ["bdt", "nn-regular", "nn-inf", "nn-one", ]  # "qda", "lda", "gnb", ]

models = {}
if len(subset) > 0:
    for fname in subset:
        file = join(dirs.results_eff, fname)
        with open(file, "rb") as f:
            models[fname.split("_")[1]] = pickle.load(f)

dfs = []
for df, model in [(models[name], name)for name in order if name in models.keys()]:
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


with open(join(dirs.results_eff, f"table_{'_'.join(strings)}.txt"), "w") as f:
    table = make_table(df)
    print(table)
    f.writelines(table)
