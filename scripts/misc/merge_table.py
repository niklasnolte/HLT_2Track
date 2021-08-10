
from os.path import join
import os
from hlt2trk.utils.config import dirs
import pandas as pd
import numpy as np
import pickle


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

dfs = []
for df, model in [(models[name], name)for name in order if name in models.keys()]:
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


with open(join(dirs.results_eff, "merged_table.txt"), "w") as f:
    f.writelines(make_table(df))
