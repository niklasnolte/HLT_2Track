import pandas as pd
import numpy as np
import uproot3 as u

# Tha paths are all wrong here. Needs fix.

prefix = "/data/toy-hlt1-data"

def from_root(path : str, tree : str = 'data') -> pd.DataFrame:
    return u.open(path)[tree].pandas.df()

def preprocess(df):
    to_log = ["fdchi2", "minipchi2"]
    df[to_log] = df[to_log].apply(np.log)


sig_beauty : pd.DataFrame = from_root(f"{prefix}/beauty.root")
preprocess(sig_beauty)
sig_beauty.to_pickle("../../data/lhcb/beauty.pkl")

sig_charm : pd.DataFrame = from_root(f"{prefix}/charm.root")
preprocess(sig_charm)
sig_charm.to_pickle("../../data/lhcb/charm.pkl")

bkg : pd.DataFrame = from_root(f"{prefix}/bkgd.root")
preprocess(bkg)
bkg.to_pickle("../../data/lhcb/bkgd.pkl")

