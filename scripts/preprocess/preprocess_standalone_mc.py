import pandas as pd
import numpy as np
import uproot3 as u
from hlt2trk.utils.config import get_config, Locations, format_location
from hlt2trk.utils.data import signal_type_int

cfg = get_config()

prefix = format_location(Locations.raw_data_path, cfg)

def from_root(path : str, tree : str = 'data') -> pd.DataFrame:
    return u.open(path)[tree].pandas.df()

def preprocess(df):
    to_log = ["fdchi2", "minipchi2"]
    df[to_log] = df[to_log].apply(np.log)


sig_beauty : pd.DataFrame = from_root(f"{prefix}/beauty.root")
preprocess(sig_beauty)
sig_beauty["signal_type"] = signal_type_int("beauty")

sig_charm : pd.DataFrame = from_root(f"{prefix}/charm.root")
preprocess(sig_charm)
sig_charm["signal_type"] = signal_type_int("charm")

bkg : pd.DataFrame = from_root(f"{prefix}/bkgd.root")
preprocess(bkg)
bkg["signal_type"] = 0

df = pd.concat([sig_beauty, sig_charm, bkg]).reset_index(drop=True)

df.to_pickle(format_location(Locations.data, cfg))
