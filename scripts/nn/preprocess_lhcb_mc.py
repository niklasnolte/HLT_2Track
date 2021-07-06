import pandas as pd
import uproot3 as u
import numpy as np
from hlt2trk.data import meta_info as meta
from os.path import join


prefix = "~/ntuples/"


def from_root(path: str, columns="*") -> pd.DataFrame:
    return u.open(prefix + path)["DecayTreeTuple#1/N2Trk"].pandas.df(columns)


columns = [
    "sv_MCORR",
    "sv_IPCHI2_OWNPV",
    "sv_FDCHI2_OWNPV",
    "sv_DIRA_OWNPV",
    "sv_PT",
    "sv_P",
    "sv_ENDVERTEX_CHI2",
    "trk1_IPCHI2_OWNPV",
    "trk1_PT",
    "trk1_P",
    "trk1_signal_type",
    "trk2_IPCHI2_OWNPV",
    "trk2_PT",
    "trk2_P",
    "trk2_signal_type",
    "EventInSequence",
]

tupleTrees = [
    # "upgrade_magdown_sim10_up08_30000000_digi_MVATuple.root",
    "2018MinBias_MVATuple.root",
    "upgrade_magdown_sim10_up08_11102202_digi_MVATuple.root",
    "upgrade_magdown_sim10_up08_11124001_digi_MVATuple.root",
    "upgrade_magdown_sim10_up08_21103100_digi_MVATuple.root",
    "upgrade_magdown_sim10_up08_11874004_digi_MVATuple.root",
    "upgrade_magdown_sim10_up08_27163003_digi_MVATuple.root",
    "upgrade_magdown_sim10_up08_13104012_digi_MVATuple.root",
]


dfs = [from_root(x, columns) for x in tupleTrees]


def presel(df: pd.DataFrame) -> pd.DataFrame:
    sel = df.sv_PT > 2000
    sel &= df.trk1_PT > 500
    sel &= df.trk2_PT > 500
    df = df[sel].copy()

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df["sumpt"] = df[["trk1_PT", "trk2_PT"]].sum(axis=1)
    df["minipchi2"] = df[["trk1_IPCHI2_OWNPV", "trk2_IPCHI2_OWNPV"]].min(axis=1)
    df["label"] = (df[["trk1_signal_type", "trk2_signal_type"]].min(axis=1) > 0).astype(
        int
    )
    df = presel(df)
    df.rename(
        columns={"sv_FDCHI2_OWNPV": "fdchi2", "sv_ENDVERTEX_CHI2": "vchi2"},
        inplace=True,
    )
    to_log = [
        "sv_IPCHI2_OWNPV",
        "fdchi2",
        "trk1_IPCHI2_OWNPV",
        "trk2_IPCHI2_OWNPV",
        "minipchi2",
    ]
    to_scale = [
        "sv_PT",
        "sv_P",
        "sumpt",
        "trk1_PT",
        "trk1_P",
        "trk2_PT",
        "trk2_P",
    ]
    df.dropna(inplace=True)
    lower_bound = 1e-10
    df["vchi2"] = df["vchi2"].clip(lower_bound)
    df[to_log] = df[to_log].clip(lower_bound)
    df[to_log] = df[to_log].apply(np.log)
    df[to_scale] = df[to_scale].clip(lower_bound, 1e5)
    df[to_scale] = df[to_scale] / 1000  # to GeV
    return df


for i, df in enumerate(dfs):
    df["eventtype"] = i

df = pd.concat(dfs)
df = preprocess(df)
df.to_pickle(join(meta.locations.project_root, "data/MC.pkl"))
df.to_hdf(join(meta.locations.project_root, "data/MC.h5"), "MC", mode="w")
