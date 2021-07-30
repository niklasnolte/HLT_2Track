from os.path import join
import json
import numpy as np
import pandas as pd
import uproot3 as u
from hlt2trk.utils.config import get_config, Locations, format_location, dirs

cfg = get_config()


def from_root(path: str, columns="*", evttuple=False) -> pd.DataFrame:
    ttree = u.open(join(dirs.raw_data, path))
    if evttuple:
      return ttree["EventTuple/Evt"].pandas.df(columns)
    else:
      return ttree["DecayTreeTuple#1/N2Trk"].pandas.df(columns)


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
    "2018MinBias_MVATuple.root",
    "upgrade_magdown_sim10_up08_11102202_digi_MVATuple.root",  # Bd -> (Kst -> K pi) gamma
    "upgrade_magdown_sim10_up08_11124001_digi_MVATuple.root",  # Bd -> (Kst -> K pi) ee
    "upgrade_magdown_sim10_up08_21103100_digi_MVATuple.root",  # D+ -> Ks pi+
    # B0 -> (D* -> (D -> K pi) pi) mu nu
    "upgrade_magdown_sim10_up08_11874004_digi_MVATuple.root",
    "upgrade_magdown_sim10_up08_27163003_digi_MVATuple.root",  # D* -> (D -> K pi) pi
    "upgrade_magdown_sim10_up08_13104012_digi_MVATuple.root",  # Bs -> phi phi
]


dfs = [from_root(x, columns) for x in tupleTrees]
evttuples = [from_root(x, evttuple=True) for x in tupleTrees]

def presel(df: pd.DataFrame, evttuple: pd.DataFrame) -> pd.DataFrame:
    evt_grp = ["EventInSequence", "eventtype"]
    # only take the events that have a B candidate with more
    # than 2 GeV PT and more than .2 ps flight distance
    grpd_truth = evttuple.groupby(evt_grp)
    hasbeauty = grpd_truth.signal_type.max() == 2
    TRUEPT_cut = grpd_truth.signal_TRUEPT.max() > 2000
    #TRUETAU_cut = grpd_truth.signal_TRUETAU.max() > 2e-4
    evts_passing_truth_cut = hasbeauty[hasbeauty & TRUEPT_cut].index

    df = df[df.set_index(evt_grp).index.isin(evts_passing_truth_cut)]

    sel = df.sv_PT > cfg.presel_conf["svPT"]
    sel &= df.trk1_PT > cfg.presel_conf["trkPT"]
    sel &= df.trk2_PT > cfg.presel_conf["trkPT"]

    selt = df[sel].copy()
    # calculate efficiency
    # EventInSequence.max() should be close to the total number of events run over
    n_events_before = evts_passing_truth_cut.to_frame(index=False).groupby("eventtype").nunique()
    n_events_after = {
        et: selt[selt.eventtype == et].EventInSequence.nunique()
        for et in selt.eventtype.unique()
    }

    effs = {int(et): n_events_after[et] / int(n_events_before.loc[int(et)]) for et in n_events_after}
    with open(format_location(Locations.presel_efficiencies, cfg), "w") as f:
        json.dump(effs, f)

    return selt


def preprocess(df: pd.DataFrame, evttuple: pd.DataFrame) -> pd.DataFrame:
    df["sumpt"] = df[["trk1_PT", "trk2_PT"]].sum(axis=1)
    df["minipchi2"] = df[["trk1_IPCHI2_OWNPV", "trk2_IPCHI2_OWNPV"]].min(axis=1)
    df["signal_type"] = df[["trk1_signal_type", "trk2_signal_type"]].min(axis=1)
    df = presel(df, evttuple)
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
for i, t in enumerate(evttuples):
    t["eventtype"] = i

df = pd.concat(dfs)
evttuple = pd.concat(evttuples)
df = preprocess(df, evttuple)
save_file = format_location(Locations.data, cfg)
df.to_pickle(save_file)
print("Preprocessed data saved to:")
print(save_file)
