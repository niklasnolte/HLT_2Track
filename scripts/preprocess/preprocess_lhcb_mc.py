from os.path import join
import json
import numpy as np
import pandas as pd
import uproot3 as u
from collections.abc import Iterable
from hlt2trk.utils.config import get_config, Locations, format_location, dirs, evttypes

cfg = get_config()


def from_root(path, columns="*", tuple="event", maxEvt=None) -> pd.DataFrame:
    if not isinstance(path, Iterable):
      path = (path,)

    dfs = []
    for pathi in path:
      ttree = u.open(join(dirs.raw_data, pathi))
      if tuple=="event":
          df = ttree["EventTuple/Evt"].pandas.df(columns, flatten=False)
      elif tuple=="two":
          df = ttree["DecayTreeTuple#1/N2Trk"].pandas.df(columns)
      elif tuple=="one":
          df = ttree["DecayTreeTuple/N1Trk"].pandas.df(columns)
      if maxEvt is not None:
          df = df[df.EventInSequence < maxEvt//len(path)].copy()
      dfs.append(df)

    for i in range(1, len(dfs)):
      last_evt = dfs[i-1].EventInSequence.max()
      dfs[i]["EventInSequence"] += last_evt

    df = pd.concat(dfs)

    return df


def presel(df: pd.DataFrame, evttuple: pd.DataFrame, kind: str) -> pd.DataFrame:
    evt_grp = ["EventInSequence", "eventtype"]
    # only take the events that have a B/C candidate with more
    # than 2 GeV PT and more than .2 ps flight distance
    signals = evttuple[evttuple.eventtype != 0]
    signals.set_index(evt_grp, inplace=True)
    signals = signals[signals.n_signals > 0] #TODO look up n_signals
    hasheavyflavor = signals.signal_type.apply(max) > 0
    TRUEPT_cut = signals.signal_TRUEPT.apply(max) > 2000  # 2 GeV
    TRUETAU_cut = signals.signal_TRUETAU.apply(max) > 2e-4  # .2 ps
    evts_passing_truth_cut = signals[
        (hasheavyflavor & TRUEPT_cut & TRUETAU_cut)
    ].index

    df = df[df.set_index(evt_grp).index.isin(evts_passing_truth_cut) | (df.eventtype == 0)] # no truth cut on minbias

    if kind == "two":
      # two track has a preselection
      sel = df.sv_PT > cfg.presel_conf["svPT"]
      sel &= df.trk1_PT > cfg.presel_conf["trkPT"]
      sel &= df.trk2_PT > cfg.presel_conf["trkPT"]
      sel &= df.sv_ENDVERTEX_CHI2 < cfg.presel_conf["svchi2"]
      sel &= df.sv_MCORR > cfg.presel_conf["mcorr"]
      df = df[sel]
    else:
      sel = df.trk_PT > cfg.presel_conf["trkPT"]
      df = df[sel]
    
    # for signal samples, the denominator for the efficiency
    # is defined with respect to the truth cut
    n_events_before = (
        evts_passing_truth_cut.to_frame(index=False).groupby("eventtype").nunique()
    )
    # for the rate, we just look how many events we ran over.
    n_events_before.loc[0] = (evttuple.eventtype == 0).sum()

    n_events_after = df.groupby("eventtype").EventInSequence.nunique()

    effs = n_events_after / n_events_before["EventInSequence"]

    if kind == "two":
      loc = Locations.presel_efficiencies
    elif kind == "one":
      loc = Locations.presel_efficiencies_onetrack

    with open(format_location(loc, cfg), "w") as f:
        json.dump(effs.to_dict(), f)

    return df.copy()
    


def preprocess(df: pd.DataFrame, evttuple: pd.DataFrame, kind: str) -> pd.DataFrame:
    if kind=="two":
      df["sumpt"] = df[["trk1_PT", "trk2_PT"]].sum(axis=1)
      df["minipchi2"] = df[["trk1_IPCHI2_OWNPV", "trk2_IPCHI2_OWNPV"]].min(axis=1)
      df["signal_type"] = df["trk1_signal_type"] * df["trk2_signal_type"]
      df = presel(df, evttuple, kind)
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
    elif kind=="one":
      df = presel(df, evttuple, kind)
      df.dropna(inplace=True)
    else:
      raise ValueError("unknown kind, choose between \"one\" and \"two\"")

    return df


columns_two = [
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
    "trk1_fromHFPV",
    "trk1_signal_TRUEENDVERTEX_X",
    "trk1_signal_TRUEENDVERTEX_Y",
    "trk1_signal_TRUEENDVERTEX_Z",
    "trk1_signal_TRUEORIGINVERTEX_X",
    "trk1_signal_TRUEORIGINVERTEX_Y",
    "trk1_signal_TRUEORIGINVERTEX_Z",
    "trk1_signal_TRUETAU",
    "trk2_IPCHI2_OWNPV",
    "trk2_PT",
    "trk2_P",
    "trk2_signal_type",
    "trk2_fromHFPV",
    "trk2_signal_TRUEENDVERTEX_X",
    "trk2_signal_TRUEENDVERTEX_Y",
    "trk2_signal_TRUEENDVERTEX_Z",
    "trk2_signal_TRUEORIGINVERTEX_X",
    "trk2_signal_TRUEORIGINVERTEX_Y",
    "trk2_signal_TRUEORIGINVERTEX_Z",
    "trk2_signal_TRUETAU",
    "EventInSequence",
]

columns_one = [
    "trk_IPCHI2_OWNPV",
    "trk_PT",
    "trk_OWNPV_Z",
    "trk_signal_type",
    "trk_fromHFPV",
    "trk_signal_TRUEENDVERTEX_X",
    "trk_signal_TRUEENDVERTEX_Y",
    "trk_signal_TRUEENDVERTEX_Z",
    "trk_signal_TRUEORIGINVERTEX_X",
    "trk_signal_TRUEORIGINVERTEX_Y",
    "trk_signal_TRUEORIGINVERTEX_Z",
    "trk_signal_TRUETAU",
    "EventInSequence",
]

mb_tupleTrees = [
    "MagDown_aU1_30000000_MVATuple_IsLepton.root",
    "MagUp_aU1_30000000_MVATuple_IsLepton.root",
    "../2018MinBias_MVATuple_IsLepton.root"    
    #"MagDown_30000000_MVATuple_IsLepton.root",  # minbias
    #"upgrade_magup_sim10_up08_30000000_digi_MVATuple.root",  # minbias
    # "upgrade_magdown_sim10_up08_30000000_digi_MVATuple_IsLepton.root",
    # "upgrade_magup_sim10_up08_30000000_digi_MVATuple_IsLepton.root"
]

sig_tupleTrees = [(f"MagDown_aU1_{evttype}_MVATuple_IsLepton.root",) for evttype in evttypes.values()]

# sig_tupleTrees = [f"upgrade_magupdown_sim10_up08_{evttype}_digi_MVATuple_IsLepton.root" for evttype in evttypes.values()]


n_sig_per_sample = 2000

unprocessed_two = [from_root(mb_tupleTrees, columns_two, tuple="two")]
unprocessed_two += [from_root(x, columns_two, tuple="two", maxEvt=n_sig_per_sample) for x in sig_tupleTrees]

unprocessed_one = [from_root(mb_tupleTrees, columns_one, tuple="one")]
unprocessed_one += [from_root(x, columns_one, tuple="one", maxEvt=n_sig_per_sample) for x in sig_tupleTrees]

evttuples = [from_root(mb_tupleTrees, tuple="event")]
evttuples += [from_root(x, tuple="event", maxEvt=n_sig_per_sample) for x in sig_tupleTrees]

n_mb_tuples = len(mb_tupleTrees)


for i, df in enumerate(unprocessed_one):
    df["eventtype"] = i
for i, df in enumerate(unprocessed_two):
    df["eventtype"] = i
for i, t in enumerate(evttuples):
    t["eventtype"] = i

df_two = pd.concat(unprocessed_two)
df_one = pd.concat(unprocessed_one)
evttuple = pd.concat(evttuples)

df_two = preprocess(df_two, evttuple, "two")
save_file_two = format_location(Locations.data_two, cfg)
df_two.to_pickle(save_file_two)
print("Preprocessed data saved to:")
print(save_file_two)

df_one = preprocess(df_one, evttuple, "one")
save_file_one = format_location(Locations.data_one, cfg)
df_one.to_pickle(save_file_one)
print("Preprocessed data saved to:")
print(save_file_one)
