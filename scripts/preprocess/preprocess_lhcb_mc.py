from os.path import join
import json
import numpy as np
import pandas as pd
import uproot3 as u
from hlt2trk.utils.config import get_config, Locations, format_location, dirs, evttypes

cfg = get_config()


def from_root(path: str, columns="*", tuple="event", maxEvt=None) -> pd.DataFrame:
    ttree = u.open(join(dirs.raw_data, path))
    if tuple=="event":
        df = ttree["EventTuple/Evt"].pandas.df(columns, flatten=False)
    elif tuple=="two":
        df = ttree["DecayTreeTuple#1/N2Trk"].pandas.df(columns)
    elif tuple=="one":
        df = ttree["DecayTreeTuple/N1Trk"].pandas.df(columns)
    if maxEvt is None:
        return df
    else:
        return df[df.EventInSequence < maxEvt].copy()


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
    #"2018MinBias_MVATuple.root",
    "MagDown_30000000_MVATuple_IsLepton.root",  # minbias
    #"upgrade_magup_sim10_up08_30000000_digi_MVATuple.root",  # minbias
    "upgrade_magdown_sim10_up08_30000000_digi_MVATuple_IsLepton.root",
    "upgrade_magup_sim10_up08_30000000_digi_MVATuple_IsLepton.root"
]

sig_tupleTrees = [f"MagDown_{evttype}_MVATuple_IsLepton.root" for evttype in evttypes.values()]
# sig_tupleTrees = [f"upgrade_magupdown_sim10_up08_{evttype}_digi_MVATuple_IsLepton.root" for evttype in evttypes.values()]


def presel(df: pd.DataFrame, evttuple: pd.DataFrame, kind: str) -> pd.DataFrame:
    evt_grp = ["EventInSequence", "eventtype"]
    # only take the events that have a B candidate with more
    # than 2 GeV PT and more than .2 ps flight distance
    signals = evttuple[evttuple.eventtype != 0]
    signals.set_index(evt_grp, inplace=True)
    signals = signals[signals.n_signals > 0]
    hasbeauty = signals.signal_type.apply(max) > 0
    TRUEPT_cut = signals.signal_TRUEPT.apply(max) > 2000  # 2 GeV
    TRUETAU_cut = signals.signal_TRUETAU.apply(max) > 2e-4  # .2 ps
    evts_passing_truth_cut = signals[
        (hasbeauty & TRUEPT_cut & TRUETAU_cut)
    ].index

    df = df[df.set_index(evt_grp).index.isin(evts_passing_truth_cut) | (df.eventtype == 0)] # no truth cut on minbias

    if kind == "two":
      # two track has a preselection
      sel = df.sv_PT > cfg.presel_conf["svPT"]
      sel &= df.trk1_PT > cfg.presel_conf["trkPT"]
      sel &= df.trk2_PT > cfg.presel_conf["trkPT"]
      sel &= df.sv_ENDVERTEX_CHI2 < cfg.presel_conf["svchi2"]
      sel &= df.sv_MCORR > 1000
      df = df[sel].copy()
    
    # for signal samples, the denominator for the efficiency
    # is defined with respect to the truth cut
    n_events_before = (
        evts_passing_truth_cut.to_frame(index=False).groupby("eventtype").nunique()
    )
    # for the rate, we just look how many events we ran over.
    n_events_before.loc[0] = evttuple[evttuple.eventtype == 0].EventInSequence.max()

    n_events_after = df.groupby("eventtype").EventInSequence.nunique()

    effs = n_events_after / n_events_before["EventInSequence"]

    if kind == "two":
      loc = Locations.presel_efficiencies
    elif kind == "one":
      loc = Locations.presel_efficiencies_onetrack

    with open(format_location(loc, cfg), "w") as f:
        json.dump(effs.to_dict(), f)

    return df
    


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


unprocessed_two = [from_root(x, columns_two, tuple="two") for x in mb_tupleTrees]
unprocessed_two += [from_root(x, columns_two, tuple="two", maxEvt=5000) for x in sig_tupleTrees]

unprocessed_one = [from_root(x, columns_one, tuple="one") for x in mb_tupleTrees]
unprocessed_one += [from_root(x, columns_one, tuple="one", maxEvt=5000) for x in sig_tupleTrees]

evttuples = [from_root(x, tuple="event") for x in mb_tupleTrees]
evttuples += [from_root(x, tuple="event", maxEvt=5000) for x in sig_tupleTrees]


# CAREFUL: this assumes 3 mb files
# TODO improve this wretched logic
last_evt_mb2018 = evttuples[0].EventInSequence.max()
evttuples[1]["EventInSequence"] += last_evt_mb2018 + 1
unprocessed_two[1]["EventInSequence"] += last_evt_mb2018 + 1
unprocessed_one[1]["EventInSequence"] += last_evt_mb2018 + 1

last_evt_mbnew = evttuples[1].EventInSequence.max()
evttuples[2]["EventInSequence"] += last_evt_mbnew + 1
unprocessed_two[2]["EventInSequence"] += last_evt_mbnew + 1
unprocessed_one[2]["EventInSequence"] += last_evt_mbnew + 1

merged_two = [
    pd.concat([unprocessed_two[0], unprocessed_two[1], unprocessed_two[2]])
] + unprocessed_two[3:]
merged_one = [
    pd.concat([unprocessed_one[0], unprocessed_one[1], unprocessed_one[2]])
] + unprocessed_one[3:]
merged_evttuples = [pd.concat([evttuples[0], evttuples[1], evttuples[2]])] + evttuples[
    3:
]


for i, df in enumerate(merged_two):
    df["eventtype"] = i
for i, df in enumerate(merged_one):
    df["eventtype"] = i
for i, t in enumerate(merged_evttuples):
    t["eventtype"] = i

df_two = pd.concat(merged_two)
df_one = pd.concat(merged_one)
evttuple = pd.concat(merged_evttuples)

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
