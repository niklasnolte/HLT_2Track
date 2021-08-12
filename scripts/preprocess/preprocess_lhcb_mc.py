from os.path import join
import json
import numpy as np
import pandas as pd
import uproot3 as u
from hlt2trk.utils.config import get_config, Locations, format_location, dirs, evttypes

cfg = get_config()


def from_root(path: str, columns="*", evttuple=False, maxEvt=None) -> pd.DataFrame:
    ttree = u.open(join(dirs.raw_data, path))
    if evttuple:
        df = ttree["EventTuple/Evt"].pandas.df(columns)
    else:
        df = ttree["DecayTreeTuple#1/N2Trk"].pandas.df(columns)
    if maxEvt is None:
        return df
    else:
        return df[df.EventInSequence < maxEvt].copy()


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
    "trk1_fromHFPV",
    "trk2_IPCHI2_OWNPV",
    "trk2_PT",
    "trk2_P",
    "trk2_signal_type",
    "trk2_fromHFPV",
    "EventInSequence",
]

mb_tupleTrees = [
    "2018MinBias_MVATuple.root",
    "MagDown_30000000_MVATuple.root",  # minbias
    "upgrade_magup_sim10_up08_30000000_digi_MVATuple.root",  # minbias
]

sig_tupleTrees = [f"MagDown_{evttype}_MVATuple.root" for evttype in evttypes.values()]


def presel(df: pd.DataFrame, evttuple: pd.DataFrame) -> pd.DataFrame:
    evt_grp = ["EventInSequence", "eventtype"]
    # only take the events that have a B candidate with more
    # than 2 GeV PT and more than .2 ps flight distance
    grpd_truth = evttuple.groupby(evt_grp)
    is_minbias = grpd_truth.eventtype == 0  # no truth cuts on minbias
    hasbeauty = grpd_truth.signal_type.max() > 0
    TRUEPT_cut = grpd_truth.signal_TRUEPT.max() > 2000  # 2 GeV
    TRUETAU_cut = grpd_truth.signal_TRUETAU.max() > 2e-4  # .2 ps
    evts_passing_truth_cut = hasbeauty[
        (hasbeauty & TRUEPT_cut & TRUETAU_cut) | is_minbias
    ].index

    df = df[df.set_index(evt_grp).index.isin(evts_passing_truth_cut)]

    sel = df.sv_PT > cfg.presel_conf["svPT"]
    sel &= df.trk1_PT > cfg.presel_conf["trkPT"]
    sel &= df.trk2_PT > cfg.presel_conf["trkPT"]
    sel &= df.sv_ENDVERTEX_CHI2 < cfg.presel_conf["svchi2"]
    sel &= df.sv_MCORR > 1000

    selt = df[sel].copy()
    # for signal samples, the denominator for the efficiency
    # is defined with respect to the truth cut
    n_events_before = (
        evts_passing_truth_cut.to_frame(index=False).groupby("eventtype").nunique()[1:]
    )
    # for the rate, we just look how many events we ran over.
    n_events_before.loc[0] = evttuple[evttuple.eventtype == 0].EventInSequence.max()

    n_events_after = selt.groupby("eventtype").EventInSequence.nunique()

    effs = n_events_after / n_events_before["EventInSequence"]
    with open(format_location(Locations.presel_efficiencies, cfg), "w") as f:
        json.dump(effs.to_dict(), f)

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


unprocessed = [from_root(x, columns) for x in mb_tupleTrees]
unprocessed += [from_root(x, columns, maxEvt=3000) for x in sig_tupleTrees]
evttuples = [from_root(x, evttuple=True) for x in mb_tupleTrees]
evttuples += [from_root(x, evttuple=True, maxEvt=3000) for x in sig_tupleTrees]


last_evt_mb2018 = evttuples[0].EventInSequence.max()
evttuples[1]["EventInSequence"] += last_evt_mb2018 + 1
unprocessed[1]["EventInSequence"] += last_evt_mb2018 + 1
last_evt_mbnew = evttuples[1].EventInSequence.max()
evttuples[2]["EventInSequence"] += last_evt_mbnew + 1
unprocessed[2]["EventInSequence"] += last_evt_mbnew + 1

merged_unprocessed = [
    pd.concat([unprocessed[0], unprocessed[1], unprocessed[2]])
] + unprocessed[3:]
merged_evttuples = [pd.concat([evttuples[0], evttuples[1], evttuples[2]])] + evttuples[
    3:
]


# for i in range(3, len(tupleTrees), 2):
#     last_evt = evttuples[i].EventInSequence.max()
#     evttuples[i + 1]["EventInSequence"] += last_evt + 1
#     unprocessed[i + 1]["EventInSequence"] += last_evt + 1
#     merged_unprocessed.append(pd.concat([unprocessed[i], unprocessed[i + 1]]))
#     merged_evttuples.append(pd.concat([evttuples[i], evttuples[i + 1]]))


for i, df in enumerate(merged_unprocessed):
    df["eventtype"] = i
for i, t in enumerate(merged_evttuples):
    t["eventtype"] = i

df = pd.concat(merged_unprocessed)
evttuple = pd.concat(merged_evttuples)
df = preprocess(df, evttuple)
save_file = format_location(Locations.data, cfg)
df.to_pickle(save_file)
print("Preprocessed data saved to:")
print(save_file)
