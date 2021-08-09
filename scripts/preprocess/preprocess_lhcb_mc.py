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

# the order of these is important
# [0] -> minbias
# [2i] -> magdown
# [2i+1] -> mag up for the same sample
tupleTrees = [
    "2018MinBias_MVATuple.root",
    "MagDown_up08_30000000_MVATuple.root",  # minbias
    "upgrade_magup_sim10_up08_30000000_digi_MVATuple.root",  # minbias
    "MagDown_up05_11104054_MVATuple.root",
    "MagDown_up05_23103042_MVATuple.root",
    #"MagDown_up08_16103330_MVATuple.root",
    "MagDown_up05_11104055_MVATuple.root",
    "MagDown_up05_23103062_MVATuple.root",
    "MagDown_up08_16103332_MVATuple.root",
    "MagDown_up05_11104056_MVATuple.root",
    "MagDown_up05_23163003_MVATuple.root",
    #"MagDown_up08_21101402_MVATuple.root",
    "MagDown_up05_11104057_MVATuple.root",
    "MagDown_up05_23163052_MVATuple.root",
    "MagDown_up08_21101411_MVATuple.root",
    "MagDown_up05_11104058_MVATuple.root",
    "MagDown_up08_11102521_MVATuple.root",
    "MagDown_up08_21103100_MVATuple.root",
    "MagDown_up05_11164063_MVATuple.root",
    "MagDown_up08_11264001_MVATuple.root",
    #"MagDown_up08_21103110_MVATuple.root",
    "MagDown_up05_11166107_MVATuple.root",
    "MagDown_up08_11264011_MVATuple.root",
    "MagDown_up08_21113000_MVATuple.root",
    "MagDown_up05_11196000_MVATuple.root",
    "MagDown_up08_11874004_MVATuple.root",
    #"MagDown_up08_21113016_MVATuple.root",
    "MagDown_up05_11196011_MVATuple.root",
    #"MagDown_up08_12101401_MVATuple.root",
    #"MagDown_up08_21123203_MVATuple.root",
    "MagDown_up05_11196099_MVATuple.root",
    #"MagDown_up08_12103110_MVATuple.root",
    #"MagDown_up08_21123240_MVATuple.root",
    #"MagDown_up05_11264001_MVATuple.root",
    "MagDown_up08_12103406_MVATuple.root",
    "MagDown_up08_23103100_MVATuple.root",
    "MagDown_up05_12103009_MVATuple.root",
    "MagDown_up08_12103422_MVATuple.root",
    "MagDown_up08_23103110_MVATuple.root",
    "MagDown_up05_12103019_MVATuple.root",
    "MagDown_up08_12103423_MVATuple.root",
    #"MagDown_up08_25103102_MVATuple.root",
    "MagDown_up05_12103028_MVATuple.root",
    "MagDown_up08_12103443_MVATuple.root",
    "MagDown_up08_25113000_MVATuple.root",
    "MagDown_up05_12103038_MVATuple.root",
    #"MagDown_up08_12103444_MVATuple.root",
    #"MagDown_up08_25123000_MVATuple.root",
    "MagDown_up05_12103041_MVATuple.root",
    "MagDown_up08_12103445_MVATuple.root",
    "MagDown_up08_26104186_MVATuple.root",
    "MagDown_up05_12103051_MVATuple.root",
    #"MagDown_up08_12163001_MVATuple.root",
    "MagDown_up08_26104187_MVATuple.root",
    "MagDown_up05_15104142_MVATuple.root",
    #"MagDown_up08_12163021_MVATuple.root",
    "MagDown_up08_26106182_MVATuple.root",
    "MagDown_up05_15364010_MVATuple.root",
    #"MagDown_up08_12165106_MVATuple.root",
    "MagDown_up08_27163003_MVATuple.root",
    "MagDown_up05_21163002_MVATuple.root",
    #"MagDown_up08_13264021_MVATuple.root",
    #"MagDown_up08_27163206_MVATuple.root",
    "MagDown_up05_21163012_MVATuple.root",
    #"MagDown_up08_13264031_MVATuple.root",
    #"MagDown_up08_27163207_MVATuple.root",
    "MagDown_up05_21163022_MVATuple.root",
    #"MagDown_up08_15102320_MVATuple.root",
    "MagDown_up08_27173002_MVATuple.root",
    "MagDown_up05_21163032_MVATuple.root",
    "MagDown_up08_15364010_MVATuple.root",
    "MagDown_up08_27225003_MVATuple.root",
    "MagDown_up05_21163042_MVATuple.root",
    "MagDown_up08_16103130_MVATuple.root",
    "MagDown_up08_27375075_MVATuple.root",
    "MagDown_up05_23103012_MVATuple.root",
    #"MagDown_up08_16103131_MVATuple.root",
    # "upgrade_magdown_sim10_up08_11102202_digi_MVATuple.root",  # Bd -> (Kst -> K pi) gamma
    # "upgrade_magup_sim10_up08_11102202_digi_MVATuple.root",  # Bd -> (Kst -> K pi) gamma
    # "upgrade_magdown_sim10_up08_11124001_digi_MVATuple.root",  # Bd -> (Kst -> K pi) ee
    # "upgrade_magup_sim10_up08_11124001_digi_MVATuple.root",  # Bd -> (Kst -> K pi) ee
    # "upgrade_magdown_sim10_up08_21103100_digi_MVATuple.root",  # D+ -> Ks pi+
    # "upgrade_magup_sim10_up08_21103100_digi_MVATuple.root",  # D+ -> Ks pi+
    # # B0 -> (D* -> (D -> K pi) pi) mu nu
    # "upgrade_magdown_sim10_up08_11874004_digi_MVATuple.root",
    # "upgrade_magup_sim10_up08_11874004_digi_MVATuple.root",
    # "upgrade_magdown_sim10_up08_27163003_digi_MVATuple.root",  # D* -> (D -> K pi) pi
    # "upgrade_magup_sim10_up08_27163003_digi_MVATuple.root",  # D* -> (D -> K pi) pi
    # "upgrade_magdown_sim10_up08_13104012_digi_MVATuple.root",  # Bs -> phi phi
    # "upgrade_magup_sim10_up08_13104012_digi_MVATuple.root",  # Bs -> phi phi
]


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

    selt = df[sel].copy()
    # for signal samples, the denominator for the efficiency
    # is defined with respect to the truth cut
    n_events_before = (
        evts_passing_truth_cut.to_frame(index=False).groupby("eventtype").nunique()[1:]
    )
    # for the rate, we just look how many events we ran over.
    n_events_before.loc[0] = evttuple[evttuple.eventtype == 0].EventInSequence.max()

    n_events_after = {
        et: selt[selt.eventtype == et].EventInSequence.nunique()
        for et in n_events_before.index
    }
    # eff = after / before
    effs = {
        et: n_events_after[et] / int(n_events_before.loc[et]) for et in n_events_after
    }
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


unprocessed = [from_root(x, columns) for x in tupleTrees]
evttuples = [from_root(x, evttuple=True) for x in tupleTrees]


last_evt_mb2018 = evttuples[0].EventInSequence.max()
evttuples[1]["EventInSequence"] += last_evt_mb2018 + 1
unprocessed[1]["EventInSequence"] += last_evt_mb2018 + 1
last_evt_mbnew = evttuples[1].EventInSequence.max()
evttuples[2]["EventInSequence"] += last_evt_mbnew + 1
unprocessed[2]["EventInSequence"] += last_evt_mbnew + 1

merged_unprocessed = [pd.concat([unprocessed[0], unprocessed[1], unprocessed[2]])] + unprocessed[3:]
merged_evttuples = [pd.concat([evttuples[0], evttuples[1], evttuples[2]])] + evttuples[3:]


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
