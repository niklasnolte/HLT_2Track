import numpy as np
import pandas as pd
from . import config
from .config import Locations
from typing import Tuple

__all__ = [
    "signal_type_int",
    "get_data",
    "get_data_for_training",
]


def signal_type_int(signal_type: str) -> int:
    # this should match with the signal type definition in the root data
    if signal_type == "beauty":
        return 4 # two tracks with beauty (==2) -> 2*2
    elif signal_type == "charm":
        return 1 # two tracks with charm (==1) -> 1*1
    else:
        raise ValueError(f"signal_type must be one of (\"charm\", \"beauty\")")


def is_signal(cfg, df): # elementwise
    same_ev_mask = df.trk1_signal_TRUEENDVERTEX_Z == df.trk2_signal_TRUEENDVERTEX_Z
    if cfg.signal_type == "beauty":
        st_mask = df.signal_type == signal_type_int("beauty")
    elif cfg.signal_type == "charm":
        st_mask = df.signal_type == signal_type_int("charm")
    elif cfg.signal_type == "heavy-flavor":
        st_mask = (df.signal_type == signal_type_int("beauty")) | (df.signal_type == signal_type_int("charm"))
    return same_ev_mask & st_mask


def get_data(cfg: config.Configuration) -> pd.DataFrame:
    mc = pd.read_pickle(config.format_location(Locations.data, cfg))
    if cfg.normalize:
        x = mc[cfg.features]
        mc[cfg.features] = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
    mc["validation"] = mc.EventInSequence % 4 == 0
    return mc.reset_index(drop=True)


def get_data_for_training(cfg: config.Configuration) -> Tuple[np.ndarray]:
    df = get_data(cfg)
    df = df[df["minipchi2"] < cfg.presel_conf["ipcuttrain"]]
    bkg = df[df.signal_type == 0]
    sig = df[is_signal(cfg, df)]

    if cfg.data_type == "lhcb":
        # minbias + svs of which the tracks associated pvs
        # are at least 10mm away from the signal pv
        bkg = bkg[(bkg.eventtype == 0)]  # | (bkg.trk1_fromHFPV + bkg.trk2_fromHFPV == 0)]
        sig = sig[sig.eventtype != 0]
        sig = sig.groupby(sig.eventtype).head(len(bkg) // sig.eventtype.max())

    bkg_train = bkg[~bkg.validation][cfg.features].values
    sig_train = sig[~sig.validation][cfg.features].values
    bkg_valid = bkg[bkg.validation][cfg.features].values
    sig_valid = sig[sig.validation][cfg.features].values

    X_train = np.concatenate((bkg_train, sig_train))
    y_train = np.concatenate((np.zeros(len(bkg_train)), np.ones(len(sig_train))))
    X_valid = np.concatenate((bkg_valid, sig_valid))
    y_valid = np.concatenate((np.zeros(len(bkg_valid)), np.ones(len(sig_valid))))

    return X_train, y_train, X_valid, y_valid
