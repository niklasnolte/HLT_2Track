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
        return 2
    elif signal_type == "charm":
        return 1
    else:
        raise ValueError(f"signal_type must be one of (\"charm\", \"beauty\")")


def is_signal(cfg, signal_int) -> bool:
    if cfg.signal_type == "beauty":
        return signal_int == signal_type_int("beauty")
    elif cfg.signal_type == "charm":
        return signal_int == signal_type_int("charm")
    elif cfg.signal_type == "heavy-flavor":
        return signal_int > 0


def get_data(cfg: config.Configuration) -> pd.DataFrame:
    mc = pd.read_pickle(config.format_location(Locations.data, cfg))
    if cfg.normalize:
        x = mc[cfg.features]
        mc[cfg.features] = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
    # test kink
    mc = mc[mc["minipchi2"] < 6]
    return mc.reset_index(drop=True)


def get_data_for_training(cfg: config.Configuration) -> Tuple[np.ndarray]:
    df = get_data(cfg)
    bkg = df[df.signal_type == 0]
    sig = df[is_signal(cfg, df.signal_type)]

    if cfg.data_type == "lhcb":
        bkg = bkg[bkg.eventtype == 0]  # only take minbias as bkg for now
        sig = sig[sig.eventtype != 0]  # why is there signal in minbias?

    X = np.concatenate((bkg[cfg.features].values, sig[cfg.features].values))
    y = np.concatenate((np.zeros(len(bkg)), np.ones(len(sig))))

    np.random.seed(cfg.seed)
    shuffle_idx: np.ndarray = np.random.permutation(np.arange(len(X)))
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    split = (len(X) * 3) // 4
    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    y_test = y[split:]
    return X_train, y_train, X_test, y_test
