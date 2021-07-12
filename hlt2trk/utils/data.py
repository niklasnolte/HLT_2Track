import numpy as np
import pandas as pd
from .utils import to_np
from . import config
from .config import Locations
from typing import Tuple

__all__ = [
    "get_dataframe",
    "get_data",
    "get_data_for_training",
]


def get_dataframe(cfg: config.Configuration) -> pd.DataFrame:
    return pd.read_pickle(config.format_location(Locations.data, cfg))


def get_data(
    cfg: config.Configuration, preprocess_for_training: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    mc = get_dataframe(cfg)
    bkg = mc[mc.label == 0].reset_index(drop=True)
    sig = mc[mc.label != 0].reset_index(drop=True)
    if preprocess_for_training:
        if cfg.data_type == "lhcb":
            bkg = bkg[bkg.eventtype == 0]  # only take minbias as bkg for now
            sig = sig[sig.eventtype != 0]  # why is there signal in minbias?

    X: np.ndarray = np.concatenate([to_np(sig, cfg.features), to_np(bkg, cfg.features)])

    if cfg.normalize:
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    y: np.ndarray = np.concatenate([np.ones(len(sig)), np.zeros(len(bkg))])
    print(y.mean())
    return X, y


def get_data_for_training(cfg: config.Configuration) -> Tuple[np.ndarray]:
    X, y = get_data(cfg, preprocess_for_training=True)

    np.random.seed(3)
    shuffle_idx: np.ndarray = np.random.permutation(np.arange(len(X)))
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    split = (len(X) * 3) // 4
    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    y_test = y[split:]
    return X_train, y_train, X_test, y_test
