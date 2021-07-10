import numpy as np
import pandas as pd
from .utils import to_np
from . import config
from .config import Locations

__all__ = [
    'lhcb_sim',
    'two_dim',
    'sigma_net',
    'features',
    'path_suffix',
    'locations',
    'get_data',
    'get_data_for_training',
    'load_model',
]



def get_data(cfg):
    mc: pd.DataFrame = pd.read_pickle(config.format_location(Locations.data, cfg))
    bkg = mc[mc.label == 0].reset_index(drop=True)
    sig = mc[mc.label != 0].reset_index(drop=True)
    return sig, bkg


def get_data_for_training(cfg):
    sig, bkg = get_data(cfg)
    # some preprocessing
    if cfg.data_type == "lhcb":
        bkg = bkg[bkg.eventtype == 0]  # only take minbias as bkg for now
        sig = sig[sig.eventtype != 0]  # why is there signal in minbias?

    X: np.ndarray = np.concatenate([to_np(sig, cfg.features),
                                    to_np(bkg, cfg.features)])

    if cfg.normalize:
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    Y: np.ndarray = np.concatenate([np.ones(len(sig)), np.zeros(len(bkg))])

    np.random.seed(3)

    shuffle_idx: np.ndarray = np.random.permutation(np.arange(len(X)))
    X = X[shuffle_idx]
    Y = Y[shuffle_idx]

    split = (len(X) * 3) // 4
    X_train = X[:split]
    X_test = X[split:]
    Y_train = Y[:split]
    Y_test = Y[split:]
    return X_train, Y_train, X_test, Y_test
