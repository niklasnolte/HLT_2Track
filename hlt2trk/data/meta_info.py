from os.path import dirname, join, abspath
from sys import argv

import numpy as np
import pandas as pd
from .utils import to_np

__all__ = [
    'lhcb_sim',
    'two_dim',
    'sigma_net',
    'features',
    'path_suffix',
    'locations',
    'get_data',
    'get_data_for_training',
    'load_model'
]

# lhcb_sim = "lhcb" in argv
lhcb_sim = "simple" not in argv
two_dim = "2d" in argv
sigma_net = "sigma" in argv

if sigma_net:
    path_suffix = "sigma"
else:
    path_suffix = "regular"

# TODO remove and use experiments instead
if two_dim:
    features = ["minipchi2", "sumpt"]
else:
    features = ["fdchi2", "sumpt", "vchi2", "minipchi2"]

path_suffix += "_" + "_".join(features)

if lhcb_sim:
    path_suffix += "_lhcb"

model_names = [
    "LinearDiscriminantAnalysis",
    "QuadraticDiscriminantAnalysis",
    "GaussianNB"]


class locations:
    project_root = abspath(dirname(__file__) + "/../..")
    grid_X = f"{project_root}/savepoints/gridX_{path_suffix}.pkl"
    grid_Y = f"{project_root}/savepoints/gridY_{path_suffix}.pkl"
    sig_pkl = f"{project_root}/data/beauty.pkl"
    bkg_pkl = f"{project_root}/data/bkgd.pkl"
    sim_pkl = f"{project_root}/data/MC.pkl"
    model_dir = f"{project_root}/models/"
    model = join(model_dir, f"{path_suffix}.torch")
    # TODO fix this to import from config
    # dont like this hack
    sigma_model = join(
        model, f"{path_suffix.replace('regular', 'sigma')}.torch")
    regular_model = join(
        model, f"{path_suffix.replace('sigma', 'regular')}.torch")


class experiments:
    features = [
        ["fdchi2", "sumpt"],
        ["minipchi2", "vchi2"],
        ["fdchi2", "sumpt", "minipchi2", "vchi2"]]


def get_data():
    if not lhcb_sim:
        sig: pd.DataFrame = pd.read_pickle(locations.sig_pkl)
        bkg: pd.DataFrame = pd.read_pickle(locations.bkg_pkl)
    else:
        mc: pd.DataFrame = pd.read_pickle(locations.sim_pkl)
        bkg = mc[mc.label == 0].reset_index(drop=True)
        sig = mc[mc.label != 0].reset_index(drop=True)
    return sig, bkg


def get_data_for_training(features=features, normalize=False):
    sig, bkg = get_data()
    # some preprocessing
    if lhcb_sim:
        bkg = bkg[bkg.eventtype == 0]  # only take minbias as bkg for now
        sig = sig[sig.eventtype != 0]  # why is there signal in minbias?

    X: np.ndarray = np.concatenate([to_np(sig, features),
                                    to_np(bkg, features)])

    if normalize:
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


def load_model(model=None):
    import torch
    if model is None:
        from hlt2trk.models.models import default_model as m

        m.load_state_dict(torch.load(locations.model))
    elif model == "sigma":
        from hlt2trk.models.models import sigma_network as m

        m.load_state_dict(torch.load(locations.sigma_model))
    elif model == "regular":
        from hlt2trk.models.models import regular_model as m

        m.load_state_dict(torch.load(locations.regular_model))
    else:
        raise ValueError(
            "model should either be 'regular' or 'sigma' or None\
                to get the default model")
    return m
