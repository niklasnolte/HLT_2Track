import pickle
import typing as t
from itertools import product
from os import makedirs
from os.path import join

import fire
import numpy as np
from hlt2trk.models import load_model
from hlt2trk.utils import meta_info as meta
from hlt2trk.utils.config import get_config
from hlt2trk.utils.data import get_data
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.naive_bayes import GaussianNB


def _get_grid(grid_size: t.Iterable = 20):
    X = np.meshgrid(*[np.linspace(0, 1, size) for size in grid_size])
    X = np.stack([feature.flatten() for feature in X], axis=1)
    return X


def _index_drop(dims, index_keep):
    mask = np.ones_like(dims).astype(bool)
    mask[index_keep] = False
    return tuple(np.arange(len(dims))[mask])


def _flattenXy(X: np.ndarray, y: np.ndarray, dims: t.Iterable = None,
               index_keep: t.Iterable = [0, 1]):
    ndims = X.shape[1]
    if ndims > 2:
        drop = _index_drop(dims, index_keep)
        X = X.reshape(*dims, len(dims))
        X = X.mean(axis=drop)[:, :, index_keep]
        X = X.reshape(-1, 2)
        y = y.reshape(*dims)
        y = y.mean(axis=drop).reshape(-1)
    return X, y


if __name__ == '__main__':
    cfg = get_config()
    X, y = get_data(cfg)
    nfeats = len(cfg.features)

    model = load_model(cfg)

    limits = [np.quantile(X[:, i], (0.02, 0.98)) for i in range(nfeats)]
    linspaces = [np.linspace(*xi, 100 if nfeats == 2 else 20) for xi in limits]
    grid = np.array(tuple(product(*linspaces)))

    gs = (100, 100, 10, 10) if cfg.experiment == 2 else (100, 100)
    X = _get_grid(grid_size=gs)
    for i, model_name in enumerate(meta.model_names):
        fname = f"{model_name}_{cfg.experiment}.pkl"
        with open(join(meta.locations.model_dir, fname), 'rb') as f:
            model = pickle.load(f)
            y = model.predict_proba(X)[:, 1]
            # plotting
            index_keep = [0, 1]
            X, y = _flattenXy(X, y, dims=gs,
                              index_keep=index_keep)

            file_dir = join(meta.locations.plot_dir, "data")
            makedirs(file_dir, exist_ok=True)
            file = join(file_dir, f"{cfg.model}-{cfg.experiment}")
            np.savez_compressed(file, X, y)
