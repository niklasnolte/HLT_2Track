from hlt2trk.utils.config import Configs, dirs
import pickle
from os.path import join
import numpy as np
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plt
import typing as t
# plt.style.use('~/.dark_paper.mplstyle')


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


def _heatmap(X, y: np.ndarray, ax: plt.Axes = None):
    if ax is None:
        ax = plt.subplots(1, 1)
    sc = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, marker='s')
    return sc


def _axis_labels(ax, labels):
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])


if __name__ == '__main__':
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)

    for exp in range(len(Configs.features)):
        gs = (100, 100, 10, 10) if len(features) else (100, 100)
        X = _get_grid(grid_size=gs)
        model_names = ["LinearDiscriminantAnalysis",
                       "QuadraticDiscriminantAnalysis", "GaussianNB", ]
        for i, model_name in enumerate(model_names):
            fname = f"{model_name}_{exp}.pkl"
            with open(join(dirs.models, fname), 'rb') as f:
                model = pickle.load(f)
                y = model.predict_proba(X)[:, 1]
                # plotting
                ax: plt.Axes = axes[exp, i]
                ax.text(0.1, .9, ''.join(x for x in model_name if x.isupper()))

                index_keep = [0, 1]
                sc = _heatmap(*_flattenXy(X, y, dims=gs,
                              index_keep=index_keep), ax)
        labels = Configs.features[exp][index_keep]
        _axis_labels(axes[exp, 0], labels)
    plt.colorbar(sc)
    plt.tight_layout()
    plt.show()
