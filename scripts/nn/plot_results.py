import typing as t
from itertools import combinations
from os.path import join

import numpy as np
import torch
from matplotlib import pyplot as plt

plt.style.use("seaborn")

import hlt2trk.utils.meta_info as meta

X = torch.load(meta.locations.grid_X).numpy()
Y = torch.load(meta.locations.grid_Y).squeeze().numpy()


def plot_feat_vs_output(x: np.ndarray, y: np.ndarray, params: t.Optional[dict] = None):
    params = params or {}
    for i in range(len(meta.features)):
        xi = x[:, i]
        srt = np.argsort(xi)
        xi = xi[srt]
        xiuniq, idxs = np.unique(xi, return_index=True)
        y_by_xi_mean = [yi.mean() for yi in np.split(y[srt], idxs[1:])]
        y_by_xi_min = [yi.min() for yi in np.split(y[srt], idxs[1:])]
        y_by_xi_max = [yi.max() for yi in np.split(y[srt], idxs[1:])]
        plt.scatter(xiuniq, y_by_xi_mean, **params, label="mean")
        plt.scatter(xiuniq, y_by_xi_min, **params, label="min")
        plt.scatter(xiuniq, y_by_xi_max, **params, label="max")
        plt.legend()
        plt.xlabel(meta.features[i])
        plt.ylabel("output")
        plt.title("sigma" if meta.sigma_net else "regular")
        plt.savefig(join(meta.locations.project_root, f"plots/{meta.features[i]}_vs_output_{meta.path_suffix}.pdf"))
        plt.show()


def plot_2d_vs_output(
        x: np.ndarray,
        y: np.ndarray,
        params: t.Optional[dict] = None,
):
    assert y.ndim == 1, "y is expected to be a 1-d array"
    x0 = x[:, 0]
    x1 = x[:, 1]
    fig = plt.figure()
    ax: plt.Axes3D = fig.add_subplot(projection="3d")
    ax.scatter(x0, x1, y, **(params or {}))
    ax.set_xlabel(meta.features[0])
    ax.set_ylabel(meta.features[1])
    ax.set_zlabel("output")
    ax.set_title("sigma" if meta.sigma_net else "regular")
    plt.savefig(join(meta.locations.project_root,
                      f"plots/{meta.features[0]}_and_{meta.features[1]}_vs_output_{meta.path_suffix}.pdf"))
    plt.show()

def plot_2d_vs_output_heatmap(
        x: np.ndarray,
        y: np.ndarray,
        params: t.Optional[dict] = None,
):
    assert y.ndim == 1, "y is expected to be a 1-d array"
    x0 = x[:, 0]
    x1 = x[:, 1]
    fig = plt.figure()
    ax: plt.Axes = fig.add_subplot()
    sc = ax.scatter(x0, x1, c=y, cmap=plt.cm.RdBu, s=150, marker="s", **(params or {}))
    ax.set_xlabel(meta.features[0])
    ax.set_ylabel(meta.features[1])
    ax.set_title("sigma" if meta.sigma_net else "regular")
    plt.colorbar(sc)
    plt.savefig(join(meta.locations.project_root,
                      f"plots/{meta.features[0]}_and_{meta.features[1]}_vs_output_{meta.path_suffix}_heatmap.pdf"))
    plt.show()


def plot_prediction_line(x: np.ndarray, y: np.ndarray, params: t.Optional[dict] = None):
    nfeatures: int = x.shape[1]
    combs: t.List[t.Tuple] = combinations(range(nfeatures), 2)
    xthresh: np.ndarray = x[(y >= 0.5) & (y <= 0.501)]
    params = params or {}
    for c in combs:
        x0 = xthresh[:, c[0]]
        x1 = xthresh[:, c[1]]
        srt = np.argsort(x0)
        x0 = x0[srt]
        x1 = x1[srt]
        x0uniq, idxs = np.unique(x0, return_index=True)
        x1_by_x0: t.List[np.ndarray] = np.split(x1, idxs[1:])
        minx1 = np.array([min(x1i) for x1i in x1_by_x0])
        maxx1 = np.array([max(x1i) for x1i in x1_by_x0])
        meanx1 = np.array([x1i.mean() for x1i in x1_by_x0])
        medianx1 = np.array([np.median(x1i) for x1i in x1_by_x0])
        plt.plot(x0uniq, minx1, label="min", **params)
        plt.plot(x0uniq, maxx1, label="max", **params)
        plt.plot(x0uniq, meanx1, label="mean", **params)
        plt.plot(x0uniq, medianx1, label="median", **params)
        plt.legend()
        plt.xlabel(meta.features[c[0]])
        plt.ylabel(meta.features[c[1]])
        plt.show()


if meta.two_dim:
    plot_2d_vs_output_heatmap(X, Y)
    #plot_2d_vs_output(X, Y)
else:
    plot_feat_vs_output(X, Y)
