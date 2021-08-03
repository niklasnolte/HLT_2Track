from itertools import product
from os.path import join

import numpy as np
import pandas as pd
import seaborn as sns
from hlt2trk.utils import dirs
from hlt2trk.utils.config import Configuration
from hlt2trk.utils.data import get_data, get_data_for_training
from matplotlib import pyplot as plt

cfg = Configuration()
cfg.plot_style = "dark"
if cfg.plot_style in ["dark", "light"]:
    plt.style.use(join(dirs.project_root, f'scripts/plot/paper-{cfg.plot_style}'))


TRAIN_ONLY = True
X_train, y_train, X_test, y_test = get_data_for_training(cfg)

if TRAIN_ONLY:
    X = X_train
    y = y_train
else:
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])

mask = (X[:, 1] < 10) & (X[:, 0] < 9)
X = X[mask]
y = y[mask]


def plot_hist(X, y):
    bins = 30
    eps = 1e-3

    H0, xedges, yedges = np.histogram2d(
        X[:, 0][y == 0],
        X[:, 1][y == 0],
        density=False,
        bins=bins)
    H1, xedges, yedges = np.histogram2d(
        X[:, 0][y == 1],
        X[:, 1][y == 1],
        density=False,
        bins=(xedges, yedges),)

    H = H1 / (H0 + H1 + eps)

    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2

    # np.meshgrid(xcenters, ycenters)
    x0, x1 = np.array(list(product(xcenters, ycenters))).T
    x1 = x1.flatten()
    x0 = x0.flatten()
    H = H.flatten()

    fig, ax = plt.subplots(1, 1)
    sc = ax.scatter(x0, x1, c=H, cmap=plt.cm.RdBu, s=92, marker="s", alpha=1)
    ax.set_xlabel(cfg.features[0])
    ax.set_ylabel(cfg.features[1])
    plt.colorbar(sc)

    name = 'hist_scatter_train.pdf' if TRAIN_ONLY else 'hist_scatter.pdf'
    save_fig(name)

    # from scipy.interpolate import interp2d
    # mask = ~np.isnan(H)
    # f = interp2d(x0[mask], x1[mask], H[mask])
    # x = np.linspace(0, 6, 100)
    # y = np.linspace(0, 10, 100)
    # xx, yy = np.meshgrid(x, y)
    # H2 = f(x, y)
    # sc = ax.scatter(xx.flatten(), yy.flatten(), c=H2.flatten(),
    #                 cmap=plt.cm.RdBu, s=92, marker="s", alpha=1)
    # ax.set_xlabel(cfg.features[0])
    # ax.set_ylabel(cfg.features[1])
    # plt.colorbar(sc)
    # path = join(dirs.plots, 'hist_interp_train.pdf')
    # plt.savefig(path)
    # print(f"saved hist to {path}")
    # plt.close()
    return H, x0, x1


def save_fig(name):
    path = join(dirs.plots, name)
    plt.savefig(path)
    print(f"saved hist to {path}")
    plt.close()


def plot_scatter(X, y):
    df = pd.DataFrame(X, columns=['minipchi2', 'sumpt'])
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='0', s=1, alpha=.5, c='crimson')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='1', s=1, alpha=.5, c='royalblue')
    plt.legend(markerscale=10, framealpha=.9)
    plt.xlabel('minipchi2')
    plt.ylabel('sumpt')
    name = 'scatter_train.pdf' if TRAIN_ONLY else 'scatter.pdf'
    save_fig(name)
    return df


def plot_kde(y, df):
    sns.kdeplot(data=df[y == 0], x='minipchi2', y='sumpt', label='0', color='crimson')
    sns.kdeplot(data=df[y == 1], x='minipchi2', y='sumpt', label='1', color='royalblue')
    plt.legend()
    name = 'kde_train.pdf' if TRAIN_ONLY else 'kde.pdf'
    save_fig(name)


H, x0, x1 = plot_hist(X, y)
df = plot_scatter(X, y)
plot_kde(y, df)
