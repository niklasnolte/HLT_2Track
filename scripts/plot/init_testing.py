# %%
from functools import partial
import math
from itertools import product
from os.path import join
import matplotlib

import numpy as np
import pytorch_lightning as pl
import torch
from hlt2trk.models import LightModule
from hlt2trk.utils import dirs
from InfinityNorm import infnorm
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.nn import init
from torch.utils.data import DataLoader, TensorDataset

plt.style.use("paper-dark")
# TODO look what random init looks like
# conclusion: very little spread. Even less when the network is deep. When input
# magnitude is increased
# the spread increases sublinearly.
# TODO training to fit -x does that make convergence better
# TODO one hot init
# %%
x = torch.linspace(-2, 2, 100) * 10
X = torch.Tensor(list(product(x, x)))


def f(X):
    return torch.sin(X.prod(dim=1, keepdim=True) / 100)


Y = f(X)

fig, ax = plt.subplots(1, 1)
sc = ax.scatter(X[:, 0], X[:, 1], c=Y,
                cmap=plt.cm.RdBu, s=9, marker="s", alpha=1)
plt.colorbar(sc)
plt.show()

# %%


def infnorm2(m: nn.Module, name='weight') -> nn.Module:
    def absi(m: nn.Module, _) -> None:
        weight = getattr(m, name + '_orig')
        norms = weight.abs().sum(axis=0)
        weight = weight / torch.max(torch.ones_like(norms), norms)
        setattr(m, name, weight)
    w = m._parameters[name]
    delattr(m, name)
    m.register_parameter(name + "_orig", w)
    setattr(m, name, w.data)
    m.register_forward_pre_hook(absi)
    return m
# %%


class Linear2(torch.nn.Linear):
    def reset_parameters(self) -> None:
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            self.bias = torch.nn.Parameter(torch.Tensor([0]))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        n_output, n_input = self.weight.shape
        std = .1 / math.sqrt(n_output)
        # init.normal_(self.weight, mean=0, std=std)
        init.uniform_(self.weight, a=-std, b=std)
        mask = torch.randint(0, n_input, (n_output, ))
        with torch.no_grad():
            self.weight[np.arange(n_output), mask] = 1


def test_init(X, norm, layer, title=""):
    torch.manual_seed(32)
    fig, axes = plt.subplots(5, 2, sharex=True, sharey=True, figsize=(8, 12))
    for ax in axes.flatten():
        model = torch.nn.Sequential(norm(layer(2, 64)), torch.nn.ReLU(),
                                    norm(layer(64, 64)), torch.nn.ReLU(),
                                    norm(layer(64, 64)), torch.nn.ReLU(),
                                    layer(64, 1))
        y = model(X)
        sc = ax.scatter(X[:, 0], X[:, 1], c=y.detach().numpy(),
                        cmap=plt.cm.RdBu, s=9, marker="s", alpha=1)
        plt.colorbar(sc, ax=ax)
    fig.suptitle(title, y=1)
    fig.tight_layout()
    plt.show()


# %%
test_init(X, partial(infnorm, alpha=1 / 100, always_norm=False), Linear2,
          title="init: onehot " "norm: unsafe" r"$a=.01$")
# %%
test_init(X, lambda x: x, torch.nn.Linear,
          title="init: regular " "unnormed ")
# %%
test_init(X, partial(infnorm, always_norm=False), Linear2,
          title="init: onehot " "norm:safe")
# %%
X_train = torch.from_numpy(np.random.uniform(X.min(), X.max(), (1000, 2))).float()
Y_train = f(X_train).float()
train_dataloader = DataLoader(
    TensorDataset(X_train, Y_train),
    batch_size=32, shuffle=True)
# %%
# Define a model


class Network(torch.nn.Module):
    def __init__(self, skip=False, one_hot_init=False, norm=None):
        super().__init__()
        norm = norm if norm is not None else lambda x: x

        def Lin(*args):
            f = Linear2(*args) if one_hot_init else torch.nn.Linear(*args)
            return norm(f)

        self.gx = torch.nn.Sequential(Lin(2, 64), torch.nn.ReLU(),
                                      Lin(64, 64), torch.nn.ReLU(),
                                      Lin(64, 64), torch.nn.ReLU(),
                                      Lin(64, 1))
        self.skip = skip

    def forward(self, x: torch.Tensor):
        gx = self.gx(x)
        if self.skip:
            return gx - x.sum(dim=1, keepdim=True)
        else:
            return gx


def fit(
        model, train_dataloader, name, max_epochs=50, dir="", learning_rate=1e-3,
        loss=torch.nn.MSELoss()):
    module = LightModule(model, learning_rate=1e-3, loss=loss)
    logger = TensorBoardLogger(
        join(dirs.project_root, "lightning_logs", dir),
        default_hp_metric=False, name=name)

    trainer = pl.Trainer(logger=logger, max_epochs=max_epochs)
    trainer.fit(module, train_dataloader=train_dataloader)
    return module


def plot_heatmap(X, Y, module):
    Ypred = module(X).detach()
    fig, axes = plt.subplots(3, 2, sharey=True, figsize=(6, 10))
    [plt.delaxes(ax=ax) for ax in axes[1:].flatten()]
    ax, ax1 = axes[0]
    sc = ax.scatter(X[:, 0], X[:, 1], c=Ypred,
                    cmap=plt.cm.RdBu, s=9, marker="s", alpha=1, vmin=-1, vmax=1)
    sc = ax1.scatter(X[:, 0], X[:, 1], c=Y,
                     cmap=plt.cm.RdBu, s=20, marker="s", alpha=1, vmin=-1, vmax=1)
    cax = ax1.inset_axes([1.04, 0.2, 0.05, 0.6])
    plt.colorbar(sc, cax=cax)
    ax = fig.add_subplot(3, 2, (3, 6))
    sc = ax.scatter(X[:, 0], X[:, 1], c=Ypred - Y.view(-1, 1),
                    cmap=plt.cm.RdBu, s=9, marker="s", alpha=1)
    fig.colorbar(sc, ax=ax, orientation="horizontal")
    plt.tight_layout()
    plt.show()


# %%
pl.seed_everything(32)
model = Network(skip=False)
name = "regular"
dir = "large-input"
module = fit(model, train_dataloader, name, dir=dir, max_epochs=100)
plot_heatmap(X, Y, module)

# %%
pl.seed_everything(32)
model = Network(skip=True,)
name = "skip-large_input"
dir = "large-input"
module = fit(model, train_dataloader, name, dir=dir, max_epochs=100)
plot_heatmap(X, Y, module)

# %%
pl.seed_everything(32)
model = Network(skip=True, one_hot_init=True)
name = "skip-onehot"
dir = "large-input"
module = fit(model, train_dataloader, name, dir=dir, max_epochs=100)
plot_heatmap(X, Y, module)
# %%
pl.seed_everything(32)
model = Network(skip=False, one_hot_init=True)
name = "regular-onehot"
dir = "large-input"
module = fit(model, train_dataloader, name, dir=dir, max_epochs=100)
plot_heatmap(X, Y, module)

# %%
