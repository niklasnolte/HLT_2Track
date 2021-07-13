from typing import Union

import lightgbm as lgb
import numpy as np
import torch
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.naive_bayes import GaussianNB
from torch.utils.data import DataLoader, TensorDataset


def eval_torch_network(model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    torch.manual_seed(2)
    X: torch.Tensor = torch.from_numpy(X).float()
    data = TensorDataset(X)
    loader = DataLoader(data, batch_size=100000, shuffle=False)
    Y: torch.Tensor = torch.zeros(len(X), 1)
    idx = 0
    with torch.no_grad():
        for (x,) in loader:
            y: torch.Tensor = model(x)
            Y[idx: idx + len(y)] = y
            idx += len(y)
    return Y.numpy()


def eval_bdt(model: lgb.Booster, X: np.ndarray) -> np.ndarray:
    return model.predict(X)


def eval_simple(
        model: Union[LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis,
                     GaussianNB],
        grid: np.ndarray) -> np.ndarray:
    y: np.ndarray = model.predict_proba(grid)[:, 1]
    return y