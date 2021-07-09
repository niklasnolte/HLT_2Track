import torch
from sys import argv
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from itertools import product
from hlt2trk.utils.utils import to_np

# TODO import from config
two_dim: bool = "2d" in argv
sigmanet: bool = "sigma" in argv

import hlt2trk.utils.meta_info as meta

sig, bkg = meta.get_data()

nfeats = len(meta.features)

torch.manual_seed(2)

model = meta.load_model()

X: torch.Tensor = torch.from_numpy(np.concatenate(
    [to_np(sig, meta.features), to_np(bkg, meta.features)])).float()

limits = [np.quantile(X[:, i], (0.02, 0.98)) for i in range(nfeats)]
linspaces = [np.linspace(*xi, 100 if two_dim else 20) for xi in limits]
grid = np.array(tuple(product(*linspaces)))
X: torch.Tensor = torch.from_numpy(grid).float()
data = TensorDataset(X)
loader = DataLoader(data, batch_size=100000, shuffle=False)

Y: torch.Tensor = torch.zeros(len(X), 1)
idx = 0
with torch.no_grad():
    for (x,) in loader:
        y: torch.Tensor = model(x)
        Y[idx: idx + len(y)] = y
        idx += len(y)

# persist the numbers
torch.save(X, meta.locations.grid_X)
torch.save(Y, meta.locations.grid_Y)
