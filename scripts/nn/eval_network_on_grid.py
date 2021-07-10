import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from itertools import product
from hlt2trk.utils import config
from hlt2trk.utils.data import get_data
import lightgbm as lgb
from hlt2trk.models import load_model

cfg = config.get_config()

X, y = get_data(cfg)

nfeats = len(cfg.features)

torch.manual_seed(2)

model = load_model(cfg)

limits = [np.quantile(X[:, i], (0.02, 0.98)) for i in range(nfeats)]
linspaces = [np.linspace(*xi, 100 if nfeats == 2 else 20) for xi in limits]
grid = np.array(tuple(product(*linspaces)))


def _eval_torch_network(model: torch.nn.Module, grid: np.ndarray) -> np.ndarray:
    X: torch.Tensor = torch.from_numpy(grid).float()
    data = TensorDataset(X)
    # we might run out of data
    loader = DataLoader(data, batch_size=100000, shuffle=False)
    Y: torch.Tensor = torch.zeros(len(X), 1)
    idx = 0
    with torch.no_grad():
        for (x,) in loader:
            y: torch.Tensor = model(x)
            Y[idx : idx + len(y)] = y
            idx += len(y)
    return Y.numpy()


def _eval_bdt(model: lgb.Booster, grid: np.ndarray) -> np.ndarray:
    return model.predict(grid)


if cfg.model == "bdt":
    eval_fun = _eval_bdt
elif cfg.model in ["regular", "sigma"]:
    eval_fun = _eval_torch_network

Y = eval_fun(model, grid).flatten()

# persist the numbers
np.savez(config.format_location(config.Locations.grid_X, cfg), grid)
np.savez(config.format_location(config.Locations.grid_Y, cfg), Y)
