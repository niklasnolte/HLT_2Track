import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from itertools import product
from hlt2trk.utils.utils import to_np
from hlt2trk.utils import config
from hlt2trk.utils.data import get_data
from hlt2trk.models import load_model

cfg = config.get_config()

sig, bkg = get_data(cfg)

nfeats = len(cfg.features)

torch.manual_seed(2)

model = load_model(cfg)

X: torch.Tensor = torch.from_numpy(np.concatenate(
    [to_np(sig, cfg.features), to_np(bkg, cfg.features)])).float()

limits = [np.quantile(X[:, i], (0.02, 0.98)) for i in range(nfeats)]
linspaces = [np.linspace(*xi, 100 if nfeats == 2 else 20) for xi in limits]
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
torch.save(X, config.format_location(config.Locations.grid_X, cfg))
torch.save(Y, config.format_location(config.Locations.grid_Y, cfg))
