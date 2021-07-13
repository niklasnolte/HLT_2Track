from itertools import product
from typing import Union

import lightgbm as lgb
import numpy as np
import torch
from hlt2trk.models import load_model
from hlt2trk.utils import config
from hlt2trk.utils.data import get_data

from evaluate import eval_bdt, eval_simple, eval_torch_network

cfg = config.get_config()

X, y = get_data(cfg)

nfeats = len(cfg.features)

torch.manual_seed(2)

model = load_model(cfg)

limits = [np.quantile(X[:, i], (0.02, 0.98)) for i in range(nfeats)]
linspaces = [np.linspace(*xi, 100 if nfeats == 2 else 20) for xi in limits]
grid = np.array(tuple(product(*linspaces)))


if cfg.model == "bdt":
    eval_fun = eval_bdt
elif cfg.model in ["regular", "sigma"]:
    eval_fun = eval_torch_network
elif cfg.model in ["lda", "qda", "gnb"]:
    eval_fun = eval_simple

Y = eval_fun(model, grid).flatten()

# persist the numbers
np.savez_compressed(config.format_location(config.Locations.gridXY, cfg), X=grid, Y=Y)