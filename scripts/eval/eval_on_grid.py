from itertools import product

import numpy as np
from hlt2trk.models import load_model
from hlt2trk.utils import config
from hlt2trk.utils.data import get_data

from hlt2trk.models import get_evaluator

cfg = config.get_config()
model = load_model(cfg)
eval_fun = get_evaluator(cfg)

X = get_data(cfg)
X = X[X.validation][cfg.features].values

nfeats = len(cfg.features)


limits = [np.quantile(X[:, i], (0.02, 0.98)) for i in range(nfeats)]
linspaces = [np.linspace(*xi, 100 if nfeats == 2 else 30) for xi in limits]
grid = np.array(tuple(product(*linspaces)))


Y = eval_fun(model, grid).flatten()

# save the numbers to disk
np.savez_compressed(config.format_location(config.Locations.gridXY, cfg), X=grid, Y=Y)
