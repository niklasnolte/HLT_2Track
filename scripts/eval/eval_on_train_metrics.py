import numpy as np
from hlt2trk.models import load_model
from hlt2trk.utils import Locations, config
from hlt2trk.utils.data import get_data_for_training
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from hlt2trk.models import get_evaluator

cfg = config.get_config()
eval_fun = get_evaluator(cfg)
model = load_model(cfg)
_, _, x_val, y_val = get_data_for_training(cfg)

ypred = eval_fun(model, x_val).flatten()

# evalutation
cuts = np.linspace(0, 1, 500)
accs = [balanced_accuracy_score(y_val.reshape(-1), ypred > cut) for cut in cuts]
idx = np.argmax(accs)

cut = cuts[idx]
acc = accs[idx]
auc = roc_auc_score(y_val.reshape(-1), ypred)

with open(config.format_location(Locations.auc_acc, cfg), "w") as f:
    f.write(f"{auc}, {acc}, {cut}")
