import numpy as np
from hlt2trk.models import load_model
from hlt2trk.utils import Locations, config
from hlt2trk.utils.data import get_data_for_training
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from evaluate import eval_bdt, eval_simple, eval_torch_network

cfg = config.get_config()
model = load_model(cfg)

_, _, x_val, y_val = get_data_for_training(cfg)

if cfg.model == "bdt":
    eval_fun = eval_bdt
elif cfg.model.startswith("nn"):
    eval_fun = eval_torch_network
elif cfg.model in ["lda", "qda", "gnb"]:
    eval_fun = eval_simple
else:
    raise ValueError(f"Unknown model: {cfg.model}, please specify eval function in case\
         this is a new model")

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
