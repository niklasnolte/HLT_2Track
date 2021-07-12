import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, roc_curve

from hlt2trk.utils import config
from hlt2trk.utils import data
from hlt2trk.models import load_model
from evaluate import eval_bdt, eval_torch_network

cfg = config.get_config()

data = data.get_dataframe(cfg)


def eval(data):
    X = data[cfg.features].to_numpy()
    model = load_model(cfg)

    if cfg.model == "bdt":
        return eval_bdt(model, X)
    elif cfg.model in ["sigma", "regular"]:
        return eval_torch_network(model, X)


def plot_roc(truth, pred):
    fpr, tpr, _ = roc_curve(truth, pred)
    acc = max([balanced_accuracy_score(truth, pred > i) for i in np.linspace(0, 1, 50)])
    _, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"auc {roc_auc_score(truth, pred):.5f}, acc {acc:.5f}")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="random choice")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1, 0.05), minor=True)
    ax.set_yticks(np.arange(0, 1, 0.05), minor=True)
    ax.grid(linestyle="--")
    ax.grid(linestyle=":", which="minor")
    ax.legend(loc="lower right")
    plt.savefig(config.format_location(config.Locations.roc, cfg))


data["pred"] = eval(data)
# per event performance evaluation
Ys = data.groupby(["eventtype", "EventInSequence"])[["label", "pred"]].agg(max)

truth = Ys['label']
pred = Ys['pred']

# plot performance

cutrange = np.linspace(0,1,50)
for mode in data.eventtype:
  if mode != 0:
    rate = 
    plot_roc(truth, pred)
