import numpy as np
import matplotlib.pyplot as plt
import json

from hlt2trk.utils.config import get_config, Locations, format_location
from hlt2trk.utils.data import get_data, is_signal
from hlt2trk.models import load_model
from evaluate import eval_bdt, eval_torch_network, eval_simple

cfg = get_config()


def eval(data):
    X = data[cfg.features].to_numpy()

    model = load_model(cfg)

    if cfg.model == "bdt":
        return eval_bdt(model, X)
    elif cfg.model in ["sigma", "regular"]:
        return eval_torch_network(model, X)
    elif cfg.model in ["lda", "qda", "gnb"]:
        return eval_simple(model, X)


def roc_auc_score(rates, eff):
    eff = np.array(eff)
    rates = np.array(rates)
    return sum(center(eff) * np.diff(rates))


def center(a):
    return (a[1:] + a[:-1]) * 0.5


def plot_rates_vs_effs(data, presel_effs):
    truths = is_signal(cfg, data["signal_type"])
    print(truths.mean())
    preds = data["pred"]
    cutrange = np.linspace(1, 0, 100)
    minbias_preds = preds[data.eventtype == 0]
    input_rate = 30000 #kHz
    presel_rate = presel_effs[0]
    rates = [input_rate * (minbias_preds > i).mean() for i in cutrange]

    _, ax = plt.subplots()
    for mode in data.eventtype.unique():
        pred = preds[data.eventtype == mode]
        truth = truths[data.eventtype == mode]

        if mode != 0:
            eff = [(pred[truth > 0] > i).mean() for i in cutrange]
            auc = roc_auc_score(rates, eff)
            ax.plot(rates, eff, label=f"{mode:^4} / {auc:^5.4f}", c=f"C{mode}")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="random choice")
    ax.set_xlabel("rate")
    ax.set_ylabel("efficiency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1, 0.05), minor=True)
    ax.set_yticks(np.arange(0, 1, 0.05), minor=True)
    ax.grid(linestyle="--")
    ax.grid(linestyle=":", which="minor")
    ax.set_title(cfg.model)
    ax.legend(loc="lower right", title="mode / auc")
    plt.savefig(format_location(Locations.rate_vs_eff, cfg))


data = get_data(cfg)

data["pred"] = eval(data)

# per event performance evaluation
grpd = data.groupby(["eventtype", "EventInSequence"]).agg(max).reset_index()

# get presel_efficiencies
with open(format_location(Locations.presel_efficiencies, cfg), "r") as f:
    presel_effs = {int(k) : v for k,v in json.load(f).items()}

plot_rates_vs_effs(grpd, presel_effs)
