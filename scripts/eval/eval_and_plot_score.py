import numpy as np
import matplotlib.pyplot as plt

from hlt2trk.utils import config
from hlt2trk.utils.data import get_data, is_signal
from hlt2trk.models import load_model
from evaluate import eval_bdt, eval_torch_network, eval_simple

cfg = config.get_config()


def eval(data):
    X = data[cfg.features].to_numpy()

    model = load_model(cfg)

    if cfg.model == "bdt":
        return eval_bdt(model, X)
    elif cfg.model in ["sigma", "regular"]:
        return eval_torch_network(model, X)
    elif cfg.model in ["lda", "qda", "gnb"]:
        return eval_simple(model, X)


def plot_rates_vs_effs(data):
    truths = is_signal(cfg, data['signal_type'])
    print(truths.mean())
    preds = data["pred"]
    cutrange = np.linspace(0, 1, 50)
    minbias_preds = preds[data.eventtype == 0]
    rates = [(minbias_preds > i).mean() for i in cutrange]

    _, ax = plt.subplots()
    for mode in data.eventtype.unique():
        pred = preds[data.eventtype == mode]
        truth = truths[data.eventtype == mode]
        if mode != 0:
            eff = [(pred[truth > 0] > i).mean() for i in cutrange]
            ax.plot(rates, eff, label=f"mode = {mode}")
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
    ax.legend(loc="lower right")
    plt.savefig(config.format_location(config.Locations.rate_vs_eff, cfg))


data = get_data(cfg)

data["pred"] = eval(data)

# per event performance evaluation
grpd = data.groupby(["eventtype", "EventInSequence"]).agg(max).reset_index()

plot_rates_vs_effs(grpd)
