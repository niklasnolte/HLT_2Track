import numpy as np
import matplotlib.pyplot as plt
import json

from hlt2trk.utils.config import get_config, Locations, format_location
from hlt2trk.utils.data import get_data
from hlt2trk.models import load_model
from evaluate import eval_bdt, eval_torch_network, eval_simple

cfg = get_config()

evt_grp = ["eventtype", "EventInSequence"]


def eval(data):
    X = data[cfg.features].to_numpy()

    model = load_model(cfg)

    if cfg.model == "bdt":
        return eval_bdt(model, X)
    elif cfg.model.startswith("nn"):
        return eval_torch_network(model, X)
    elif cfg.model in ["lda", "qda", "gnb"]:
        return eval_simple(model, X)


def roc_auc_score(rates, eff):
    eff = np.array(eff)
    rates = np.array(rates)
    return sum(center(eff) * np.diff(rates))


def center(a):
    return (a[1:] + a[:-1]) * 0.5

# TODO this only works for heavy-flavor right now


def plot_rates_vs_effs(data, presel_effs):
    cutrange = np.linspace(1, 0, 100)
    tos_preds = data.groupby(evt_grp).tos_pred.agg(max).reset_index()
    preds = data.groupby(evt_grp).pred.agg(max).reset_index()
    # truths = (data.groupby(evt_grp).signal_type.agg(max) > 0).reset_index()

    # minimum bias rates (per event)
    minbias_preds = preds[preds.eventtype == 0].pred.values
    input_rate = 30000  # kHz
    presel_rate = presel_effs[0]
    rates = [input_rate * presel_rate * (minbias_preds > i).mean() for i in cutrange]

    _, ax = plt.subplots()
    for mode in data.eventtype.unique():
        tos_pred = tos_preds[tos_preds.eventtype == mode].tos_pred.values
        pred = preds[preds.eventtype == mode].pred.values
        # truth = truths[data.eventtype == mode]
        if mode != 0:
            eff = [presel_effs[mode] * (pred > i).mean() for i in cutrange]
            tos_eff = [presel_effs[mode] * (tos_pred > i).mean() for i in cutrange]
            auc = roc_auc_score(rates / max(rates), eff)
            tos_auc = roc_auc_score(rates / max(rates), tos_eff)
            ax.plot(rates, eff, label=f"{mode:^4} / {auc:^5.4f}", c=f"C{mode}")
            ax.plot(
                rates, tos_eff, label=f"{mode:^4} / {tos_auc:^5.4f}", ls="--",
                c=f"C{mode}")
    ax.set_xlabel("rate (kHz)")
    ax.set_ylabel("efficiency")
    # ax.set_ylim(0, max_eff)
    ax.grid(linestyle="--")
    ax.grid(linestyle=":", which="minor")
    ax.set_title(cfg.model)
    ax.legend(loc="best", title="mode / auc")
    plt.savefig(format_location(Locations.rate_vs_eff, cfg))


data = get_data(cfg)

data["pred"] = eval(data)

# per event performance evaluation
data = data[evt_grp + ["pred", "signal_type"]]
data["tos_pred"] = data.pred * (data.signal_type > 0)


# get presel_efficiencies
with open(format_location(Locations.presel_efficiencies, cfg), "r") as f:
    presel_effs = {int(k): v for k, v in json.load(f).items()}

plot_rates_vs_effs(data, presel_effs)
