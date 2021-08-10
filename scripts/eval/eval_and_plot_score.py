import numpy as np
import matplotlib.pyplot as plt
import json

from hlt2trk.utils.config import get_config, Locations, format_location
from hlt2trk.utils.data import get_data
from hlt2trk.models import load_model
from evaluate import eval_bdt, eval_torch_network, eval_simple
import pandas as pd


int_to_evttype = {
  1: 11104054,
  2: 23103042,
  3: 11104055,
  4: 23103062,
  5: 16103332,
  6: 11104056,
  7: 23163003,
  8: 11104057,
  9: 23163052,
  10: 21101411,
  11: 11104058,
  12: 11102521,
  13: 21103100,
  14: 11164063,
  15: 11264001,
  16: 11166107,
  17: 11264011,
  18: 21113000,
  19: 11196000,
  20: 11874004,
  21: 11196011,
  22: 11196099,
  23: 12103406,
  24: 23103100,
  25: 12103009,
  26: 12103422,
  27: 23103110,
  28: 12103019,
  29: 12103423,
  30: 12103028,
  31: 12103443,
  32: 25113000,
  33: 12103038,
  34: 12103041,
  35: 12103445,
  36: 26104186,
  37: 12103051,
  38: 26104187,
  39: 26106182,
  40: 15364010,
  41: 27163003,
  42: 21163002,
  43: 21163012,
  44: 21163022,
  45: 27173002,
  46: 21163032,
  47: 15364010,
  48: 27225003,
  49: 21163042,
  50: 16103130,
  51: 27375075,
  52: 23103012,
}

cfg = get_config()

evt_grp = ["eventtype", "EventInSequence"]


def eval_(data):
    X = data[cfg.features].to_numpy()

    model = load_model(cfg)

    if cfg.model == "bdt":
        return eval_bdt(model, X)
    elif cfg.model.startswith("nn"):
        return eval_torch_network(model, X)
    elif cfg.model in ["lda", "qda", "gnb"]:
        return eval_simple(model, X)

# TODO this only works for heavy-flavor right now

def plot_rates_vs_effs(data, presel_effs):
    cutrange = np.linspace(1, 0, 500)
    tos_preds = data.groupby(evt_grp).tos_pred.agg(max).reset_index()
    preds = data.groupby(evt_grp).pred.agg(max).reset_index()

    # minimum bias rates (per event)
    minbias_preds = preds[preds.eventtype == 0].pred.values
    input_rate = 30000  # kHz
    presel_rate = presel_effs[0]  # minbias
    rates = [input_rate * presel_rate * (minbias_preds > i).mean() for i in cutrange]

    target_rate = 660
    target_rate_idx = [i for i, r in enumerate(rates) if r < target_rate][-1]

    target_eff_list = []
    target_tos_eff_list = []
    mode_list = []
    _, ax = plt.subplots()
    for mode in data.eventtype.unique():
        tos_pred = tos_preds[tos_preds.eventtype == mode].tos_pred.values
        pred = preds[preds.eventtype == mode].pred.values
        if mode != 0:
            eff = [presel_effs[mode] * (pred > i).mean() for i in cutrange]
            tos_eff = [presel_effs[mode] * (tos_pred > i).mean() for i in cutrange]
            target_eff_list.append(eff[target_rate_idx])
            target_tos_eff_list.append(tos_eff[target_rate_idx])
            mode_list.append(int_to_evttype[int(mode)])
            ax.plot(
                rates, eff, label=f"{mode:^4} / {target_eff_list[-1]:^5.4f}",
                c=f"C{mode}")
            ax.plot(
                rates, tos_eff, label=f"{mode:^4} / {target_tos_eff_list[-1]:^5.4f}",
                ls="--", c=f"C{mode}")

    ax.set_xlabel("rate (kHz)")
    ax.set_ylabel("efficiency")
    # ax.set_ylim(0, max_eff)
    ax.axvline(x=target_rate, color="red")
    ax.grid(linestyle="--")
    ax.grid(linestyle=":", which="minor")
    ax.set_title(cfg.model)
    ax.legend(loc="lower right", title="mode / eff at 660Hz")
    plt.savefig(format_location(Locations.rate_vs_eff, cfg))

    df = pd.DataFrame(np.stack([mode_list, target_eff_list, target_tos_eff_list]).T)
    df.columns = ["mode", "$\epsilon$", "$\epsilon_\rm{tos}$"]
    df = df.astype({"mode": int})

    table = df.to_latex(
        column_format="c" * 7,
        bold_rows=True,
        float_format="%.3f",
        multicolumn_format="c",
        caption="efficiencies",
        index=False,
        escape=False,
    )
    with open(format_location(Locations.eff_table, cfg), "w") as f:
        f.writelines(table)


data = get_data(cfg)
data = data[data.validation]

data["pred"] = eval_(data)

# per event performance evaluation
data = data[evt_grp + ["pred", "signal_type"]]
data["tos_pred"] = data.pred * (data.signal_type > 0)


# get presel_efficiencies
with open(format_location(Locations.presel_efficiencies, cfg), "r") as f:
    presel_effs = {int(k): v for k, v in json.load(f).items()}

plot_rates_vs_effs(data, presel_effs)
