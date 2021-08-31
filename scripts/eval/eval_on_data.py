import numpy as np
import json
import pickle

from hlt2trk.utils.config import get_config, Locations, format_location, evttypes
from hlt2trk.utils.data import get_data, is_signal
from hlt2trk.models import load_model, get_evaluator
import pandas as pd

# Load configuration
cfg = get_config()
eval_fun = get_evaluator(cfg)
model = load_model(cfg)
data = get_data(cfg)

evt_grp = ["eventtype", "EventInSequence"]

# get presel_efficiencies
with open(format_location(Locations.presel_efficiencies, cfg), "r") as f:
    presel_effs = {int(k): v for k, v in json.load(f).items()}


# Evaluate model on data
data = data[data.validation]
data["pred"] = eval_fun(model, data[cfg.features].to_numpy())
# per event performance evaluation
data["tos_pred"] = data.pred * is_signal(cfg, data)


# TODO this only works for heavy-flavor right now
def rates_vs_effs(data, presel_effs):
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

    # save efficiencies at 660hz target rate for each mode
    target_effs = []
    target_tos_effs = []
    modes = []
    # save all efficiencies for each mode
    efficiencies = {0: rates}

    for mode in data.eventtype.unique():
        tos_pred = tos_preds[tos_preds.eventtype == mode].tos_pred.values
        pred = preds[preds.eventtype == mode].pred.values
        if mode != 0:
            eff = [presel_effs[mode] * (pred > i).mean() for i in cutrange]
            tos_eff = [presel_effs[mode] * (tos_pred > i).mean() for i in cutrange]
            # save all efficiencies
            efficiencies[mode] = {"eff": eff, "tos_eff": tos_eff}
            # save target efficiencies
            target_effs.append(eff[target_rate_idx])
            target_tos_effs.append(tos_eff[target_rate_idx])
            modes.append(evttypes[int(mode)])

    df = pd.DataFrame(np.stack([modes, target_effs, target_tos_effs]).T)
    df.columns = ["mode", "eff", "tos_eff"]
    df = df.astype({"mode": int})
    return df, efficiencies, cutrange[target_rate_idx]


df, efficiencies, cut = rates_vs_effs(data, presel_effs)

with open(format_location(Locations.target_cut, cfg), "w") as f:
    f.write(str(cut))

with open(format_location(Locations.full_effs, cfg), "wb") as f:
    pickle.dump(efficiencies, f)

df.to_pickle(format_location(Locations.target_effs, cfg))
