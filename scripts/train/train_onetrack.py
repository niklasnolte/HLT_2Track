import json
from hlt2trk.utils.data import get_data_onetrack
from hlt2trk.utils.config import get_config, format_location, Locations, evttypes
from collections import defaultdict
import numpy as np
import pandas as pd
import pickle

input_rate = 30000
target_rate = 330  # kHz

class OneTrackMVA:
    def __init__(self, alpha=0):
        self.maxChi2Ndof = 0
        self.minPt = 2e3
        self.maxPt = 26e3
        self.minIPChi2 = 7.4
        self.minBPVz = -300
        self.param1 = 1e6
        self.param2 = 2e3
        self.param3 = 1.248
        self.alpha = alpha  # the only parameter we tune here

    def decide(self, X):
        pt = X.trk_PT
        pv_z = X.trk_OWNPV_Z
        ipchi2 = X.trk_IPCHI2_OWNPV

        ptShift = pt - self.alpha
        # TODO validate that this is in the preselection
        # dec = chi2ndof < self.maxChi2Ndof
        dec = (pv_z > self.minBPVz) & (
            ((ptShift > self.maxPt) & (ipchi2 > self.minIPChi2))
            | (
                (ptShift > self.minPt)
                & (ptShift < self.maxPt)
                & (
                    np.log(ipchi2)
                    > self.param1 / (ptShift - self.param2) ** 2
                    + (self.param3 / self.maxPt) * (self.maxPt - ptShift)
                    + np.log(self.minIPChi2)
                )
            )
        )
        return dec


def tune_one_track(df, presel_effs):
    # we need to tune the alpha parameter to match a rate
    evt_grp = ["eventtype", "EventInSequence"]
    onetrack = OneTrackMVA()
    df["dec"] = False
    # get presel efficiency
    effs = defaultdict(list)
    tos_effs = defaultdict(list)
    alphas = np.arange(-500, 500, 5)
    evttypeidxs = [0] + list(evttypes.keys())

    for alpha in alphas:
        onetrack.alpha = alpha
        df["dec"] = onetrack.decide(df)
        df["tos_dec"] = df.dec * df.trk_signal_type > 0
        decs = df.groupby(evt_grp).dec.max().groupby("eventtype").mean()
        tos_decs = df.groupby(evt_grp).tos_dec.max().groupby("eventtype").mean()
        for i in evttypeidxs:
          effs[i] += [presel_effs[i] * decs[i]]
          tos_effs[i] += [presel_effs[i] * tos_decs[i]]
        #make rates out of "minbias efficiencies"
    effs[0] = [x * input_rate for x in effs[0]]

    nominal_idx = [
        idx for idx, eff in enumerate(effs[0]) if eff < target_rate
    ][0] # get the right rate
    target_effs = [effs[i][nominal_idx] for i in evttypeidxs]
    target_tos_effs = [tos_effs[i][nominal_idx] for i in evttypeidxs]
    modes = [30000000] + [evttypes[i] for i in evttypeidxs[1:]]
    df = pd.DataFrame(np.stack([modes, target_effs, target_tos_effs]).T)
    df.columns = ["mode", "eff", "tos_eff"]
    df = df.astype({"mode": int})
    full_effs = {}
    for mode in effs.keys():
      full_effs[mode] = {"eff": effs[mode], "tos_eff": tos_effs[mode]}
    return df, alphas[nominal_idx], full_effs


if __name__ == "__main__":

    cfg = get_config()

    # get presel_efficiencies
    with open(format_location(Locations.presel_efficiencies_onetrack, cfg), "r") as f:
        presel_effs = {int(k): v for k, v in json.load(f).items()}

    dataframe = get_data_onetrack(cfg)
    target_effs, alpha_nominal, full_effs = tune_one_track(dataframe, presel_effs)


    with open(format_location(Locations.onetrack_ptshift, cfg), "w") as f:
        f.write(str(alpha_nominal))

    with open(format_location(Locations.onetrack_full_effs, cfg), "wb") as f:
        pickle.dump(full_effs, f)

    target_effs.to_pickle(format_location(Locations.onetrack_target_effs, cfg))


