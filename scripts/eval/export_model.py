from hlt2trk.utils.config import Locations, format_location, get_config
from hlt2trk.models import load_model
from InfinityNorm import get_normed_weights
import json

cfg = get_config()

with open(format_location(Locations.target_cut, cfg), "r") as f:
    target_cut = float(f.read())

model = load_model(cfg)

if cfg.model == "nn-regular":
    state_dict = model.state_dict()
    for k, v in state_dict.items():
        state_dict[k] = v.tolist()
elif cfg.model == "nn-inf":
    if cfg.division == "vector":
        vectorwise = True
    elif cfg.division == "scalar":
        vectorwise = False
    state_dict = model.state_dict()
    weight_keys = [x for x in state_dict if "weight" in x]
    depth = len(weight_keys)
    for k in state_dict:
        if k in weight_keys:
            state_dict[k] = get_normed_weights(
                state_dict[k],
                # ordered dict -> last weight_key is the one that
                # needs to be one-inf normed
                "one-inf" if k == weight_keys[-1] else "inf",
                always_norm=not cfg.max_norm,
                alpha=cfg.sigma_final ** (1.0 / depth),
                vectorwise=vectorwise,
            ).tolist()
        else:
            state_dict[k] = state_dict[k].tolist()
else:
    raise NotImplementedError

state_dict["nominal_cut"] = target_cut

with open(format_location(Locations.exported_model, cfg), "w") as f:
    json.dump(state_dict, f)
