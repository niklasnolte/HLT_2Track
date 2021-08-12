from functools import lru_cache
import re
from os.path import abspath, dirname, join
from typing import Iterable, Optional
from collections import OrderedDict
from warnings import warn
from .utils import load_config

import torch


class dirs:
    project_root = abspath(dirname(__file__) + "/../..")
    models = join(project_root, "models")
    plots = join(project_root, "plots")
    heatmaps = join(plots, "heatmaps")
    scatter = join(plots, "scatter")
    gifs = join(plots, "gifs")
    violins = join(plots, "violins")
    data = join(project_root, "data")
    raw_data = join(data, "raw")
    savepoints = join(project_root, "savepoints")
    results = join(project_root, "results")
    results_eff = join(results, "eff")
    results_latex = join(results, "latex")


class Locations:
    project_root = abspath(dirname(__file__) + "/../..")
    model = join(
        dirs.models,
        "{model}_{features}_{data_type}_{normalize}_{signal_type}_{presel_conf}"
        "_{max_norm}_{regularization}_{division}.pkl",
    )
    data = join(dirs.data, "MC_{data_type}_{presel_conf}.pkl")
    # grid evaluation
    gridXY = join(
        dirs.savepoints,
        "gridXY_{model}_{features}_{data_type}_{normalize}"
        "_{signal_type}_{presel_conf}_{max_norm}_{regularization}_{division}.npz",
    )
    # plots
    train_distribution_gif = join(
        dirs.gifs,
        "training_distributions_{model}_{features}_{data_type}_{normalize}"
        "_{signal_type}_{presel_conf}_{max_norm}_{regularization}_{division}.gif",
    )
    heatmap = join(
        dirs.heatmaps,
        "heatmap_{model}_{features}_{data_type}_{normalize}"
        "_{signal_type}_{presel_conf}_{max_norm}_{regularization}_{division}.pdf",
    )
    twodim_vs_output = join(
        dirs.scatter,
        "twodim_vs_output_{model}_{features}_{data_type}_{normalize}"
        "_{signal_type}_{presel_conf}_{max_norm}_{regularization}_{division}.pdf",
    )
    feat_vs_output = join(
        dirs.scatter,
        "feat_vs_output_{model}_{features}_{data_type}_{normalize}"
        "_{signal_type}_{presel_conf}_{max_norm}_{regularization}_{division}.pdf",
    )
    roc = join(
        dirs.scatter,
        "roc_{model}_{features}_{data_type}_{normalize}_{signal_type}_{presel_conf}"
        "_{max_norm}_{regularization}_{division}.pdf",
    )
    rate_vs_eff = join(
        dirs.scatter,
        "rate_vs_eff_{model}_{features}_{data_type}_{normalize}"
        "_{signal_type}_{presel_conf}_{max_norm}_{regularization}_{division}.pdf",
    )
    presel_efficiencies = join(
        dirs.results, "presel_efficiencies_{data_type}_{presel_conf}.json",
    )
    auc_acc = join(
        dirs.results,
        "auc_acc_{model}_{features}_{data_type}_{normalize}"
        "_{signal_type}_{presel_conf}_{max_norm}_{regularization}_{division}.json",
    )
    target_effs = join(
        dirs.results_eff,
        "target-eff_{model}_{features}_{data_type}_{normalize}"
        "_{signal_type}_{presel_conf}_{max_norm}_{regularization}_{division}.pkl",
    )
    target_cut = join(
        dirs.results_eff,
        "target-cut_{model}_{features}_{data_type}_{normalize}"
        "_{signal_type}_{presel_conf}_{max_norm}_{regularization}_{division}.txt",
    )
    full_effs = join(
        dirs.results_eff,
        "full-eff_{model}_{features}_{data_type}_{normalize}"
        "_{signal_type}_{presel_conf}_{max_norm}_{regularization}_{division}.pkl",
    )
    violins = join(
        dirs.violins,
        "violins_{features}_{data_type}_{normalize}"
        "_{signal_type}_{presel_conf}_{max_norm}_{regularization}_{division}.pdf",
    )
    eff_table = join(
        dirs.results_latex,
        "eff_table_{features}_{data_type}_{normalize}"
        "_{signal_type}_{presel_conf}_{max_norm}_{regularization}_{division}.txt",
    )


def to_string_features(features: Optional[list]) -> str:
    if features is None:
        return "None"
    return "+".join(features)


def from_string_features(features: str) -> Optional[list]:
    if features == "None":
        return None
    return features.split("+")


def to_string_normalize(normalize: Optional[bool]) -> str:
    if normalize is None:
        return "None"
    return "normed" if normalize else "unnormed"


def from_string_normalize(normalize: str) -> Optional[bool]:
    if normalize == "None":
        return None
    return normalize == "normed"


def to_string_presel_conf(presel_conf: Optional[dict]) -> str:
    if presel_conf is None:
        return "None"
    return "+".join([f"{k}:{v}" for k, v in presel_conf.items()])


def from_string_presel_conf(presel_conf: str) -> Optional[dict]:
    if presel_conf == "None":
        return None
    return {k: v for k, v in (kv.split(":") for kv in presel_conf.split("+"))}


def to_string_max_norm(max_norm: Optional[bool]) -> str:
    if max_norm is None:
        return "None"
    return "max-norm" if max_norm else "always-norm"


def from_string_max_norm(max_norm: str) -> Optional[bool]:
    if max_norm == "max-norm":
        return True
    if max_norm == "None":
        return None
    return False


def format_location(location: str, config):
    return location.format(
        model=config.model,
        features=to_string_features(config.features),
        data_type=config.data_type,
        normalize=to_string_normalize(config.normalize),
        signal_type=config.signal_type,
        presel_conf=to_string_presel_conf(config.presel_conf),
        max_norm=to_string_max_norm(config.max_norm),
        regularization=config.regularization,
        division=config.division,
    )


def get_cli_args(config) -> str:
    """
    config has a subset of .model, .features, .data_type, .normalize
    """
    argstr = ""
    if hasattr(config, "model"):
        argstr += f"--model={config.model} "
    if hasattr(config, "features"):
        argstr += f"--features='{from_string_features(config.features)}' "
    if hasattr(config, "data_type"):
        argstr += f"--data_type={config.data_type} "
    if hasattr(config, "normalize"):
        argstr += f"--normalize={from_string_normalize(config.normalize)} "
    if hasattr(config, "signal_type"):
        argstr += f"--signal_type={config.signal_type} "
    if hasattr(config, "presel_conf"):
        # need to double curly brace, so f strings do not work here
        argstr += (
            "--presel_conf='{"
            + str(from_string_presel_conf(config.presel_conf))
            + "}' "
        )
    if hasattr(config, "max_norm"):
        argstr += f"--max_norm={from_string_max_norm(config.max_norm)} "
    if hasattr(config, "regularization"):
        argstr += f"--regularization={config.regularization} "
    if hasattr(config, "division"):
        argstr += f"--division={config.division} "
    return argstr


Configs = load_config(join(dirs.project_root, "config.yml"))


class Configuration:
    def __init__(
        self,
        model: Optional[str] = None,
        features: Optional[list] = None,
        normalize: Optional[bool] = None,
        data_type: Optional[str] = None,
        signal_type: Optional[str] = None,
        presel_conf: Optional[dict] = None,
        max_norm: Optional[bool] = None,
        regularization: Optional[str] = None,
        division: Optional[str] = None,
        seed: int = Configs.seed,
        use_cuda: bool = Configs.use_cuda,
        sigma_final: float = Configs.sigma_final,
        sigma_init: float = Configs.sigma_init,
        gamma_final: float = Configs.gamma_final,
        gamma_init: float = Configs.gamma_init,
        plot_style: bool = Configs.plot_style,
    ):

        self.model = model
        self.features = features
        self.normalize = normalize
        self.data_type = data_type
        self.signal_type = signal_type
        self.presel_conf = presel_conf
        self.max_norm = max_norm
        self.regularization = regularization
        self.division = division
        self.seed = seed
        self.sigma_final = sigma_final
        self.sigma_init = sigma_init
        self.gamma_final = gamma_final
        self.gamma_init = gamma_init
        self.plot_style = plot_style

        self.device = torch.device("cpu")
        if use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                warn("use_cuda is set to True but CUDA is unavailable...")

    def __str__(self):
        return "\n".join(
            (
                f"model={self.model}",
                f"features={self.features}",
                f"normalize={self.normalize}",
                f"data_type={self.data_type}",
                f"signal_type={self.signal_type}",
                f"presel_conf={self.presel_conf}",
                f"max_norm={self.max_norm}",
                f"regularization={self.regularization}",
                f"division={self.division}",
                f"seed={self.seed}",
                f"device={self.device}",
            )
        )

    def __repr__(self):
        return self.__str__()


def expand_with_rules(location, **cfg):
    """
    snakemake like expand function,
    but conditional expansion based on the rules in valid_config.
    The location is expanded with None for a key if no rule in the
    current expansion is valid with the key configuration
    """
    for k, v in cfg.items():
        cfg[k] = list(v)
        if cfg[k] == []:
            return []

    def valid_config(cfg: dict, key: str, value):
        # rules for combinations
        # if you have a new rule to restrict combinations, add it here
        if key == "presel_conf":
            if cfg["data_type"] == "standalone":
                # standalone does not support preselections
                return False
        if key in ["max_norm", "regularization", "division"]:
            # only regularized nn models have these keywords
            regularized_models = [
                "nn-inf",
                "nn-inf-oc",
                "nn-inf-large",
                "nn-inf-mon-vchi2",
                "nn-one",
            ]
            if "model" not in cfg:
                # only consider the keywords if a regularized model is used
                if not any([m in regularized_models for m in Configs.model]):
                    return False
            elif cfg.get("model") not in regularized_models:
                return False
        if key == "features":
            if cfg.get("model") == "nn-inf-mon-vchi2":
                if len(from_string_features(value)) == 2:
                    return False
        return True

    def expand(These: Iterable[dict], key: str, With: Iterable):
        # expand configurations in a cartesian product fashion
        # with a new list
        for t in These:
            any = False
            for w in With:
                if valid_config(t, key, w):
                    any = True
                    new = t.copy()
                    new[key] = w
                    yield new
            if not any:
                yield t

    cfgs = [{}]

    for key, vals in cfg.items():
        cfgs = expand(cfgs, key, vals)

    def format_if_present(loc, **kwargs):
        # replace the existing keywords
        for k, v in kwargs.items():
            to_replace = "{" + k + "}"
            if to_replace in loc:
                loc = loc.replace(to_replace, str(v))
        # remove the ones that were not filled
        loc = re.sub("{.*?}", "None", loc)
        return loc

    out = [format_if_present(location, **cfg) for cfg in cfgs]
    return out


@lru_cache(1)
def get_config() -> Configuration:
    from fire import Fire

    return Fire(Configuration)


def get_config_from_file(file):
    obj = load_config(file)

    def default(x):
        return x[0] if type(x) == list else x

    obj_default = {k: default(v) for k, v in obj.__dict__.items()}
    return Configuration(**obj_default)


def feature_repr(feature):
    if feature == "minipchi2":
        return "log(min($\chi^2_{IP}$))"
    elif feature == "sumpt":
        return "$\sum_{tracks}p_{T}$ [GeV]"
    elif feature == "fdchi2":
        return "log($\chi^2_{FD}$)"
    elif feature == "vchi2":
        return "$\chi^2_{Vertex}$"


# signal sample eventtypes
evttypes = [
    # 11102521,
    11104054,
    11104055,
    11104056,
    11104057,
    11104058,
    11164063,
    11166107,
    11196000,
    11196011,
    11196099,
    11264001,
    11264011,
    11874004,
    12103009,
    12103019,
    12103028,
    12103038,
    12103041,
    12103051,
    12103406,
    12103422,
    12103423,
    12103443,
    12103445,
    15364010,
    # 16103130,
    # 16103332,
    # 21101411,
    # 21103100,
    21113000,
    21163002,
    21163012,
    21163022,
    21163032,
    21163042,
    23103012,
    23103042,
    23103062,
    # 23103100,
    # 23103110,
    23163003,
    23163052,
    25113000,
    # 26104186,
    # 26104187,
    26106182,
    27163003,
    27173002,
    27225003,
    27375075
    # buggy
    # 15104142
    # 12163001
    # 16103330
    # 21101402
    # 21103110
    # 21113016
    # 12101401
    # 21123203
    # 12103110
    # 21123240
    # 11264001
    # 25103102
    # 12103444
    # 25123000
    # 12163021
    # 12165106
    # 13264021
    # 27163206
    # 13264031
    # 27163207
    # 15102320
    # 16103131
]

evttypes = OrderedDict((i + 1, evttype) for i, evttype in enumerate(evttypes))
