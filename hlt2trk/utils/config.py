from functools import lru_cache
import re
from os.path import abspath, dirname, join
from typing import Iterable
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
    data = join(project_root, "data")
    raw_data = join(data, "raw")
    savepoints = join(project_root, "savepoints")
    results = join(project_root, "results")


class Locations:
    project_root = abspath(dirname(__file__) + "/../..")
    model = join(
        dirs.models,
        "{model}_{features}_{data_type}_{normalize}_{signal_type}_{presel_conf}_{max_norm}_{regularization}_{division}.pkl",
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
        "roc_{model}_{features}_{data_type}_{normalize}_{signal_type}_{presel_conf}_{max_norm}_{regularization}_{division}.pdf",
    )
    rate_vs_eff = join(
        dirs.scatter,
        "rate_vs_eff_{model}_{features}_{data_type}_{normalize}"
        "_{signal_type}_{presel_conf}_{max_norm}_{regularization}_{division}.pdf",
    )
    presel_efficiencies = join(
        dirs.results,
        "presel_efficiencies_{data_type}_{presel_conf}.json",
    )
    auc_acc = join(
        dirs.results,
        "metrics_{model}_{features}_{data_type}_{normalize}"
        "_{signal_type}_{presel_conf}_{max_norm}_{regularization}_{division}.json",
    )


def to_string_features(features: list) -> str:
    return "+".join(features)


def from_string_features(features: str) -> list:
    return features.split("+")


def to_string_normalize(normalize: bool) -> str:
    return "normed" if normalize else "unnormed"


def from_string_normalize(normalize: str) -> bool:
    return normalize == "normed"


def to_string_presel_conf(presel_conf: dict) -> str:
    return "+".join([f"{k}:{v}" for k, v in presel_conf.items()])


def from_string_presel_conf(presel_conf: str) -> dict:
    return {k: v for k, v in (kv.split(":") for kv in presel_conf.split("+"))}

def to_string_max_norm(max_norm: bool) -> str:
    return "max-norm" if max_norm else "always-norm"

def from_string_max_norm(max_norm : str) -> bool:
    return max_norm == "max-norm"


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
        argstr += ("--presel_conf='{"
            + str(from_string_presel_conf(config.presel_conf))
            + "}' ")
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
        model: str = Configs.model[0],
        features: list = Configs.features[0],
        normalize: bool = Configs.normalize[0],
        data_type: str = Configs.data_type[0],
        signal_type: str = Configs.signal_type[0],
        presel_conf: dict = Configs.presel_conf[0],
        max_norm: bool = Configs.max_norm[0],
        regularization: str = Configs.regularization[0],
        division: str = Configs.division[0],
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


def expand_with_rules(location, **cfg):
  def valid_config(cfg:dict, key:str, value):
    #rules for combinations
    # if you have a new rule to restrict combinations, add it here
    if key == "presel_conf":
      if cfg["data_type"] == "standalone":
        # standalone does not support preselections
        return False
    if key in ["max_norm", "regularization", "division"]:
      # only nn models have these keywords
      if "nn" not in cfg["model"]:
        return False
    return True

  def expand(These: Iterable[dict], key: str,  With : Iterable):
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
  
  def format_if_present(location, **kwargs):
    # replace the existing keywords
    for k, v in kwargs.items():
      to_replace = "{" + k + "}"
      if to_replace in location:
        location = location.replace(to_replace, str(v))
    # remove the ones that were not filled
    location = re.sub("{.*?}", "None", location)
    return location
  out = [format_if_present(location, **cfg) for cfg in cfgs]
  return out





@lru_cache(1)
def get_config() -> Configuration:
    from fire import Fire

    return Fire(Configuration)
