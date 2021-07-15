from functools import lru_cache
from os.path import abspath, dirname, join
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
        "{model}_{features}_{data_type}_{normalize}_{signal_type}.pkl")
    data = join(dirs.data, "MC_{data_type}.pkl")
    # grid evaluation
    gridXY = join(
        dirs.savepoints,
        "gridXY_{model}_{features}_{data_type}_{normalize}_{signal_type}.npz")
    # plots
    train_distribution_gif = join(
        dirs.gifs,
        "training_distributions_{model}_{features}_{data_type}_{normalize}_{signal_type}.gif",
    )
    heatmap = join(
        dirs.heatmaps,
        "heatmap_{model}_{features}_{data_type}_{normalize}_{signal_type}.pdf",
    )
    twodim_vs_output = join(
        dirs.scatter,
        "twodim_vs_output_{model}_{features}_{data_type}_{normalize}_{signal_type}.pdf",
    )
    feat_vs_output = join(
        dirs.scatter,
        "feat_vs_output_{model}_{features}_{data_type}_{normalize}_{signal_type}.pdf",
    )
    roc = join(
        dirs.scatter,
        "roc_{model}_{features}_{data_type}_{normalize}_{signal_type}.pdf")
    rate_vs_eff = join(
        dirs.scatter,
        "rate_vs_eff_{model}_{features}_{data_type}_{normalize}_{signal_type}.pdf",
    )
    presel_efficiencies = join(
        dirs.results,
        "presel_efficiencies_{data_type}.json",
    )


def to_string_features(features: list):
    return "+".join(features)


def from_string_features(features: str):
    return features.split("+")


def to_string_normalize(normalize: bool):
    return "normed" if normalize else "unnormed"


def from_string_normalize(normalize: str):
    return normalize == "normed"


def format_location(location: str, config):
    return location.format(
        model=config.model,
        features=to_string_features(config.features),
        data_type=config.data_type,
        normalize=to_string_normalize(config.normalize),
        signal_type=config.signal_type,
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
    return argstr


Configs = load_config(join(dirs.project_root, 'config.yml'))


class Configuration:
    def __init__(
        self,
        model: str = Configs.model[0],
        features: list = Configs.features[0],
        normalize: bool = Configs.normalize[0],
        data_type: str = Configs.data_type[0],
        signal_type: str = Configs.signal_type[0],
        seed: int = Configs.seed,
        use_cuda: bool = Configs.use_cuda,
    ):

        self.model = model
        self.features = features
        self.normalize = normalize
        self.data_type = data_type
        self.signal_type = signal_type
        self.seed = seed

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
                f"seed={self.seed}",
                f"device={self.device}",
            )
        )


@lru_cache(1)
def get_config() -> Configuration:
    from fire import Fire
    return Fire(Configuration)
