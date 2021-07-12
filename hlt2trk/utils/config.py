from os.path import dirname, join, abspath
from functools import lru_cache

class Configs:
    # list all possible configurations (cartesian product of these)
    model = ("regular", "sigma", "bdt")
    data_type = ("lhcb",)#, "standalone")
    features = (["minipchi2", "sumpt"], ["fdchi2", "sumpt"], ["fdchi2", "sumpt", "vchi2", "minipchi2"])
    normalize = (False, True)


class Configuration:
    def __init__(
        self,
        model: str = Configs.model[0],
        features: list = Configs.features[0],
        normalize: bool = Configs.normalize[0],
        data_type: str = Configs.data_type[0],
    ) -> None:
        assert model in Configs.model
        assert data_type in Configs.data_type
        assert features in Configs.features
        assert normalize in Configs.normalize

        self.model = model
        self.normalize = normalize
        self.data_type = data_type
        self.features = features

    def __str__(self):
        return "\n".join((
            f"model={self.model}",
            f"features={self.features}",
            f"data_type={self.data_type}",
            f"normalize={self.normalize}",
        ))


class Locations:
    project_root = abspath(dirname(__file__) + "/../..")
    model = join(project_root, "models/{model}_{features}_{data_type}_{normalize}.pkl")
    data = join(project_root, "data/MC_{data_type}.pkl")
    # grid evaluation
    grid_X = join(
        project_root, "savepoints/gridX_{model}_{features}_{data_type}_{normalize}.npy"
    )
    grid_Y = join(
        project_root, "savepoints/gridY_{model}_{features}_{data_type}_{normalize}.npy"
    )
    # plots
    train_distribution_gif = join(
        project_root,
        "plots/training_distributions_{model}_{features}_{data_type}_{normalize}.gif",
    )
    heatmap = join(project_root, "plots/heatmap_{model}_{features}_{data_type}_{normalize}.pdf")
    twodim_vs_output = join(project_root, "plots/twodim_vs_output_{model}_{features}_{data_type}_{normalize}.pdf")
    feat_vs_output = join(project_root, "plots/feat_vs_output_{model}_{features}_{data_type}_{normalize}.pdf")
    roc = join(project_root, "plots/roc_{model}_{features}_{data_type}_{normalize}.pdf")

def to_string_features(features: list):
    return "+".join(features)


def from_string_features(features: str):
    return features.split("+")


def to_string_normalize(normalize: bool):
    return "normed" if normalize else "unnormed"


def from_string_normalize(normalize: str):
    return normalize == "normed"


def format_location(location, config):
    return location.format(
        model=config.model,
        features=to_string_features(config.features),
        data_type=config.data_type,
        normalize=to_string_normalize(config.normalize),
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
    return argstr


@lru_cache(1)
def get_config() -> Configuration:
  from fire import Fire
  return Fire(Configuration)
