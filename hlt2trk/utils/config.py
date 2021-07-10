from os.path import dirname, join, abspath
from functools import lru_cache

# class Configuration:
#     def __init__(self, model: str = 'LDA', experiment: int = 0,
#                  normalize: bool = True, data: str = 'lhcb') -> None:
#         """Configure everything.

#         Args:
#             model (str, optional): Which model to train one of:
#                 'LDA', 'QDA', 'GNB', 'NN', 'INN', 'BDT'. Defaults to 'LDA'.
#             experiment (int, optional): 0, 1, or 2.
#                 Exp1: uses "fdchi2", "sumpt"
#                 Exp2: uses "minipchi2", "vchi2"
#                 Exp3: uses "fdchi2", "sumpt", "minipchi2", "vchi2"
#                 Defaults to 0.
#             normalize (bool, optional): Whether to normalize the data
#                 between 0 and 1.
#                 Defaults to True.
#             data (str, optional): 'lhcb' or 'sim' data. Defaults to 'lhcb'.
#         """
#         self.model = model
#         self.normalize = normalize
#         self.data = data
#         self.experiment = experiment
#         self.output =  f"models/{model}_{features}_{data_type}_{normalized}.pkl"


class Configs:
    # list all possible configurations (cartesian product of these)
    model = ("regular", "sigma", "bdt")
    data_type = ("lhcb",)#, "standalone")
    features = (["minipchi2", "sumpt"], ["fdchi2", "sumpt"])
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
        project_root, "savepoints/gridX_{model}_{features}_{data_type}_{normalize}.npz"
    )
    grid_Y = join(
        project_root, "savepoints/gridY_{model}_{features}_{data_type}_{normalize}.npz"
    )
    # plots
    train_distribution_gif = join(
        project_root,
        "plots/training_distributions_{model}_{features}_{data_type}_{normalize}.gif",
    )


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
