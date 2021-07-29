from functools import partial
import pickle
from typing import Callable, Iterable, Union

import lightgbm as lgb
import torch
from hlt2trk.utils import config
from InfinityNorm import SigmaNet, project_norm, divide_norm, GroupSort
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.naive_bayes import GaussianNB
from torch import nn


def build_module(
    in_features: int = 1,
    nclasses: int = 1,
    nunits: Union[int, Iterable] = 15,
    biases: Union[bool, Iterable] = True,
    nlayers: int = 3,
    layer: Callable = nn.Linear,
    norm: Callable = None,
    activation: Callable = nn.LeakyReLU(),
    out_activation: Callable = None,
    bn_layer: int = None,
) -> nn.Module:
    """

  Parameters
  ----------
  in_features : int
  nclasses : int
  nunits : [int, Iterable]
      If int is provided, use nlayers to specify depth. Otherwise,
      if Iterable, the depth is inferred.
  nlayers : int
      Used to specify depth when nunits is int. Ignored if nunits is passed.
  layer : Callable
      Layer type for example: nn.Linear.
  bn_layer : int
      Integer between 1 and nlayers. Used to specify where batchnorm would be inserted.
  norm : Callable
      Function used for norming the layers.
  activation : Callable
      Activation used between hidden layers.
  out_activation : Callable
      Activation used in the last layer.
  Returns
  -------
  nn.Module
      Sequential module.
  """
    norm_func = norm if norm is not None else lambda x: x
    assert(isinstance(norm_func, Callable)), f"norm: {norm} is not callable."

    nunits = to_iter(nunits, nlayers - 1)
    biases = to_iter(biases, nlayers)

    layers = []
    for out_features, bias in zip(nunits, biases):
        layers.append(norm_func(layer(in_features, out_features, bias)))
        layers.append(activation)
        in_features = out_features
    layers.append(norm_func(layer(in_features, nclasses, biases[-1])))
    if bn_layer is not None:
        layers.insert(bn_layer, nn.BatchNorm1d(nunits[bn_layer - 1]))
    if out_activation is not None:
        layers.append(out_activation)
    return nn.Sequential(*layers)


def to_iter(nunits, nlayers):
    nunits = [nunits] * (nlayers) if isinstance(nunits, int) else nunits
    if not isinstance(nunits, Iterable):
        raise TypeError("nunits must be either int or iterable.")
    if len(nunits) != nlayers:
        raise ValueError(
            f"property list has len {len(nunits)} while nlayers is {nlayers}."
            "Check nunits or biases.")
    return nunits


def get_model(cfg: config.Configuration) -> Union[nn.Module, lgb.Booster]:
    nfeatures = len(cfg.features)

    if cfg.model.endswith(("one", "inf")):
        be_monotonic_in = list(range(len(cfg.features)))
        try:
            be_monotonic_in.pop(cfg.features.index("vchi2"))
        except ValueError:
            # if vchi2 is not in there, we don't need to remove it
            pass

        class Sigma(nn.Module):
            def __init__(self, sigma):
                super().__init__()
                depth = 3
                norm_cfg = dict(
                  always_norm = not cfg.max_norm,
                  alpha = sigma**(1 / depth),
                  vectorwise = "vector" == cfg.division
                )
                if cfg.regularization == "direct":
                    normfunc = partial(divide_norm, **norm_cfg)
                elif cfg.regularization == "project":
                    normfunc = partial(project_norm, **norm_cfg)
                self.sigmanet = SigmaNet(
                    build_module(nlayers=depth,
                                 in_features=nfeatures,
                                 norm=normfunc,
                                 # nn.ReLU(),  # GroupSort(num_units=1),
                                 activation=GroupSort(num_units=1)
                                 ),
                    sigma=sigma, monotonic_in=be_monotonic_in,
                    nfeatures=nfeatures)

            def forward(self, x):
                x = self.sigmanet(x)
                # x = torch.sigmoid(x)
                return x

        sigma = cfg.sigma_init if cfg.sigma_init is not None else 1
        sigma_network = Sigma(sigma=sigma)
        return sigma_network

    elif cfg.model.startswith("nn"):
        # regular full dim model for comparison
        regular_model = torch.nn.Sequential(
            build_module(
                in_features=nfeatures, norm=None,
                activation=GroupSort(num_units=1)),
            nn.Sigmoid(),)
        return regular_model

    elif cfg.model == "bdt":
        clf = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            is_unbalance=True,
            num_leaves=15,
            boosting_type="gbdt",
        )
        return clf

    elif cfg.model == "lda":
        model = LinearDiscriminantAnalysis()
        return model

    elif cfg.model == "qda":
        model = QuadraticDiscriminantAnalysis()
        return model

    elif cfg.model == 'gnb':
        model = GaussianNB()
        return model


def load_model(cfg: config.Configuration) -> Union[nn.Module, lgb.Booster,
                                                   LinearDiscriminantAnalysis,
                                                   QuadraticDiscriminantAnalysis,
                                                   GaussianNB]:
    location = config.format_location(config.Locations.model, cfg)
    if cfg.model.startswith("nn"):
        m = get_model(cfg)
        m.load_state_dict(torch.load(location))
    elif cfg.model == 'bdt':
        m = lgb.Booster(model_file=location)
    else:
        with open(location, 'rb') as f:
            m = pickle.load(f)
    return m
