from typing import Union, Iterable, Callable

import torch
from torch import nn

from hlt2trk.utils import config
from InfinityNorm import infnorm
from InfinityNorm import SigmaNet


def get_model(cfg):
    nfeatures = len(cfg.features)

    def build_module(
        in_features: int = 1,
        nclasses: int = 1,
        nunits: Union[int, Iterable] = 15,
        nlayers: int = 3,
        norm: bool = True,
        activation: Callable = nn.LeakyReLU(),
        out_activation: Callable = None,
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
          Used to specify depth when nunits is int. Ignored if nunits is
          Iterable.
      norm : bool
          Whether or not to use Inf norm. If False a simple nn.Linear is used.
      activation : Callable
          Activation used between hidden layers.
      out_activation : Callable
          Activation used in the last layer.
      Returns
      -------
      nn.Module
          Sequential module.
      """
        norm_func = infnorm if norm else lambda x: x
        nunits = [nunits] * (nlayers - 1) if isinstance(nunits, int) else nunits
        try:
            nunits = iter(nunits)
        except TypeError:
            raise TypeError("nunits must be either int or iterable.")
        layers = []
        for out_features in nunits:
            layers.append(norm_func(nn.Linear(in_features, out_features)))
            layers.append(activation)
            in_features = out_features
        layers.append(norm_func(nn.Linear(in_features, nclasses)))
        if out_activation is not None:
            layers.append(out_activation)
        return nn.Sequential(*layers)

    if cfg.model == "sigma":
        be_monotonic_in = list(range(len(cfg.features)))
        try:
            be_monotonic_in.pop(cfg.features.index("vchi2"))
        except ValueError:
            # if vchi2 is not in there, we don't need to remove it
            pass

        sigma_network = nn.Sequential(
            SigmaNet(
                build_module(in_features=nfeatures, norm=True),
                sigma=1.5,
                monotonic_in=be_monotonic_in,
                nfeatures=nfeatures,
            ),
            nn.Sigmoid(),
        )
        return sigma_network

    elif cfg.model == "regular":
        # regular full dim model for comparison
        regular_model = torch.nn.Sequential(
            build_module(in_features=nfeatures, norm=False), nn.Sigmoid(),
        )
        return regular_model


def load_model(cfg):
    location = config.format_location(config.Locations.model, cfg)
    m = get_model(cfg)
    m.load_state_dict(torch.load(location))
    return m
