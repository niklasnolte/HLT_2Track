from hlt2trk.utils import config
from hlt2trk.utils.config import Configs, Locations

rule all:
  input:
    expand(
      Locations.grid_Y,
      model=Configs.model,
      data_type=Configs.data_type,
      features=map(config.to_string_features,Configs.features),
      normalize=map(config.to_string_normalize,Configs.normalize),
    )

rule eval_on_grid:
  input:
    Locations.model,
    script = "scripts/nn/eval_network_on_grid.py"
  output:
    Locations.grid_Y,
    Locations.grid_X
  run:
    args = config.get_cli_args(wildcards)
    print(args)
    shell(f"python {input.script} {args}")

rule train:
  input:
    Locations.data,
    script = "scripts/nn/training.py"
  output:
    Locations.model
  run:
    args = config.get_cli_args(wildcards)
    print(args)
    shell(f"python {input.script} {args}")
