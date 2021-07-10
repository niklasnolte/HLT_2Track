from hlt2trk.utils import config
from hlt2trk.utils.config import Configs, Locations

rule all:
  input:
    expand(
      Locations.model,
      model=Configs.model,
      data_type=Configs.data_type,
      features=map(config.to_string_features,Configs.features),
      normalize=map(config.to_string_normalize,Configs.normalize),
    )

rule train:
  input:
    Locations.data,
    script = "scripts/nn/2track.py"
  output:
    Locations.model
  run:
    args = config.get_cli_args(wildcards)
    print(args)
    shell(f"python {input.script} {args}")
