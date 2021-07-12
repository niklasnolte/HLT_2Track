# type: ignore
from hlt2trk.utils import config
from hlt2trk.utils.config import Configs, Locations

rule all:
    input:
        # feat_vs_output plots
        expand(
            Locations.feat_vs_output,
            model=Configs.model,
            data_type=Configs.data_type,
            features=map(config.to_string_features, Configs.features),
            normalize=map(config.to_string_normalize, Configs.normalize),
        ),
        # heatmap plots for 2d trainings
        expand(Locations.heatmap, model=Configs.model,
               data_type=Configs.data_type,
               features=map(
                   config.to_string_features,
                   [feats for feats in Configs.features if len(feats) == 2]),
               normalize=map(config.to_string_normalize, Configs.normalize),)

rule plot_heatmap:
    input:
        Locations.gridXY,
        script = "scripts/nn/plot_heatmap.py"
    output:
        Locations.heatmap
    wildcard_constraints:
        features = "\w+\+\w+"
    run:
        args = config.get_cli_args(wildcards)
        shell(f"python {input.script} {args}")

rule plot_feat_vs_output:
    input:
        Locations.gridXY,
        script = "scripts/nn/plot_feat_vs_output.py"
    output:
        Locations.feat_vs_output
    run:
        args = config.get_cli_args(wildcards)
        shell(f"python {input.script} {args}")

rule eval_on_grid:
    input:
        Locations.model,
        script = "scripts/nn/eval_network_on_grid.py"
    output:
        Locations.gridXY,
    run:
        args = config.get_cli_args(wildcards)
        shell(f"python {input.script} {args}")


def get_inputs_train(wildcards):
    if wildcards.model == "bdt":
        return "scripts/nn/train_bdt_model.py"
    elif wildcards.model in ["regular", "sigma"]:
        return "scripts/nn/train_torch_model.py"
    else:
        return "scripts/nn/train_simple_model.py"


rule train:
    input:
        Locations.data,
        "hlt2trk/models/models.py",
        get_inputs_train,
        script = "scripts/nn/training.py"
    output:
        Locations.model
    run:
        args = config.get_cli_args(wildcards)
        shell(f"python {input.script} {args}")
