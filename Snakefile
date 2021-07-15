# type: ignore
from hlt2trk.utils import config, load_config
from hlt2trk.utils.config import Locations, dirs, Configs
from os import makedirs

for k, v in dirs.__dict__.items():
    if not k.startswith("__"):
        makedirs(v, exist_ok=True)


rule all:
    input:
        # feat_vs_output plots
        expand(
            Locations.feat_vs_output,
            model=Configs.model,
            data_type=Configs.data_type,
            features=map(config.to_string_features, Configs.features),
            normalize=map(config.to_string_normalize, Configs.normalize),
            signal_type=Configs.signal_type,
        ),
        # heatmap plots for 2d trainings
        expand(
            Locations.heatmap,
            model=Configs.model,
            data_type=Configs.data_type,
            features=map(
                config.to_string_features,
                [feats for feats in Configs.features if len(feats) == 2],
            ),
            normalize=map(config.to_string_normalize, Configs.normalize),
            signal_type=Configs.signal_type,
        ),
        # rates vs efficiencies, only for lhcb data
        expand(
            Locations.rate_vs_eff,
            model=Configs.model,
            data_type=["lhcb"], # only for lhcb data
            features=map(config.to_string_features, Configs.features),
            normalize=map(config.to_string_normalize, Configs.normalize),
            signal_type=Configs.signal_type,
        ),

rule plot_rate_vs_eff:
    input:
        Locations.data,
        Locations.model,
        Locations.presel_efficiencies,
        script="scripts/eval/eval_and_plot_score.py",
    output:
        Locations.rate_vs_eff,
    run:
        args = config.get_cli_args(wildcards)
        shell(f"python {input.script} {args}")



rule plot_heatmap:
    input:
        Locations.gridXY,
        script="scripts/plot/plot_heatmap.py",
    output:
        Locations.heatmap,
    wildcard_constraints:
        features="\w+\+\w+",
    run:
        args = config.get_cli_args(wildcards)
        shell(f"python {input.script} {args}")



rule plot_feat_vs_output:
    input:
        Locations.gridXY,
        script="scripts/plot/plot_feat_vs_output.py",
    output:
        Locations.feat_vs_output,
    run:
        args = config.get_cli_args(wildcards)
        shell(f"python {input.script} {args}")


rule eval_on_grid:
    input:
        Locations.model,
        script="scripts/eval/eval_network_on_grid.py",
    output:
        Locations.gridXY,
    run:
        args = config.get_cli_args(wildcards)
        shell(f"python {input.script} {args}")



def get_inputs_train(wildcards):
    if wildcards.model == "bdt":
        return "scripts/train/train_bdt_model.py"
    elif wildcards.model in ["regular", "sigma"]:
        return "scripts/train/train_torch_model.py"
    elif wildcards.model in ["lda", "qda", "gnb"]:
        return "scripts/train/train_simple_model.py"
    else:
        raise ValueError(f"Unknown model {wildcards.model}")


rule train:
    input:
        Locations.data,
        "hlt2trk/models/models.py",
        get_inputs_train,
        script="scripts/train/training.py",
    output:
        Locations.model,
    run:
        args = config.get_cli_args(wildcards)
        shell(f"python {input.script} {args}")



def get_script_preprocess(wildcards):
    if wildcards.data_type == "lhcb":
        return "scripts/preprocess/preprocess_lhcb_mc.py"
    elif wildcards.data_type == "standalone":
        return "scripts/preprocess/preprocess_standalone_mc.py"


rule preprocess:
    input:
        raw_data=dirs.raw_data,
        script=get_script_preprocess,
    output:
        Locations.data,
        Locations.presel_efficiencies
    run:
        args = config.get_cli_args(wildcards)
        shell(f"python {input.script} {args}")
