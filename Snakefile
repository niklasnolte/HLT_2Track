# type: ignore
from hlt2trk.utils import config
from hlt2trk.utils.config import Locations, dirs, Configs, expand_with_rules
from os import makedirs

for k, v in dirs.__dict__.items():
    if not k.startswith("__"):
        makedirs(v, exist_ok=True)


rule all:
    input:
        # eval on validation data
        expand_with_rules(Locations.auc_acc,
               model=Configs.model,
               data_type=Configs.data_type,
               features=map(config.to_string_features, Configs.features),
               normalize=map(config.to_string_normalize, Configs.normalize),
               signal_type=Configs.signal_type,
               presel_conf=map(config.to_string_presel_conf, Configs.presel_conf),
               max_norm=map(config.to_string_max_norm, Configs.max_norm),
               regularization=Configs.regularization,
               division=Configs.division,
               ),
        # feat_vs_output plots
        expand_with_rules(
            Locations.feat_vs_output,
            model=Configs.model,
            data_type=Configs.data_type,
            features=map(config.to_string_features, Configs.features),
            normalize=map(config.to_string_normalize, Configs.normalize),
            signal_type=Configs.signal_type,
            presel_conf=map(config.to_string_presel_conf, Configs.presel_conf),
            max_norm=map(config.to_string_max_norm, Configs.max_norm),
            regularization=Configs.regularization,
            division=Configs.division,
        ),
        # heatmap plots
        expand_with_rules(
            Locations.heatmap,
            model=Configs.model,
            data_type=Configs.data_type,
            features=map(
                config.to_string_features,
                Configs.features,
            ),
            normalize=map(config.to_string_normalize, Configs.normalize),
            signal_type=Configs.signal_type,
            presel_conf=map(config.to_string_presel_conf, Configs.presel_conf),
            max_norm=map(config.to_string_max_norm, Configs.max_norm),
            regularization=Configs.regularization,
            division=Configs.division,
        ),
        # rates vs efficiencies, only for lhcb data
        expand_with_rules(
            Locations.rate_vs_eff,
            model=Configs.model,
            data_type=["lhcb"],  # only for lhcb data
            features=map(config.to_string_features, Configs.features),
            normalize=map(config.to_string_normalize, Configs.normalize),
            signal_type=Configs.signal_type,
            presel_conf=map(config.to_string_presel_conf, Configs.presel_conf),
            max_norm=map(config.to_string_max_norm, Configs.max_norm),
            regularization=Configs.regularization,
            division=Configs.division,
        ),
        # violin plots
        expand_with_rules(
            Locations.violins,
            data_type=["lhcb"],  # only for lhcb data
            features=map(config.to_string_features, Configs.features),
            normalize=map(config.to_string_normalize, Configs.normalize),
            signal_type=Configs.signal_type,
            presel_conf=map(config.to_string_presel_conf, Configs.presel_conf),
            max_norm=map(config.to_string_max_norm, Configs.max_norm),
            regularization=Configs.regularization,
            division=Configs.division,
        ),
        # efficiency tables for all models
        expand_with_rules(
            Locations.eff_table,
            data_type=["lhcb"],  # only for lhcb data
            features=map(config.to_string_features, Configs.features),
            normalize=map(config.to_string_normalize, Configs.normalize),
            signal_type=Configs.signal_type,
            presel_conf=map(config.to_string_presel_conf, Configs.presel_conf),
            max_norm=map(config.to_string_max_norm, Configs.max_norm),
            regularization=Configs.regularization,
            division=Configs.division,
        ),

rule plot_violins:
    input:
        expand(
            Locations.target_effs,
            model=Configs.model,
            allow_missing=True,
        ),
        script = "scripts/plot/plot_violins.py"
    output:
        Locations.violins
    run:
      args = config.get_cli_args(wildcards)
      shell(f"python {input.script} {args}")

rule build_eff_table:
    input:
        expand(
            Locations.target_effs,
            model=Configs.model,
            allow_missing=True,
        ),
        script = "scripts/eval/merge_table.py"
    output:
        Locations.eff_table
    run:
      args = config.get_cli_args(wildcards)
      shell(f"python {input.script} {args}")

rule plot_rate_vs_eff:
    input:
        Locations.data,
        Locations.model,
        Locations.presel_efficiencies,
        Locations.full_effs,
        Locations.target_effs,
        script = "scripts/plot/plot_rates_vs_effs.py",
    output:
        Locations.rate_vs_eff,
    run:
        args = config.get_cli_args(wildcards)
        shell(f"python {input.script} {args}")


rule plot_heatmap:
    input:
        Locations.auc_acc,
        Locations.target_cut,
        Locations.gridXY,
        script = "scripts/plot/plot_heatmap.py",
    output:
        Locations.heatmap,
    run:
        args = config.get_cli_args(wildcards)
        shell(f"python {input.script} {args}")


rule plot_feat_vs_output:
    input:
        Locations.gridXY,
        script = "scripts/plot/plot_feat_vs_output.py",
    output:
        Locations.feat_vs_output,
    run:
        args = config.get_cli_args(wildcards)
        shell(f"python {input.script} {args}")


rule eval_on_grid:
    input:
        Locations.model,
        script = "scripts/eval/eval_on_grid.py",
    output:
        Locations.gridXY,
    run:
        args = config.get_cli_args(wildcards)
        shell(f"python {input.script} {args}")

rule eval_on_train_metrics:
    input:
        Locations.model,
        script = "scripts/eval/eval_on_train_metrics.py",
    output:
        Locations.auc_acc,
    run:
        args = config.get_cli_args(wildcards)
        shell(f"python {input.script} {args}")

rule eval_on_data:
    input:
        Locations.model,
        script = "scripts/eval/eval_on_data.py",
    output:
        Locations.full_effs,
        Locations.target_effs,
        Locations.target_cut
    run:
        args = config.get_cli_args(wildcards)
        shell(f"python {input.script} {args}")


def get_inputs_train(wildcards):
    if wildcards.model == "bdt":
        return "scripts/train/train_bdt_model.py"
    elif wildcards.model.startswith("nn"):
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
        script = "scripts/train/training.py",
    output:
        Locations.model,
    run:
        args = config.get_cli_args(wildcards)
        shell(f"python {input.script} {args}")


rule preprocess_lhcb:
    input:
        raw_data = dirs.raw_data,
        script = "scripts/preprocess/preprocess_lhcb_mc.py"
    output:
        # format doesn't work because there are other placeholders to be filled
        Locations.data.replace("{data_type}", "lhcb"),
        Locations.presel_efficiencies.replace("{data_type}", "lhcb"),
    run:
        args = config.get_cli_args(wildcards)
        shell(f"python {input.script} --data_type=lhcb {args}")


rule preprocess_standalone:
    input:
        raw_data = dirs.raw_data,
        script = "scripts/preprocess/preprocess_standalone_mc.py"
    output:
        Locations.data.replace("{data_type}", "standalone"),
    run:
        args = config.get_cli_args(wildcards)
        shell(f"python {input.script} --data_type=standalone {args}")
