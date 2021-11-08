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
               seed=Configs.seed[:1],
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
            seed=Configs.seed[:1], # not for all seeds pls
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
            seed=Configs.seed[:1], # not for all seeds pls
        ),
        # heatmap plots aggregated
        expand_with_rules(
            Locations.heatmap_agg,
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
            seed=Configs.seed[:1], # not for all seeds pls
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
            seed=Configs.seed[:1], # not for all seeds pls
        ),
        # efficiency vs hadron kinematics, only for lhcb data
        expand_with_rules(
            Locations.eff_vs_kinematics,
            model=Configs.model,
            data_type=["lhcb"],  # only for lhcb data
            features=map(config.to_string_features, Configs.features),
            normalize=map(config.to_string_normalize, Configs.normalize),
            signal_type=Configs.signal_type,
            presel_conf=map(config.to_string_presel_conf, Configs.presel_conf),
            max_norm=map(config.to_string_max_norm, Configs.max_norm),
            regularization=Configs.regularization,
            division=Configs.division,
            seed=Configs.seed[:1], # not for all seeds pls
        ),
        # efficiency vs hadron kinematics for different models in one plot, only for lhcb data
        expand_with_rules(
            Locations.multi_eff_vs_kinematics,
            data_type=["lhcb"],  # only for lhcb data
            features=map(config.to_string_features, Configs.features),
            normalize=map(config.to_string_normalize, Configs.normalize),
            signal_type=Configs.signal_type,
            presel_conf=map(config.to_string_presel_conf, Configs.presel_conf),
            max_norm=map(config.to_string_max_norm, Configs.max_norm),
            regularization=Configs.regularization,
            division=Configs.division,
            seed=Configs.seed[:1], # not for all seeds pls
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
            seed=Configs.seed[:1], # not for all seeds pls
        ),
        # violin auc/acc with different seeds
        # takes ages, take care when doing this
        # expand_with_rules(
        #     Locations.seed_violins,
        #     data_type=["lhcb"],  # only for lhcb data
        #     features=map(config.to_string_features, Configs.features),
        #     normalize=map(config.to_string_normalize, Configs.normalize),
        #     signal_type=Configs.signal_type,
        #     presel_conf=map(config.to_string_presel_conf, Configs.presel_conf),
        #     max_norm=map(config.to_string_max_norm, Configs.max_norm),
        #     regularization=Configs.regularization,
        #     division=Configs.division,
        # ),
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
            seed=Configs.seed[:1], # not for all seeds pls
        ),
        # exported model
        expand_with_rules(
            Locations.exported_model,
            model=["nn-inf"], # --> nn-inf only for now
            data_type=["lhcb"],  # only for lhcb data
            features=map(config.to_string_features, Configs.features),
            normalize=map(config.to_string_normalize, Configs.normalize),
            signal_type=Configs.signal_type,
            presel_conf=map(config.to_string_presel_conf, Configs.presel_conf),
            max_norm=map(config.to_string_max_norm, Configs.max_norm),
            regularization=Configs.regularization,
            division=Configs.division,
            seed=Configs.seed[:1], # not for all seeds pls
        )

rule export_model_to_json:
    input:
        Locations.model,
        Locations.target_cut,
        script = "scripts/eval/export_model.py"
    output:
        Locations.exported_model
    run:
      args = config.get_cli_args(wildcards)
      shell(f"python {input.script} {args}")


rule plot_violins:
    input:
        expand(
            Locations.target_effs,
            model=Configs.model,
            allow_missing=True,
        ),
        expand(
            Locations.model,
            model=Configs.model,
            allow_missing=True,
        ),
        script = "scripts/plot/plot_violins.py"
    output:
        Locations.violins
    run:
      args = config.get_cli_args(wildcards)
      shell(f"python {input.script} {args}")

rule plot_seed_violins:
    input:
        expand(
            Locations.auc_acc,
            model=Configs.model,
            seed=Configs.seed,
            allow_missing=True,
        ),
        expand(
            Locations.model,
            model=Configs.model,
            seed=Configs.seed,
            allow_missing=True,
        ),
        script = "scripts/plot/plot_seed_violins.py"
    output:
        Locations.seed_violins
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
        Locations.target_cut,
        script = "scripts/plot/plot_rates_vs_effs.py",
    output:
        Locations.rate_vs_eff,
    run:
        args = config.get_cli_args(wildcards)
        shell(f"python {input.script} {args}")

rule plot_eff_vs_kinematics:
    input:
        Locations.data,
        Locations.target_cut,
        Locations.model,
        script = "scripts/plot/plot_eff_vs_kinematics.py",
    output:
        Locations.eff_vs_kinematics,
    run:
        args = config.get_cli_args(wildcards)
        shell(f"python {input.script} {args}")

rule plot_multi_eff_vs_kinematics:
    input:
        expand(
        Locations.target_cut,
        model=Configs.model,
        allow_missing=True,
        ),
        expand(
        Locations.model,
        model=Configs.model,
        allow_missing=True,
        ),
        Locations.data,
        script = "scripts/plot/plot_multi_eff_vs_kinematics.py",
    output:
        Locations.multi_eff_vs_kinematics,
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

rule plot_heatmap_agg:
    input:
        expand(
        Locations.auc_acc,
        model=Configs.model,
        allow_missing=True,
        ),
        expand(
        Locations.target_cut,
        model=Configs.model,
        allow_missing=True,
        ),
        expand(
        Locations.gridXY,
        model=Configs.model,
        allow_missing=True,
        ),
        script = "scripts/plot/plot_multi_heatmap.py",
    output:
        Locations.heatmap_agg,
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
