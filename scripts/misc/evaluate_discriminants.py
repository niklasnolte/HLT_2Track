from hlt2trk.utils.config import Configuration, dirs, get_config_from_file, Configs
import pickle
from collections import defaultdict
from os.path import join

import fire
import numpy as np
import pandas as pd
from hlt2trk.utils.data import get_data_for_training
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB


def get_metrics(x_train, y_train, x_val, y_val, model):
    model.fit(x_train, y_train.reshape(-1))
    preds = model.predict(x_val)
    probs = model.predict_proba(x_val)
    acc = max(
        [
            balanced_accuracy_score(y_val.reshape(-1), probs[:, 1] > cut)
            for cut in np.linspace(0, 1, 100)
        ]
    )
    auc = roc_auc_score(y_val.reshape(-1), probs[:, 1])
    return acc, auc


def main(save_model: bool = True, save_path: str = None, latex: bool = False):
    """Generate LDA, QDA, and GNB models and train them on 3 subsets of the
    features in the data (experiments).
    Exp1: uses "fdchi2", "sumpt"
    Exp2: uses "minipchi2", "vchi2"
    Exp3: uses all 4
    The models can be saved and a latex table could be printed.

    Args:
        save_model (bool, optional): Whether to save trained models.
        If no save_path is specified saves to dirs.models.
        Defaults to True.
        save_path (str, optional): Directory where to save models. If nothing
        is passed no model is saved. Defaults to None.
        latex (bool, optional): Whether to print a latex table to console.
         Defaults to False.
    """
    model_results = defaultdict(list)
    model_names = ["LinearDiscriminantAnalysis",
                   "QuadraticDiscriminantAnalysis", "GaussianNB", ]
    for model_name in model_names:
        model = eval(model_name + "()")
        print(model_name)
        cfg = get_config_from_file(join(dirs.project_root, "config.yml"))
        cfg.__dict__.pop('features')
        cfg.__dict__.pop('device')
        for i, features in enumerate(Configs.features):
            acc, auc = get_metrics(
                *get_data_for_training(Configuration(**cfg.__dict__, features=features)),
                model)
            print(f"{acc:.3f}")
            print(f"{auc:.3f}")
            model_results[model_name].append([acc, auc])
            if save_model:
                if save_path is None:
                    save_path = dirs.models
                file_name = join(save_path, model_name + f"_{i}.pkl")
                with open(file_name, "wb") as f:
                    pickle.dump(model, f)
                    print(f"saved to {file_name}")

    print("\nDone training and evaluating!\n")
    df = pd.DataFrame(np.stack(list(model_results.values())).reshape(3, -1))

    df.columns = np.concatenate([[[f"{i}", "acc"],
                                [f"{i}", "auc"]]
        for i in range(len(df.columns) // 2)])

    df.columns = pd.MultiIndex.from_tuples(
        df.columns, names=["Experiment", "metric"]
    )
    df.index = ["LDA", "QDA", "GNB"]
    if latex:
        print(
            df.to_latex(
                column_format="c" * 7,
                bold_rows=True,
                float_format="%.3f",
                multicolumn_format="c",
                caption="1: uses minipchi2/vchi2, 2: uses fdchi2/sumpt, \
                        3: uses all four features.",
            )
        )

    else:
        with pd.option_context(
            "display.max_rows", None, "display.max_columns", None
        ):  # more options can be specified also
            print(df)
    return


if __name__ == "__main__":
    fire.Fire(main)
