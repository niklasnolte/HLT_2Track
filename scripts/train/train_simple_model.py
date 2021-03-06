import pickle
from typing import Union

import numpy as np
from hlt2trk.models import get_model
from hlt2trk.utils.config import Configuration, Locations, format_location
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB


def train_simple_model(
    cfg: Configuration,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
):
    assert cfg.model in ["lda", "qda", "gnb"]

    model: Union[QuadraticDiscriminantAnalysis,
                 LinearDiscriminantAnalysis, GaussianNB] = get_model(cfg)
    model.fit(x_train, y_train.reshape(-1))

    file_name = format_location(Locations.model, cfg)
    with open(file_name, "wb") as f:
        pickle.dump(model, f)
        print(f"saved to {file_name}")

    # evalutation
    probs = model.predict_proba(x_val)
    acc = max(
        [
            balanced_accuracy_score(y_val.reshape(-1), probs[:, 1] > cut)
            for cut in np.linspace(0, 1, 100)
        ]
    )
    auc = roc_auc_score(y_val.reshape(-1), probs[:, 1])

    print(f"roc: {auc:.6f}, acc: {acc:.6f}")
    return model
