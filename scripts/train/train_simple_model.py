import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from hlt2trk.utils.config import Locations, format_location, Configuration
from hlt2trk.models import get_model
import pickle


def train_simple_model(
    cfg: Configuration,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
):
    breakpoint()
    assert cfg.model in ["lda", "qda", "gnb"]

    model = get_model(cfg)
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
