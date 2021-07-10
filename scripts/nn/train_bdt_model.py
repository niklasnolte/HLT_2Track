from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
plt.style.use("seaborn")
from hlt2trk.utils.config import Locations, format_location, Configuration
from hlt2trk.models import get_model


def train_bdt_model(
    cfg: Configuration,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
):
  assert cfg.model == "bdt"

  clf = get_model(cfg)

  clf.fit(
      x_train,
      y_train,
      eval_set=(x_val, y_val),
      early_stopping_rounds=15,
      eval_metric=["logloss", "roc", "accuracy"],
      verbose=False,
      callbacks=[lambda x : print(f"{x.iteration}/300", end="\r")]
  )

  # evalutation
  preds = clf.predict_proba(x_val)[:, 1]

  clf.booster_.save_model(format_location(Locations.model, cfg))

  print((y_val == 1).mean())
  auc = roc_auc_score(y_val, preds)
  acc = max(balanced_accuracy_score(y_val, preds > i) for i in np.linspace(0, 1, 100))
  print(f"roc: {auc:.6f}, acc: {acc:.6f}")
