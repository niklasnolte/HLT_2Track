from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import lightgbm as lgb

plt.style.use("seaborn")

import hlt2trk.utils.meta_info as meta

X_train, Y_train, X_test, Y_test = meta.get_data_for_training()

print(f"mean label: {Y_train.mean()}")


clf = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    is_unbalance=True,
    num_leaves=15,
    boosting_type="gbdt",
)

clf.fit(
    X_train,
    Y_train,
    eval_set=(X_test, Y_test),
    early_stopping_rounds=15,
    eval_metric=["logloss", "roc", "accuracy"],
    verbose=False,
    callbacks=[lambda x : print(f"{x.iteration}/300", end="\r")]
)

# evalutation
preds = clf.predict_proba(X_test)[:, 1]

auc = roc_auc_score(Y_test, preds)
acc = max(balanced_accuracy_score(Y_test, preds > i) for i in np.linspace(0, 1, 100))
print(f"roc: {auc:.6f}, acc: {acc:.6f}")
