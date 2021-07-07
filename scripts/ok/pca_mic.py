import numpy as np
import pandas as pd
from hlt2trk.data.meta_info import get_data_for_training, features
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sys import argv


LATEX = "--latex" in argv

x_train, y_train, x_val, y_val = get_data_for_training(normalize=True)
X = np.concatenate([x_train, x_val], axis=0)
y = np.concatenate([y_train, y_val], axis=0)


pca = PCA(4)
fit = pca.fit(X)
pca_var = (
    (pca.components_) ** 2 * pca.explained_variance_ratio_.reshape(-1, 1)
).sum(axis=0)
mic = mutual_info_classif(X, y.reshape(-1))


df = pd.DataFrame(np.stack([pca_var, mic]))
df.columns = features
df.index = ["Explained Variance", "Mutual Information"]

if not LATEX:
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None
    ):  # more options can be specified also
        print(df)
else:
    print(
        df.to_latex(
            column_format="c" * (len(df.columns) + 1),
            bold_rows=True,
            float_format="%.3f",
            multicolumn_format="c",
            caption="caption",
        )
    )
