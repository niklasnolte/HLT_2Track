import pandas as pd
from typing import Iterable
import numpy as np


def to_np(x: pd.DataFrame, features: Iterable) -> np.ndarray:
    """Convert dataframe to numpy array given a subset of the columns.

    Args:
        x (pd.DataFrame): Dataframe to convert.
        features (Iterable): Iterable of columns to keep.

    Returns:
        np.ndarray: Array of columns to keep.
    """
    return x[features].values
