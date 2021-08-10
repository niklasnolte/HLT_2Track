from os.path import join
from hlt2trk.utils.config import Locations
import pandas as pd
import numpy as np



def make_table(df):
    df
    table = df.to_latex(
        column_format="c" * 7,
        bold_rows=True,
        float_format="%.3f",
        multicolumn_format="c",
        caption="efficiencies",
        index=False,
        escape=False,
    )
    return table
