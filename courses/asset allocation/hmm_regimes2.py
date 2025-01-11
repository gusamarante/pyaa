import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import torch
from pomegranate.distributions import Normal
from pomegranate.hmm import DenseHMM


np.random.seed(666)  # TODO test without the seed

file_path = "/Users/gustavoamarante/Library/CloudStorage/Dropbox/NAVs.xlsx"

df = pd.read_excel(file_path, index_col=0)
df.index = pd.to_datetime(df.index)

rets = df.resample("ME").last().pct_change(1).dropna()

hmm = DenseHMM(
    distributions=[
        Normal(),
        Normal(),
        Normal(),
        Normal(),
    ],
)

hmm.fit(X=np.array([rets.values]))

a = 1
