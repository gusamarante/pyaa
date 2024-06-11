"""
MAXIMUM SHARPE OF THE PORTFOLIO OF PODS
as a function of the number of pods and average correlation
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

sharpe = 0.5

max_sharpe = pd.DataFrame()
for rho in tqdm([0.01, 0.1, 0.5, 0.9]):
    for n in range(1, 101):
        corr_mat = np.eye(n) * (1 - rho) + np.ones((n, n)) * rho
        inv_c = np.linalg.inv(corr_mat)

        iota = np.ones(n)
        max_sharpe.loc[n, rho] = sharpe * np.sqrt(iota @ inv_c @ iota)

max_sharpe.to_clipboard()

# TODO chart this