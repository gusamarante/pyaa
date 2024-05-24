"""
GMM Estimates
"""
import matplotlib.pyplot as plt
from scipy.stats import chi2, f
import statsmodels.api as sm
from getpass import getuser
import numpy.linalg as la
from pathlib import Path
import seaborn as sns
import pandas as pd
import numpy as np

# User parameters
file_path = Path(f"/Users/{getuser()}/Dropbox/Aulas/Doutorado - Empirical Finance/Project 1")
show_charts = True


# ================
# ===== Data =====
# ================
# --- read portfolios ---
ff25 = pd.read_excel(file_path.joinpath("Dados.xlsx"),
                     skiprows=2, header=[0, 1], index_col=0, sheet_name="FF25")
ff25.index = pd.to_datetime(ff25.index)
ff25 = ff25.resample('M').last().astype(float)
ff25.columns = [f"FF{(s - 1) * 5 + v}" for s, v in ff25.columns]

# --- read FF factors ---
ff5f = pd.read_excel(file_path.joinpath("Dados.xlsx"),
                     index_col=0, sheet_name="Factors")
ff5f.index = pd.to_datetime(ff5f.index)
ff5f = ff5f.resample('M').last().astype(float)

# --- Execess Returns of the FF25 ---
ff25 = ff25.sub(ff5f['RF'], axis=0)

# --- Read the VIX ---
vix = pd.read_excel(file_path.joinpath("dados vix.xlsx"), sheet_name="Values", index_col=0)
vix.index = pd.to_datetime(vix.index)
vix = vix['VIX Index'].dropna().rename('VIX')
vix = vix.resample('M').mean()

# --- Orgnize Factors and returns ---
factors = pd.concat(
    [
        ff5f.drop(['RF', 'SMB', 'HML', 'RMW', 'CMA'], axis=1),
        vix,  # Comment out this line to remove VIX
    ],
    axis=1,
)
factors = factors.dropna().astype(float)
factors = factors - factors.mean()  # Demean factors
factors = factors[factors.index >= "1990-01-01"]


# --- Summary Statistics ---
means = ff25.reindex(factors.index).mean()
fmeans = factors.reindex(factors.index).mean()


# =====================================
# ===== GMM Estimates - VIX Model =====
# =====================================
X = pd.concat([ff25, factors], axis=1)
X = X.dropna()
T = X.shape[0]
varX = X.cov()
k = X.shape[1] - 25  # Number of factors

# --- 1st Stage ---
d = varX.iloc[:25, -k:].values
b1_hat = la.inv(d.T @ d) @ (d.T @ means.values.reshape((-1, 1)))
print(b1_hat)

# --- 1st stage residuals ---
m1 = 1 - factors.values @ b1_hat
u1 = X.iloc[:, :25] * m1 - 1
S1 = u1.cov()

# --- 2nd Stage ---
invS = la.inv(S1)
b2_hat = la.inv(d.T @ invS @ d) @ (d.T @ invS @ means.values.reshape((-1, 1)))
var_b2 = (1 / T) * la.inv(d.T @ invS @ d)

print(b2_hat)

# --- 2nd stage residuals ---
m2 = 1 - factors.values @ b2_hat
u2 = X.iloc[:, :25] * m2 - 1
S2 = u2.cov()  # we can update S

# --- Test if all moments have been reached ---
gT = u2.mean()
test_stat = T * gT.values @ invS @ gT.values
dof = 25 - k
pval = 1 - chi2.cdf(test_stat, dof)

print(f"Test Stat = {test_stat}")
print(f"pval = {pval}")
