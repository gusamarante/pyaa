"""
This has only the last item of question 3
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import statsmodels.api as sm
from scipy.stats import chi2, f
from getpass import getuser
import numpy as np
import numpy.linalg as la
from statsmodels.regression.rolling import RollingOLS

# User parameters
file_path = Path(f"/Users/{getuser()}/Dropbox/Aulas/Doutorado - Empirical Finance/Project 1")
show_charts = False


# ================
# ===== Data =====
# ================
# --- read portfolios ---
ff25 = pd.read_excel(file_path.joinpath("Dados.xlsx"),
                     skiprows=2, header=[0, 1], index_col=0, sheet_name="FF25")
ff25.index = pd.to_datetime(ff25.index)

# --- read factors ---
ff5f = pd.read_excel(file_path.joinpath("Dados.xlsx"),
                     index_col=0, sheet_name="Factors")
ff5f.index = pd.to_datetime(ff5f.index)

# --- Execess Returns of the FF25 ---
ff25 = ff25.sub(ff5f['RF'], axis=0)


# ==================================
# ===== Fama-MacBeth Procedure =====
# ==================================
# --- First Stage - The 25 Rolling 5y Timeseries Regressions ---
betas = pd.DataFrame(columns=ff25.columns, index=ff25.index)

for s in range(1, 6):
    for v in range(1, 6):
        reg_data = pd.concat([ff25[s][v].rename('Y'), ff5f["Mkt"].rename('X')], axis=1)
        reg_data = reg_data.dropna()

        Y = reg_data["Y"]
        X = sm.add_constant(reg_data["X"])
        model = RollingOLS(Y, X, window=5*12)
        res = model.fit()

        betas.loc[:, (s, v)] = res.params["X"]

betas = betas.dropna(how='all')


# --- Second Stage - One cross-sectional regression for each period ---
params = pd.DataFrame()
for tt in betas.index:

    Y = ff25.loc[tt]
    X = sm.add_constant(betas.loc[tt].rename('lambda'))
    X_noc = betas.loc[tt].rename('lambda')

    model = sm.OLS(Y, X)
    model_noc = sm.OLS(Y, X_noc)
    res = model.fit()
    res_noc = model_noc.fit()

    params.loc[tt, "const"] = res.params.loc['const']
    params.loc[tt, "lambda"] = res.params.loc['lambda']
    params.loc[tt, "lambda_noc"] = res_noc.params.loc['lambda']

print(params.mean())
print(params.corr())


# ===================================
# ===== Chart - Beta timeseries =====
# ===================================
fig = plt.figure(figsize=(5 * (16 / 9), 5))

ax = plt.subplot2grid((1, 1), (0, 0))
ax.plot(betas.dropna())

ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig(file_path.joinpath("figures/Q03 Rolling Betas.pdf"))
if show_charts:
    plt.show()
plt.close()


# ===================================
# ===== Chart - Beta timeseries =====
# ===================================
fig = plt.figure(figsize=(5 * (16 / 9), 5))

ax = plt.subplot2grid((1, 1), (0, 0))
ax.plot(params['lambda'].dropna(), label='Lambda')
ax.plot(params['lambda_noc'].dropna(), label='Lambda (No Constant)')

ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(frameon=True)

plt.tight_layout()
plt.savefig(file_path.joinpath("figures/Q03 Rolling lambdas.pdf"))
if True:
    plt.show()
plt.close()

