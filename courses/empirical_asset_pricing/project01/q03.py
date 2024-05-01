import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import statsmodels.api as sm
from scipy.stats import chi2, f
from getpass import getuser
import numpy as np
import numpy.linalg as la

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

# --- read factors ---
ff5f = pd.read_excel(file_path.joinpath("Dados.xlsx"),
                     index_col=0, sheet_name="Factors")
ff5f.index = pd.to_datetime(ff5f.index)

# --- Execess Returns of the FF25 ---
ff25 = ff25.sub(ff5f['RF'], axis=0)

# --- summary statistics ---
means = ff25.mean()
fmeans = ff5f.mean()


# =======================================================
# ===== First Stage - The 25 Timeseries Regressions =====
# =======================================================
betas = pd.Series(index=means.index, name='betas')
df_resids = pd.DataFrame()

for s in range(1, 6):
    for v in range(1, 6):
        reg_data = pd.concat([ff25[s][v].rename('Y'), ff5f["Mkt"].rename('X')], axis=1)
        reg_data = reg_data.dropna()

        Y = reg_data["Y"]
        X = sm.add_constant(reg_data["X"])
        model = sm.OLS(Y, X)
        res = model.fit()

        betas.loc[s, v] = res.params["X"]
        df_resids[f"S{s}V{v}"] = res.resid


# =======================================================
# ===== Second Stage - The Cross-Section Regression =====
# =======================================================
Sig = df_resids.cov()
df2nd = []
for add_cons in [True, False]:
    for estimator in ['OLS', 'GLS']:
        for test_assets in [False, True]:

            res = pd.Series({'Add Const': add_cons,
                             'Estimator': estimator,
                             'Include TA': test_assets})

            Y = means.copy()
            X = betas.copy()

            # Inclusion of test asset
            S = Sig.copy()
            if test_assets:
                X.loc['TA', 'Mkt'] = 1
                Y.loc['TA', 'Mkt'] = fmeans.loc['Mkt']
                S.loc['Mkt'] = 0
                S['Mkt'] = 0
                S.loc['Mkt', 'Mkt'] = 0.0000001

            # Add constant or not
            if add_cons:
                X = sm.add_constant(X)
            else:
                X = X.to_frame()

            # Estimator
            X = X.values
            Y = Y.values
            S = S.values
            if estimator == 'OLS':
                lambda_hat = la.inv(X.T @ X) @ (X.T @ Y)
            else:
                lambda_hat = la.inv(X.T @ la.inv(S) @ X) @ (X.T @ la.inv(S) @ Y)

            cols = ["const", "betas"] if add_cons else ["betas"]
            lambda_hat = pd.Series(index=cols, data=lambda_hat)

            # --- Save Results ---
            res.loc['lambda'] = lambda_hat.loc["betas"]

            try:
                res.loc['const'] = lambda_hat.loc["const"]
            except KeyError:
                res.loc['const'] = 0

            df2nd.append(res)

df2nd = pd.concat(df2nd, axis=1).T

print(df2nd)


# TODO CS regression
# TODO Test